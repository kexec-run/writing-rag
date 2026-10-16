"""Build evidence-backed, empirically ranked reasoning-chain objects."""

from __future__ import annotations

import argparse
import re
from typing import Any

from writing_rag.ingest.causal_probability import (
    build_edge_calibration,
    score_candidate_chain,
    score_candidate_edge,
    sentence_units,
)
from writing_rag.runtime.pipeline_common import count_tokens, read_json, write_json


SCORE_VERSION = "empirical-chain-v2"
SIGNAL_TERMS = {
    "because",
    "therefore",
    "so",
    "but",
    "however",
    "means",
    "issue",
    "problem",
    "reason",
    "why",
    "need",
    "should",
    "must",
    "prefer",
    "idea",
    "system",
    "architecture",
    "retrieval",
    "writing",
    "memory",
    "learn",
    "understand",
    "implies",
    "then",
}
CONNECTOR_RE = re.compile(
    r"\b(because|so|but|therefore|then|however|which means|leads to|hence)\b",
    re.IGNORECASE,
)
TOPIC_TERMS = {
    "rag": {"rag", "retrieval", "embedding", "chroma", "vector"},
    "writing_systems": {"writing", "article", "style", "draft", "writer"},
    "memory_architecture": {"memory", "archive", "knowledge", "reasoning", "chain"},
    "software": {"code", "software", "database", "api", "system", "architecture"},
    "relationships": {"friend", "friendship", "loneliness", "people", "social", "emotion"},
    "self_reflection": {"feel", "think", "believe", "prefer", "understand", "learn"},
}


def short_text(text: str, limit: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(count_tokens(message["content"]) for message in messages)


def explicit_claim(message: str) -> bool:
    lower = message.casefold()
    return any(
        marker in lower
        for marker in (" should ", " must ", " need ", " means ", " prefer ", " because ", " is ")
    )


def derive_topics(self_messages: list[dict[str, Any]], context_messages: list[dict[str, Any]]) -> list[str]:
    text = " ".join(message["content"] for message in self_messages + context_messages).casefold()
    topics = [topic for topic, terms in TOPIC_TERMS.items() if any(term in text for term in terms)]
    return topics or ["general_reflection"]


def raw_ref(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_id": message["message_id"],
        "source_path": message["source_path"],
        "span_start": 0,
        "span_end": len(message["content"]),
        "span_text": message["content"],
        "speaker": message["speaker"],
        "style_owner": "self" if message["speaker"] == "self" else "context",
    }


def build_reasoning_points(self_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "point_id": f"p{index}",
            "order": index,
            "summary": short_text(message["content"]),
            "raw_refs": [raw_ref(message)],
        }
        for index, message in enumerate(self_messages, start=1)
    ]


def build_connectors(self_messages: list[dict[str, Any]], points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    connectors: list[dict[str, Any]] = []
    for index, message in enumerate(self_messages[:-1]):
        match = CONNECTOR_RE.search(message["content"])
        if not match:
            continue
        connectors.append(
            {
                "connector_id": f"c{len(connectors) + 1}",
                "raw": match.group(0),
                "between": {
                    "from": [points[index]["point_id"]],
                    "to": [points[index + 1]["point_id"]],
                },
                "raw_refs": [
                    {
                        "message_id": message["message_id"],
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "span_text": match.group(0),
                    }
                ],
            }
        )
    return connectors


def build_chain(
    window: dict[str, Any],
    messages_by_id: dict[str, dict[str, Any]],
    number: int,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = [messages_by_id[message_id] for message_id in window["message_ids"]]
    self_messages = [message for message in messages if message["speaker"] == "self"]
    context_messages = [message for message in messages if message["speaker"] != "self"]
    self_tokens = message_tokens(self_messages)
    context_tokens = message_tokens(context_messages)
    total_tokens = self_tokens + context_tokens
    token_words = set(re.findall(r"[a-z][a-z'-]+", " ".join(message["content"] for message in self_messages).casefold()))
    signal_count = len(token_words & SIGNAL_TERMS)
    claim_count = sum(explicit_claim(message["content"]) for message in self_messages)
    topics = derive_topics(self_messages, context_messages)
    gaps = [
        later["timestamp_ms"] - earlier["timestamp_ms"]
        for earlier, later in zip(messages, messages[1:])
        if later["timestamp_ms"] >= earlier["timestamp_ms"]
    ]
    max_gap_minutes = max(gaps, default=0) / 60_000

    reasoning_density = min(1.0, 0.12 * min(signal_count, 5) + 0.12 * min(claim_count, 3) + (0.18 if self_tokens >= 80 else 0.0) + (0.12 if len(self_messages) >= 2 else 0.0))
    continuity = max(0.0, 1.0 - min(max_gap_minutes / 30.0, 1.0))
    context_dependency = min(1.0, context_tokens / total_tokens) if total_tokens else 0.0
    article_usefulness = min(1.0, 0.18 + 0.12 * min(len(topics), 4) + 0.1 * min(claim_count, 3) + (0.15 if self_tokens >= 80 else 0.0))
    self_density = self_tokens / total_tokens if total_tokens else 0.0
    retrieval_value = min(
        1.0,
        0.35 * reasoning_density
        + 0.25 * continuity
        + 0.25 * article_usefulness
        + 0.15 * self_density,
    )
    calibration = calibration or build_edge_calibration([])
    units = sentence_units("\n\n".join(message["content"] for message in self_messages))
    edge_assessments = [
        score_candidate_edge(
            left,
            right,
            bridge="",
            shared_source=True,
            endpoint_support=1.0,
            calibration=calibration,
        )
        for left, right in zip(units, units[1:])
    ]
    chain_assessment = score_candidate_chain(
        [edge["score"] for edge in edge_assessments],
        calibration,
        fallback_score=retrieval_value,
    )
    accepted = bool(self_messages) and chain_assessment["band"] != "demoted"

    self_ids = [message["message_id"] for message in self_messages]
    context_ids = [message["message_id"] for message in context_messages]
    points = build_reasoning_points(self_messages) if accepted else []
    first_self = short_text(self_messages[0]["content"]) if self_messages else "No self-authored message is present."
    last_self = short_text(self_messages[-1]["content"]) if self_messages else "No self-authored message is present."
    middle_self = short_text(self_messages[len(self_messages) // 2]["content"]) if self_messages else ""
    direct_claims = [
        f"Direct self statement: {short_text(message['content'])}"
        for message in self_messages
        if explicit_claim(message["content"])
    ][:5]
    if not direct_claims and self_messages:
        direct_claims = [f"Direct self statement: {first_self}"]

    chain_id = f"chain_{number:06d}"
    confidence = (
        "high"
        if chain_assessment["probability"] >= 0.65
        else "medium"
        if chain_assessment["probability"] >= 0.20
        else "low"
    )
    ambiguity = "low" if accepted and len(self_messages) >= 2 and continuity >= 0.5 else "medium"
    return {
        "chain_id": chain_id,
        "window_id": window["window_id"],
        "source": window["source"],
        "source_thread_id": window["source_thread_id"],
        "sequence_type": "reasoning_chain" if accepted else "non_reasoning_sequence",
        "status": "accepted" if accepted else "rejected",
        "interpreter": "empirical_user_distribution",
        "score_version": SCORE_VERSION,
        "start_message_id": window["message_ids"][0],
        "end_message_id": window["message_ids"][-1],
        "message_ids": window["message_ids"],
        "self_message_ids": self_ids,
        "context_message_ids": context_ids,
        "context_summary": f"Extractive context evidence: {short_text(context_messages[0]['content'])}" if context_messages else "No context message is present.",
        "self_summary": f"Extractive self evidence: {first_self}" if self_messages else "No self-authored evidence is present.",
        "chain_summary": f"Extractive baseline sequence about {', '.join(topics)}: {first_self}"
        if accepted
        else "Rejected by the deterministic baseline because the window lacks sufficient self-authored reasoning signals.",
        "reasoning_arc": {
            "premise": first_self,
            "development": middle_self,
            "conclusion": last_self,
        },
        "knowledge_claims": direct_claims if accepted else [],
        "reasoning_points": points,
        "verification": {
            "evidence_for_reasoning": [
                f"Detected {signal_count} reasoning signal terms.",
                f"Detected {claim_count} explicit claim-like self statements.",
                f"Self-authored token density is {self_density:.2f}.",
            ],
            "evidence_against_reasoning": [
                "The baseline uses lexical and structural signals rather than semantic review.",
                "Adjacent messages may still belong to different thoughts.",
            ],
            "directly_supported_claims": direct_claims,
            "inferred_claims": [],
            "unsupported_or_risky_claims": [],
            "alternative_interpretations": [
                "This may be exploratory reflection or conversation context rather than a complete reasoning chain."
            ],
            "ambiguity_level": ambiguity,
            "summary_confidence": confidence,
        },
        "reasoning_topology": {
            "connectors": build_connectors(self_messages, points) if accepted else []
        },
        "labels": {
            "topics": topics,
            "reasoning_type": "systems_design" if {"rag", "software", "memory_architecture"} & set(topics) else "reflective_analysis",
            "style_mode": "long_form_reasoning" if window["source"] in {"whatsapp", "whisper"} else "conceptual_iterative",
            "context_type": "conversation_context" if context_messages else "self_authored_reflection",
            "retrieval_roles": ["reasoning", "knowledge", "style"] if accepted else [],
        },
        "metrics": {
            "self_token_count": self_tokens,
            "context_token_count": context_tokens,
            "total_token_count": total_tokens,
            "self_message_count": len(self_messages),
            "context_message_count": len(context_messages),
            "duration_minutes": (window["end_timestamp_ms"] - window["start_timestamp_ms"]) / 60_000,
        },
        "scores": {
            "reasoning_density": round(reasoning_density, 6),
            "continuity": round(continuity, 6),
            "context_dependency": round(context_dependency, 6),
            "article_usefulness": round(article_usefulness, 6),
            "retrieval_value": round(retrieval_value, 6),
            "causal_chain_score": chain_assessment["score"],
            "causal_chain_probability": chain_assessment["probability"],
            "causal_chain_band": chain_assessment["band"],
            "weakest_edge_score": chain_assessment["weakest_edge_score"],
        },
        "edge_assessments": edge_assessments,
        "scoring_explanation": {
            "reasoning_density": "Deterministic combination of signal terms, explicit claims, self-token volume, and self-message count.",
            "continuity": "Inverse normalized maximum adjacent timestamp gap within the candidate window.",
            "context_dependency": "Context token count divided by total window token count.",
            "article_usefulness": "Deterministic combination of topic coverage, claim count, and self-token volume.",
            "retrieval_value": "Feature score retained for diagnostics; it is no longer a fixed admission threshold.",
            "causal_chain_probability": "Corpus-relative percentile with geometric aggregation and a weak-link penalty.",
        },
    }


def build_reasoning_chains(windows: list[dict[str, Any]], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages_by_id = {message["message_id"]: message for message in messages}
    calibration_documents = [
        {
            "collection": "content_memory",
            "text": "\n\n".join(
                messages_by_id[message_id]["content"]
                for message_id in window["self_message_ids"]
                if message_id in messages_by_id
            ),
        }
        for window in windows
        if window.get("self_message_ids")
    ]
    calibration = build_edge_calibration(calibration_documents)
    return [
        build_chain(window, messages_by_id, index, calibration)
        for index, window in enumerate(windows, start=1)
        if all(message_id in messages_by_id for message_id in window["message_ids"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default="data/normalized/messages.json")
    parser.add_argument("--windows", default="data/windows/candidate_windows.json")
    parser.add_argument("--output", default="data/chains/reasoning_chains.json")
    args = parser.parse_args()
    chains = build_reasoning_chains(read_json(args.windows), read_json(args.messages))
    write_json(args.output, chains)
    print(f"Reasoning-chain objects: {len(chains)}")
    print(f"Accepted: {sum(chain['status'] == 'accepted' for chain in chains)}")
    print(f"Rejected: {sum(chain['status'] != 'accepted' for chain in chains)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
