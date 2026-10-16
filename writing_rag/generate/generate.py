"""Prepare or run grounded Qwen writing generation."""

from __future__ import annotations

import argparse
import resource
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from writing_rag.ingest.causal_probability import (
    build_edge_calibration,
    score_candidate_chain,
    score_candidate_edge,
)
from writing_rag.retrieve.facet_retrieve import FacetRetrievalConfig, retrieve_faceted
from .generation_calls import GenerationCallResult, run_generation_call
from writing_rag.plan.article_plan import (  # noqa: E402
    build_article_plan,
    format_article_plan_context,
    validate_article_plan,
)
from .paragraph_generation import (
    build_planning_inventory,
    generate_paragraph_article,
    ledger_content_results,
    ledger_summary,
    parse_paragraph_intents,
)
from writing_rag.plan.narrative_reconstruction import format_narrative_packets  # noqa: E402
from writing_rag.runtime.pipeline_common import PROJECT_ROOT, count_tokens, read_json, write_json
from writing_rag.runtime.model_paths import configure_model_cache
from writing_rag.retrieve.retrieve import (
    EmbeddingSession,
    lexical_overlap,
    narrative_roles,
    retrieve,
    retrieve_many,
    retrieve_narrative,
)


DEFAULT_TEMPLATE = "prompts/article.txt"
DEFAULT_NARRATIVE_PATTERN = "prompts/narrative_pattern.txt"
DEFAULT_GENERATOR_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
DOCUMENT_TYPES = ("investigative", "mdx")
DEFAULT_NARRATIVE_PATTERNS = {
    "investigative": "prompts/narrative_pattern.txt",
    "mdx": "prompts/mdx_pattern.txt",
}
CONTENT_MIN = 4
CONTENT_MAX = 6
STYLE_QUOTA = 2
NARRATIVE_ROLE_ORDER = (
    "observation",
    "motivation",
    "initial_hypothesis",
    "discarded_approach",
    "reframe",
    "experiment",
    "finding",
    "mechanism",
    "decision",
    "conclusion",
)
DOCUMENT_ROLE_ORDERS = {
    "investigative": NARRATIVE_ROLE_ORDER,
    # MDX needs a document-centered progression. The existing role names are
    # retained so the paragraph verifier and invariants remain compatible.
    "mdx": ("observation", "reframe", "finding", "mechanism", "decision", "conclusion"),
}
DOCUMENT_ROLE_GUIDANCE = {
    "investigative": (
        "For an investigative article, use observation, motivation, initial_hypothesis, "
        "discarded_approach, reframe, experiment, finding, mechanism, decision, and conclusion "
        "only when their evidence is available."
    ),
    "mdx": (
        "For an MDX article, use observation for the opening question or tension, reframe for a "
        "definition or distinction, finding for a layered textual or technical claim, mechanism "
        "for how the distinction produces an implication, decision for a tradeoff or practical "
        "consequence, and conclusion for the return to the governing question. Do not create "
        "personal scenes, experiments, or discarded approaches unless the evidence explicitly "
        "contains them."
    ),
}
NARRATIVE_TRANSITIONS = (
    "at first",
    "but",
    "the question became",
    "the control showed",
    "this changed",
    "the practical choice",
)
CONNECTOR_RE = re.compile(
    r"\b(because|so|but|therefore|then|however|which means|leads to|hence|as a result)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
STYLE_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "not",
    "one",
    "only",
    "our",
    "said",
    "some",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
QUALIFIER_TERMS = {
    "cannot",
    "could",
    "less",
    "likely",
    "may",
    "maybe",
    "might",
    "more",
    "must",
    "not",
    "only",
    "possibly",
    "probably",
    "uncertain",
    "weak",
}
QUALIFIER_EQUIVALENTS = {
    "less": re.compile(r"\b(less|lower|not as|reduced|sacrific\w*)\b", re.IGNORECASE),
    "more": re.compile(r"\b(more|higher|increased|widened|greater)\b", re.IGNORECASE),
    "may": re.compile(r"\b(may|might|perhaps|possibly)\b", re.IGNORECASE),
    "maybe": re.compile(r"\b(maybe|perhaps|possibly|might)\b", re.IGNORECASE),
    "could": re.compile(r"\b(could|might|possibly)\b", re.IGNORECASE),
    "likely": re.compile(r"\b(likely|probably|appears|suggests)\b", re.IGNORECASE),
    "weak": re.compile(r"\b(weak|small|tiny|subtle|low)\b", re.IGNORECASE),
    "ambiguous details": re.compile(
        r"\b(ambiguous|unclear|inaccessible|uncertainty|uncertain)\b",
        re.IGNORECASE,
    ),
}
STRONG_CROSS_NODE_CAUSAL_RE = re.compile(
    r"\b(rejected because|disproved by|therefore|as a result|this changed)\b",
    re.IGNORECASE,
)


def qualifier_preserved(term: str, text: str) -> bool:
    if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
        return True
    pattern = QUALIFIER_EQUIVALENTS.get(term)
    return bool(pattern and pattern.search(text))


CAUSAL_CHAIN_SLOTS = (
    (
        "observation",
        "What concrete problem or anomaly was first observed?",
        "noticed appeared visible observed problem anomaly artifact",
    ),
    (
        "motivation",
        "Why did the problem matter?",
        "goal wanted aim why mattered stakes motivation",
    ),
    (
        "initial_hypothesis",
        "What was the first explanation, including uncertainty?",
        "first suspicion hypothesis maybe uncertainty explanation",
    ),
    (
        "discarded_approach",
        "What tempting approach was rejected and why?",
        "rejected discarded approach second suspicion cannot",
    ),
    (
        "reframe",
        "How was the question turned into a fair test?",
        "reframe testable fair core comparison compare source output control",
    ),
    (
        "experiment",
        "What experiment or control was run?",
        "experiment control tested paired compare upload download dimensions method",
    ),
    (
        "finding",
        "What did the measurements show?",
        "finding result showed measured residual colors percentages widened gained supports",
    ),
    (
        "mechanism",
        "What mechanism best explains the finding?",
        "mechanism caused creates resize re-encode processing path",
    ),
    (
        "decision",
        "What practical choice followed?",
        "decision tradeoff practical choice",
    ),
)


def build_causal_scaffold(query: str) -> list[dict[str, Any]]:
    return [
        {
            "node_id": f"node_{index:02d}_{role}",
            "role": role,
            "question": question,
            "retrieval_query": f"{query}; {retrieval_terms}",
            "span_query": retrieval_terms,
            "evidence": [],
        }
        for index, (role, question, retrieval_terms) in enumerate(
            CAUSAL_CHAIN_SLOTS, start=1
        )
    ]


def evidence_sentences(text: str) -> list[str]:
    candidates = []
    parts = re.split(r"```[^\n]*\n?(.*?)```", text, flags=re.DOTALL)
    for part_index, part in enumerate(parts):
        if part_index % 2:
            code_lines = [line.strip() for line in part.splitlines() if line.strip()]
            if code_lines:
                candidates.append(" ".join(code_lines))
            continue
        blocks = re.split(r"\n\s*\n", part)
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            if any(line.startswith("|") for line in lines):
                units = lines
            else:
                units = [" ".join(lines)]
            for unit in units:
                cleaned = re.sub(r"^(?:>|[-*]|\d+[.)])\s*", "", unit).strip()
                if not cleaned or re.fullmatch(r"\|?[\s:-]+\|?", cleaned):
                    continue
                candidates.extend(
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+|\s*[;]\s*", cleaned)
                    if sentence.strip()
                )
    return candidates


def best_source_span(text: str, query: str) -> str:
    """Select one complete source sentence; never cut evidence mid-qualifier."""
    candidates = evidence_sentences(text)
    if not candidates:
        return ""
    index, sentence = max(
        enumerate(candidates),
        key=lambda pair: (lexical_overlap(query, pair[1]), -pair[0]),
    )
    if index > 0 and re.match(
        r"^(this|that|these|those|it|this approach|this option)\b",
        sentence,
        re.IGNORECASE,
    ):
        return f"{candidates[index - 1]} {sentence}"
    if sentence.endswith(":") and index + 1 < len(candidates):
        return f"{sentence} {candidates[index + 1]}"
    return sentence


def select_source_excerpt(
    item: dict[str, Any],
    result: dict[str, Any],
    *,
    excerpt_tokens: int,
    token_counter: Callable[[str], int],
) -> tuple[str, bool]:
    document = result["document"]
    source_span = result.get("source_span") or best_source_span(
        document["text"], item["span_query"]
    )
    if excerpt_tokens and token_counter(source_span) > excerpt_tokens:
        role_span = best_source_span(source_span, item["span_query"])
        if role_span:
            return role_span, role_span != source_span
    return source_span, False


def verify_causal_edges(
    scaffold: list[dict[str, Any]],
    calibration: dict[str, Any] | None = None,
) -> None:
    """Score adjacency against user writing while keeping direct support separate."""
    if calibration is None:
        calibration = next(
            (
                item["edge_calibration"]
                for item in scaffold
                if item.get("edge_calibration")
            ),
            build_edge_calibration([]),
        )
    for item in scaffold:
        item.pop("outgoing_edges", None)
    retrieved = [item for item in scaffold if item.get("status") == "retrieved"]
    incident: dict[str, list[dict[str, Any]]] = {
        item["node_id"]: [] for item in retrieved
    }
    edge_scores: list[float] = []
    for before, after in zip(retrieved, retrieved[1:]):
        before_by_id = {
            result["document"]["document_id"]: result for result in before["evidence"]
        }
        after_by_id = {
            result["document"]["document_id"]: result for result in after["evidence"]
        }
        shared_ids = sorted(set(before_by_id) & set(after_by_id))
        connector_refs = []
        shared_bridges: dict[str, str] = {}
        for document_id in shared_ids:
            document = before_by_id[document_id]["document"]
            text = document["text"]
            before_span = before_by_id[document_id].get("source_span", "")
            after_span = after_by_id[document_id].get("source_span", "")
            before_start = text.find(before_span)
            after_start = text.find(after_span)
            bridge = ""
            if before_start >= 0 and after_start >= before_start + len(before_span):
                bridge = text[before_start + len(before_span) : after_start]
            shared_bridges[document_id] = bridge
            connectors = sorted(
                set(match.casefold() for match in CONNECTOR_RE.findall(bridge))
            )
            if "->" in bridge:
                connectors.append("->")
            if before_span == after_span:
                connectors.extend(
                    match.casefold() for match in CONNECTOR_RE.findall(before_span)
                )
                if "->" in before_span:
                    connectors.append("->")
            connectors = sorted(set(connectors))
            if connectors:
                connector_refs.append(
                    {"document_id": document_id, "connectors": connectors}
                )
        if shared_ids:
            shared_document_id = max(
                shared_ids,
                key=lambda document_id: (
                    before_by_id[document_id].get("score", 0.0)
                    + after_by_id[document_id].get("score", 0.0)
                ),
            )
            before_result = before_by_id[shared_document_id]
            after_result = after_by_id[shared_document_id]
            candidate_bridge = shared_bridges.get(shared_document_id, "")
        else:
            before_result = max(
                before["evidence"],
                key=lambda result: (
                    result.get("span_overlap", 0.0),
                    result.get("score", 0.0),
                ),
            )
            after_result = max(
                after["evidence"],
                key=lambda result: (
                    result.get("span_overlap", 0.0),
                    result.get("score", 0.0),
                ),
            )
            candidate_bridge = ""
        left_span = before_result.get("source_span") or best_source_span(
            before_result["document"]["text"], before["span_query"]
        )
        right_span = after_result.get("source_span") or best_source_span(
            after_result["document"]["text"], after["span_query"]
        )
        endpoint_support = (
            max(0.0, min(1.0, before_result.get("score", 0.0)))
            * max(0.0, min(1.0, after_result.get("score", 0.0)))
        ) ** 0.5
        assessment = score_candidate_edge(
            left_span,
            right_span,
            bridge=candidate_bridge,
            shared_source=bool(shared_ids),
            endpoint_support=endpoint_support,
            calibration=calibration,
        )
        status = (
            "reasonable_inference"
            if connector_refs and assessment["band"] != "demoted"
            else "unsupported"
        )
        edge = {
            "from_node": before["node_id"],
            "to_node": after["node_id"],
            "status": status,
            "support_band": assessment["band"],
            "support_score": assessment["score"],
            "support_probability": assessment["probability"],
            "reward_multiplier": assessment["reward_multiplier"],
            "endpoint_document_ids": [
                before_result["document"]["document_id"],
                after_result["document"]["document_id"],
            ],
            "support_features": assessment["features"],
            "evidence_refs": connector_refs,
            "evidence_against": (
                []
                if connector_refs
                else [
                    "No shared source span explicitly connects these narrative nodes."
                ]
            ),
            "alternative_interpretations": [
                "The two supported claims may be adjacent rather than causally dependent."
            ],
        }
        before.setdefault("outgoing_edges", []).append(edge)
        incident[before["node_id"]].append(assessment)
        incident[after["node_id"]].append(assessment)
        edge_scores.append(assessment["score"])

    for item in retrieved:
        assessments = incident[item["node_id"]]
        probability = (
            sum(edge["probability"] for edge in assessments) / len(assessments)
            if assessments
            else 1.0
        )
        multiplier = (
            sum(edge["reward_multiplier"] for edge in assessments) / len(assessments)
            if assessments
            else 1.0
        )
        item["causal_probability"] = round(probability, 6)
        item["causal_reward_multiplier"] = round(multiplier, 6)
        for result in item["evidence"]:
            result["causal_adjusted_score"] = round(
                result["score"] * multiplier,
                6,
            )
    if scaffold:
        scaffold[0]["chain_assessment"] = score_candidate_chain(
            edge_scores, calibration
        )


def fill_causal_scaffold(
    query: str,
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any] | None,
    *,
    top_k: int = 1,
    minimum_score: float = 0.25,
    session: EmbeddingSession | None = None,
    edge_calibration: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    scaffold = build_causal_scaffold(query)
    if scaffold and edge_calibration is not None:
        scaffold[0]["edge_calibration"] = edge_calibration
    has_role_metadata = any(
        narrative_roles(document)
        for document in documents
        if document.get("collection") == "content_memory"
    )
    retrieved = retrieve_many(
        [item["retrieval_query"] for item in scaffold],
        documents,
        vectors,
        top_k=max(top_k * 4, 12),
        collection="content_memory",
        embedding_metadata=embedding_metadata,
        session=session,
    )
    for item, candidates in zip(scaffold, retrieved):
        role_candidates = [
            result
            for result in candidates
            if item["role"] in narrative_roles(result["document"])
        ]
        ranked = role_candidates if has_role_metadata else candidates
        evidence = []
        for result in ranked:
            if result["score"] < minimum_score:
                continue
            source_span = best_source_span(
                result["document"]["text"],
                item["span_query"],
            )
            span_overlap = lexical_overlap(item["span_query"], source_span)
            if has_role_metadata and span_overlap <= 0:
                continue
            evidence.append(
                {
                    **result,
                    "source_span": source_span,
                    "span_overlap": round(span_overlap, 6),
                    "support": (
                        "direct_role_span_match"
                        if has_role_metadata
                        else "retrieval_only"
                    ),
                }
            )
        evidence.sort(
            key=lambda result: (result["span_overlap"], result["score"]),
            reverse=True,
        )
        evidence = evidence[:top_k]
        item["evidence"] = evidence
        item["status"] = "retrieved" if item["evidence"] else "unsupported"
        item["rejection_reason"] = (
            "no role-matched result met the minimum retrieval score"
            if not item["evidence"]
            else ""
        )
    verify_causal_edges(scaffold, edge_calibration)
    return scaffold


def format_causal_scaffold(
    scaffold: list[dict[str, Any]],
    *,
    excerpt_tokens: int = 80,
    token_counter: Callable[[str], int] = count_tokens,
) -> str:
    node_lines = []
    supported_edges: dict[tuple[str, str], dict[str, Any]] = {}
    demoted_edges: dict[tuple[str, str], dict[str, Any]] = {}
    for item in scaffold:
        if item.get("status", "retrieved") != "retrieved":
            continue
        for result in item["evidence"]:
            document = result["document"]
            source_span, _ = select_source_excerpt(
                item,
                result,
                excerpt_tokens=excerpt_tokens,
                token_counter=token_counter,
            )
            node_lines.append(
                f"- {item['node_id']} | {item['role']} | "
                f"[{document['document_id']}]: {source_span}"
            )
        for edge in item.get("outgoing_edges", []):
            if edge["status"] == "reasonable_inference":
                supported_edges[(edge["from_node"], edge["to_node"])] = edge
            else:
                demoted_edges[(edge["from_node"], edge["to_node"])] = edge
    if not node_lines:
        return "No causal role has sufficient retrieved evidence. Do not invent a causal chain."
    edge_lines = [
        f"- {edge['from_node']} -> {edge['to_node']} | {edge['support_band']} inference | "
        f"p={edge['support_probability']:.3f} | "
        f"refs={','.join(ref['document_id'] for ref in edge['evidence_refs'])}"
        for edge in supported_edges.values()
    ]
    if not edge_lines:
        edge_lines = ["- none; node order is narrative order, not proof of causality"]
    demoted_lines = [
        f"- {edge['from_node']} -> {edge['to_node']} | "
        f"{edge.get('support_band', 'unsupported')} | "
        f"p={edge.get('support_probability', 0.0):.3f}; keep claims separate"
        for edge in demoted_edges.values()
    ] or ["- none"]
    return (
        "Supported nodes (node | role | source):\n"
        + "\n".join(node_lines)
        + "\n\nSupported edges:\n"
        + "\n".join(edge_lines)
        + "\n\nDemoted or unsupported edges:\n"
        + "\n".join(demoted_lines)
    )


def causal_scaffold_evidence_count(scaffold: list[dict[str, Any]]) -> int:
    return sum(len(item["evidence"]) for item in scaffold)


def build_strict_evidence_fallback(scaffold: list[dict[str, Any]]) -> str:
    """Compose a fail-safe narrative from source sentences without inferred edges."""
    paragraphs = []
    seen: set[str] = set()
    for item in scaffold:
        if item.get("status") != "retrieved":
            continue
        for result in item.get("evidence", []):
            source_span, _ = select_source_excerpt(
                item,
                result,
                excerpt_tokens=80,
                token_counter=count_tokens,
            )
            normalized = re.sub(r"\s+", " ", source_span).strip().casefold()
            if source_span.strip() and normalized not in seen:
                paragraphs.append(source_span.strip())
                seen.add(normalized)
    return "\n\n".join(paragraphs)


def build_content_evidence_fallback(results: list[dict[str, Any]]) -> str:
    """Compose source-addressable content when the drafting stage is unavailable."""
    paragraphs: list[str] = []
    seen: set[str] = set()
    for result in results:
        document = result.get("document", {})
        source_span = str(result.get("source_span") or document.get("text", "")).strip()
        normalized = re.sub(r"\s+", " ", source_span).casefold()
        if source_span and normalized not in seen:
            paragraphs.append(source_span)
            seen.add(normalized)
    return "\n\n".join(paragraphs)


def causal_scaffold_token_profile(
    scaffold: list[dict[str, Any]],
    *,
    token_counter: Callable[[str], int] = count_tokens,
) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for excerpt_tokens in (80, 40, 20):
        key = str(excerpt_tokens)
        rendered = format_causal_scaffold(
            scaffold,
            excerpt_tokens=excerpt_tokens,
            token_counter=token_counter,
        )
        excerpts = [
            select_source_excerpt(
                item,
                result,
                excerpt_tokens=excerpt_tokens,
                token_counter=token_counter,
            )
            for item in scaffold
            for result in item["evidence"]
        ]
        source_span_tokens = [token_counter(span) for span, _ in excerpts]
        raw_source_span_tokens = sum(
            token_counter(
                result.get("source_span")
                or best_source_span(result["document"]["text"], item["span_query"])
            )
            for item in scaffold
            for result in item["evidence"]
        )
        over_budget = sum(tokens > excerpt_tokens for tokens in source_span_tokens)
        source_span_token_total = sum(source_span_tokens)
        total_tokens = token_counter(rendered)
        profile[key] = {
            "representation": "lossless_compact",
            "total_tokens": total_tokens,
            "source_span_tokens": source_span_token_total,
            "raw_source_span_tokens": raw_source_span_tokens,
            "structural_overhead_tokens": max(
                0, total_tokens - source_span_token_total
            ),
            "slot_tokens": {
                item["role"]: token_counter(
                    format_causal_scaffold(
                        [item],
                        excerpt_tokens=excerpt_tokens,
                        token_counter=token_counter,
                    )
                )
                for item in scaffold
            },
            "role_count": len(scaffold),
            "evidence_item_count": causal_scaffold_evidence_count(scaffold),
            "evidence_ids_preserved": all(
                result["document"]["document_id"]
                in format_causal_scaffold(
                    [item],
                    excerpt_tokens=excerpt_tokens,
                    token_counter=token_counter,
                )
                for item in scaffold
                for result in item["evidence"]
            ),
            "complete_source_spans_preserved": not any(
                compacted for _, compacted in excerpts
            ),
            "complete_source_sentences_preserved": True,
            "compacted_span_count": sum(compacted for _, compacted in excerpts),
            "max_source_span_tokens": max(source_span_tokens, default=0),
            "atomic_spans_over_budget": over_budget,
            "budget_satisfied": over_budget == 0,
        }
    return profile


def format_results(results: list[dict[str, Any]]) -> str:
    """Format evidence while keeping graph provenance outside the model prompt."""
    evidence = []
    for result in results:
        document = result["document"]
        role = result.get("narrative_role_group", "")
        role_label = f", narrative_role={role}" if role else ""
        metadata = document.get("metadata", {})
        prompt_provenance = {
            key: metadata[key]
            for key in ("source", "source_scope", "speaker_filter", "grouping_method")
            if metadata.get(key) not in {None, ""}
        }
        evidence.append(
            "### "
            + document["document_id"]
            + f" ({document['collection']}/{document['view']}, score={result['score']:.4f}{role_label})\n"
            + document["text"]
            + "\n\nSource provenance:"
            + str(prompt_provenance)
        )
    return "\n\n".join(evidence) if evidence else "No matching evidence was retrieved."


def compact_results_for_planning(
    results: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    """Keep complete evidence spans while bounding the global planning context."""
    compacted = []
    for result in results:
        copied = dict(result)
        copied_document = dict(result["document"])
        span = result.get("source_span") or best_source_span(
            copied_document["text"], query
        )
        copied_document["text"] = span
        copied["document"] = copied_document
        copied["source_span"] = span
        compacted.append(copied)
    return compacted


def format_style_profile(
    results: list[dict[str, Any]],
    structure_documents: list[dict[str, Any]] | None = None,
) -> str:
    """Expose style characteristics without exposing style-source subject matter."""
    if not results:
        return "No style profile was retrieved. Use a plain, direct default voice."
    texts = [result["document"]["text"] for result in results]
    sentences = [
        sentence.strip()
        for text in texts
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    words = [WORD_RE.findall(text) for text in texts]
    word_counts = [len(item) for item in words]
    uncertainty = sum(
        len(
            re.findall(
                r"\b(maybe|could|likely|appears|suggests|uncertain|not sure)\b",
                text,
                re.I,
            )
        )
        for text in texts
    )
    first_person = sum(
        len(re.findall(r"\b(I|I'm|I've|my|me|we|our|us)\b", text, re.I))
        for text in texts
    )
    question_count = sum(text.count("?") for text in texts)
    avg_words = sum(word_counts) / max(1, len(word_counts))
    structure_documents = structure_documents or []
    architecture = []
    if structure_documents:
        metadata = [document.get("metadata", {}) for document in structure_documents]
        architecture = [
            "Long-form architecture (derived from style-only document profiles):",
            f"- finished essay profiles: {len(metadata)}",
            f"- median words per essay: {sorted(int(item.get('word_count', 0)) for item in metadata)[len(metadata) // 2]}",
            f"- median paragraphs per essay: {sorted(int(item.get('paragraph_count', 0)) for item in metadata)[len(metadata) // 2]}",
            f"- median headings per essay: {sorted(int(item.get('heading_count', 0)) for item in metadata)[len(metadata) // 2]}",
            "- use the profiles for article-scale pacing and section progression, never for claims or topic language",
        ]
    return "\n".join(
        [
            "Style profile (derived statistics only; no style-source claims are available):",
            f"- examples: {len(results)}",
            f"- average words per example: {avg_words:.1f}",
            f"- average sentences per example: {len(sentences) / max(1, len(results)):.1f}",
            f"- uncertainty markers across examples: {uncertainty}",
            f"- first-person markers across examples: {first_person}",
            f"- questions across examples: {question_count}",
            "- emulate rhythm, directness, and qualification only; do not copy subject matter or phrasing",
        ]
        + architecture
    )


def validate_retrieval_contract(
    content_results: list[dict[str, Any]],
    style_results: list[dict[str, Any]],
    *,
    require_quota: bool = False,
    content_source_scopes: set[str] | None = None,
) -> None:
    if require_quota and not CONTENT_MIN <= len(content_results) <= CONTENT_MAX:
        raise ValueError(
            f"Content retrieval must contain {CONTENT_MIN}-{CONTENT_MAX} chunks"
        )
    if require_quota and len(style_results) != STYLE_QUOTA:
        raise ValueError(f"Style retrieval must contain exactly {STYLE_QUOTA} examples")
    for result in content_results:
        document = result["document"]
        if document.get("collection") != "content_memory":
            raise ValueError("Content results must come from content_memory")
        source_scope = document.get("metadata", {}).get("source_scope")
        allowed_source_scopes = content_source_scopes or {"content"}
        if source_scope not in allowed_source_scopes:
            expected = ", ".join(sorted(allowed_source_scopes))
            raise ValueError(
                f"Content results must be marked source_scope in {{{expected}}}"
            )
        if document.get("metadata", {}).get("style_owner") != "none":
            raise ValueError("Content results cannot carry style ownership")
    for result in style_results:
        document = result["document"]
        metadata = document.get("metadata", {})
        if document.get("collection") != "style_memory":
            raise ValueError("Style results must come from style_memory")
        if metadata.get("source_scope") != "style":
            raise ValueError("Style results must be marked source_scope=style")
        if metadata.get("speaker_filter") != "self_only":
            raise ValueError("Style results must use speaker_filter=self_only")
        if metadata.get("style_owner") != "self":
            raise ValueError("Style results must be owned by self")


def retrieve_style_examples(
    query: str,
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any],
    *,
    session: EmbeddingSession,
    top_k: int = STYLE_QUOTA,
) -> list[dict[str, Any]]:
    """Reserve one slot for scarce long-form prose without losing technical voice."""
    longform = [
        document
        for document in documents
        if document.get("metadata", {}).get("style_form") == "long_form_essay"
        and document.get("metadata", {}).get("style_unit") == "section"
    ]
    other = [document for document in documents if document not in longform]
    if top_k == 2 and longform and other:
        regular_result = retrieve(
            query,
            other,
            vectors,
            top_k=1,
            view="style",
            collection="style_memory",
            embedding_metadata=embedding_metadata,
            session=session,
        )
        longform_result = retrieve(
            query + " long-form article paragraph transitions sustained explanation",
            longform,
            vectors,
            top_k=1,
            view="style",
            collection="style_memory",
            embedding_metadata=embedding_metadata,
            session=session,
        )
        return regular_result + longform_result
    return retrieve(
        query,
        documents,
        vectors,
        top_k=top_k,
        view="style",
        collection="style_memory",
        embedding_metadata=embedding_metadata,
        session=session,
    )


def build_generation_context(
    query: str,
    results: list[dict[str, Any]],
    template: str,
    *,
    style_results: list[dict[str, Any]] | None = None,
    style_template: str = "",
    narrative_template: str = "",
    narrative_packets: list[dict[str, Any]] | None = None,
    article_plan: dict[str, Any] | None = None,
    causal_scaffold: list[dict[str, Any]] | None = None,
    causal_excerpt_tokens: int = 80,
    causal_token_counter: Callable[[str], int] = count_tokens,
    include_style_evidence: bool = False,
    style_structure_documents: list[dict[str, Any]] | None = None,
) -> str:
    style_results = style_results or []
    causal_scaffold = causal_scaffold or []
    narrative_packets = narrative_packets or []
    validate_retrieval_contract(
        results,
        style_results,
        content_source_scopes={"content", "mdx_content"} if article_plan else None,
    )
    narrative_section = (
        f"## Narrative composition contract\n\n{narrative_template.strip()}\n\n"
        if narrative_template.strip()
        else ""
    )
    narrative_packet_section = (
        "## Source-backed narrative packets\n\n"
        "These packets are bounded source evidence. Explicit causal edges require "
        "source markers; inferred context links are not causal proof.\n\n"
        + format_narrative_packets(narrative_packets)
        + "\n\n"
        if narrative_packets
        else ""
    )
    article_plan_section = (
        "## Article authority and provenance plan\n\n"
        "This is a source-selection contract, not a replacement for author intent. "
        "Source-backed claims must remain inside their source allowlist. Author-asserted "
        "claims may use only assertion IDs supplied outside the model. Causal claims "
        "require an explicit narrative edge; shared source IDs do not prove causality.\n\n"
        + format_article_plan_context(article_plan)
        + "\n\n"
        if article_plan
        else ""
    )
    causal_section = (
        "## Retrieved causal scaffold\n\n"
        "This scaffold is a candidate retrieval map. Retrieved excerpts support "
        "role questions, not causal links. A connector is only a verification hint; "
        "the draft must omit or qualify any causal link that is not directly supported.\n\n"
        + format_causal_scaffold(
            causal_scaffold,
            excerpt_tokens=causal_excerpt_tokens,
            token_counter=causal_token_counter,
        )
        + "\n\n"
        if causal_scaffold
        else ""
    )
    style_section = (
        f"## Retrieved style evidence ({len(style_results)} examples)\n\n{format_results(style_results)}\n"
        if include_style_evidence
        else f"## Derived style profile ({len(style_results)} examples)\n\n{format_style_profile(style_results, style_structure_documents)}\n"
    )
    return (
        "# Grounded writing context\n\n"
        f"Query: {query}\n\n"
        "## Source contract\n\n"
        "Content sources provide claims, facts, arguments, and project evidence. "
        "Style sources provide voice, rhythm, directness, and qualification patterns. "
        "Content and style are separate evidence classes: style text must never "
        "substitute for factual support, and context evidence must never become style evidence.\n\n"
        + narrative_section
        + narrative_packet_section
        + article_plan_section
        + causal_section
        + f"## Generation instructions\n\n{template.strip()}\n\n"
        + f"## Style instructions\n\n{style_template.strip()}\n\n"
        + f"## Retrieved content evidence ({len(results)} chunks)\n\n{format_results(results)}\n\n"
        + style_section
    )


def build_outline_prompt(
    context: str,
    narrative_template: str,
    document_type: str = "investigative",
) -> str:
    if document_type not in DOCUMENT_TYPES:
        raise ValueError(f"Unknown document type: {document_type}")
    role_guidance = DOCUMENT_ROLE_GUIDANCE[document_type]
    return (
        "You are preparing an evidence-grounded narrative outline. Do not draft the article yet.\n\n"
        "Use content evidence for claims. Treat style evidence as voice guidance only. "
        "Create one or more paragraph intents for each distinct reader-facing question requested by the "
        "article goal or editorial structure. Multiple intents may share a narrative role. "
        + role_guidance
        + " Use only roles for which retrieval appears possible; "
        "there is no minimum role count. Retrieved causal nodes may guide the order, but planning-inventory entries "
        "may also define retrieval questions. Edge probabilities are soft risk signals, not proof. Never strengthen "
        "an edge beyond its source support. "
        "If an approach is named without a source-backed rejection reason, do not invent one. "
        "Preserve negation, uncertainty, comparatives, and tradeoff direction from source spans. "
        "Write each supported paragraph intent on one line using exactly: "
        "INTENT N | role=<role> | question=<reader question> | claim=<candidate claim> | "
        "claims=<comma-separated claim types> | depends_on=[intent-id] | opens=<question> | "
        "closes=<intended answer> | transition=<answer|extend|redirect|contrast|return_to_narration> | "
        "refs contains source document IDs; depends_on contains only earlier intent IDs. "
        "The candidate claim is a retrieval intention, not established evidence. Each line must cite one or "
        "more retrieved source document IDs in refs. Use only these claim types: personal_observation, "
        "personal_action, technical_fact, technical_cause, interpretation. Do not invent personal scenes, "
        "motives, experiments, or discarded approaches.\n\n"
        "Narrative contract:\n"
        + narrative_template.strip()
        + "\n\nGrounded context:\n"
        + context
    )


def _normalized_labels(text: str) -> str:
    return text.casefold().replace("_", " ").replace("-", " ")


def validate_outline(
    outline: str,
    *,
    minimum_roles: int = 0,
    valid_document_ids: set[str] | None = None,
    allowed_roles: set[str] | None = None,
    role_order: tuple[str, ...] = NARRATIVE_ROLE_ORDER,
    max_intents: int | None = None,
) -> list[str]:
    normalized = _normalized_labels(outline)
    parsed_intents = parse_paragraph_intents(outline)
    observed_roles = {
        str(intent.get("role", "")) for intent in parsed_intents if intent.get("role")
    }
    if not parsed_intents:
        observed_roles = {
            role
            for role in NARRATIVE_ROLE_ORDER
            if _normalized_labels(role) in normalized
        }
    present = [role for role in role_order if _normalized_labels(role) in normalized]
    if parsed_intents:
        present = [role for role in role_order if role in observed_roles]
    cited_ids = {
        document_id
        for intent in parsed_intents
        for document_id in intent.get("outline_document_ids", [])
    }
    if not parsed_intents and outline.strip():
        cited_ids = {
            value.strip()
            for value in re.findall(r"\[([^\]]+)\]", outline)
            if not value.strip().casefold().startswith("intent-")
        }
    issues: list[str] = []
    if len(present) < minimum_roles:
        issues.append(
            f"outline covers {len(present)} narrative roles; add at least {minimum_roles} supported roles"
        )
    if max_intents is not None and len(parsed_intents) > max_intents:
        issues.append(
            f"outline contains {len(parsed_intents)} intents; maximum is {max_intents}"
        )
    if allowed_roles is not None:
        unsupported_roles = sorted(observed_roles - allowed_roles)
        if unsupported_roles:
            issues.append(
                "outline uses unsupported narrative roles: "
                + ", ".join(unsupported_roles)
            )
    if outline.strip() and not cited_ids:
        issues.append(
            "outline bullets must cite retrieved document IDs in square brackets"
        )
    if parsed_intents:
        seen_intent_ids: set[str] = set()
        for index, intent in enumerate(parsed_intents, start=1):
            intent_id = str(intent.get("intent_id", "")).strip()
            if not intent_id:
                issues.append(f"intent {index} is missing an intent_id")
            elif intent_id in seen_intent_ids:
                issues.append(f"duplicate intent_id: {intent_id}")
            refs = {
                str(document_id).strip()
                for document_id in intent.get("outline_document_ids", [])
                if str(document_id).strip()
            }
            if not refs:
                issues.append(f"{intent_id or f'intent-{index}'} has no evidence references")
            dependencies = [
                str(dependency).strip()
                for dependency in intent.get("depends_on", [])
                if str(dependency).strip()
            ]
            future_or_missing = sorted(
                dependency
                for dependency in dependencies
                if dependency not in seen_intent_ids
            )
            if future_or_missing:
                issues.append(
                    f"{intent_id or f'intent-{index}'} depends on missing or future intents: "
                    + ", ".join(future_or_missing)
                )
            seen_intent_ids.add(intent_id)
    if valid_document_ids is not None:
        unknown_ids = sorted(cited_ids - valid_document_ids)
        if unknown_ids:
            issues.append(
                f"outline cites unknown document IDs: {', '.join(unknown_ids)}"
            )
    return issues


def build_draft_prompt(
    context: str,
    outline: str,
    narrative_template: str,
    *,
    minimum_words: int = 700,
) -> str:
    return (
        "Write the final post from the grounded context and outline below. "
        "The outline controls composition; content evidence controls claims; style evidence controls voice. "
        "Do not mention the outline, retrieval, prompts, or document IDs in the post. "
        + (
            f"Write at least {minimum_words} words across multiple paragraphs. "
            if minimum_words
            else ""
        )
        + "Preserve uncertainty and first person only when supported by content evidence.\n\n"
        "Use visible causal transitions only where supported: "
        "At first..., But..., The question became..., The control showed..., "
        "This changed..., The practical choice... "
        "There is no minimum transition count. "
        "Do not force a transition or narrative role when evidence is absent. "
        "When no edge is supported, order the nodes narratively without claiming that one "
        "disproved, caused, or changed another. If a rejected idea has no sourced rejection "
        "reason, name it only as a considered hypothesis. Preserve words such as not, maybe, "
        "may, less, and more when they control the meaning of a source claim. Equivalent wording "
        "is allowed, but never strengthen may/might/could into is/will/must.\n\n"
        "Narrative contract:\n"
        + narrative_template.strip()
        + "\n\nGrounded context:\n"
        + context
        + "\n\nGenerated outline:\n"
        + outline
    )


def content_only_context(context: str) -> str:
    for style_marker in ("\n## Retrieved style evidence", "\n## Derived style profile"):
        context = context.split(style_marker, 1)[0]
    return context


def build_correction_prompt(
    context: str,
    outline: str,
    draft: str,
    issues: list[str],
    narrative_template: str,
    *,
    minimum_words: int = 700,
) -> str:
    return (
        "Rewrite the draft below to fix every validation failure. Return only the revised post.\n\n"
        "Validation failures:\n- "
        + "\n- ".join(issues)
        + (
            f"\n\nThe revised post must be at least {minimum_words} words, "
            if minimum_words
            else "\n\n"
        )
        + "Use multiple paragraphs when the evidence supports them. Delete any sentence that cannot be "
        "traced to content evidence. Do not use style examples as factual evidence. "
        "Do not invent evidence, personal experience, motives, experiments, or discarded approaches. "
        "If a failure names unsupported cross-node causality, remove the named causal phrase and split "
        "the claims into neutral, separately sourced sentences; do not replace it with a causal synonym.\n\n"
        "Narrative contract:\n"
        + narrative_template.strip()
        + "\n\nContent-only grounded context:\n"
        + content_only_context(context)
        + "\n\nOutline:\n"
        + outline
        + "\n\nDraft to revise:\n"
        + draft
    )


def validate_draft(
    draft: str,
    *,
    minimum_words: int = 0,
    minimum_transitions: int = 0,
    content_results: list[dict[str, Any]] | None = None,
    style_results: list[dict[str, Any]] | None = None,
    causal_scaffold: list[dict[str, Any]] | None = None,
    minimum_grounded_sentence_ratio: float = 0.55,
) -> list[str]:
    word_count = len(re.findall(r"\b[\w'-]+\b", draft))
    normalized = draft.casefold()
    transition_count = sum(
        normalized.count(transition) for transition in NARRATIVE_TRANSITIONS
    )
    issues = []
    if minimum_words and draft.strip() and not re.search(
        r"[.!?](?:[\"')\]]+)?\s*$", draft.strip()
    ):
        issues.append("draft does not end with sentence punctuation")
    if minimum_words and word_count < minimum_words:
        issues.append(f"draft has {word_count} words; minimum is {minimum_words}")
    if minimum_transitions and transition_count < minimum_transitions:
        issues.append(
            f"draft has {transition_count} causal transitions; minimum is {minimum_transitions}"
        )
    if content_results:
        source_terms = {
            term.casefold()
            for result in content_results
            for term in WORD_RE.findall(result["document"]["text"])
            if len(term) >= 4 and term.casefold() not in STYLE_STOPWORDS
        }
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", draft)
            if len(WORD_RE.findall(sentence)) >= 6
        ]
        grounded = sum(
            len({term.casefold() for term in WORD_RE.findall(sentence)} & source_terms)
            >= 2
            for sentence in sentences
        )
        ratio = grounded / max(1, len(sentences))
        if sentences and ratio < minimum_grounded_sentence_ratio:
            issues.append(
                f"only {ratio:.2f} of content-bearing sentences overlap retrieved content terms"
            )
    if style_results:
        style_phrases = {
            " ".join(words[index : index + 5]).casefold()
            for result in style_results
            for words in [
                [
                    word.casefold()
                    for word in WORD_RE.findall(result["document"]["text"])
                ]
            ]
            for index in range(max(0, len(words) - 4))
        }
        draft_words = [word.casefold() for word in WORD_RE.findall(draft)]
        copied_phrases = sum(
            " ".join(draft_words[index : index + 5]) in style_phrases
            for index in range(max(0, len(draft_words) - 4))
        )
        if copied_phrases:
            issues.append(
                f"draft contains {copied_phrases} five-word phrases copied from style evidence"
            )
    if causal_scaffold:
        draft_units = evidence_sentences(draft)
        for item in causal_scaffold:
            if item.get("status") != "retrieved":
                continue
            role_spans = [
                best_source_span(
                    result.get("source_span") or result["document"]["text"],
                    item["span_query"],
                )
                for result in item.get("evidence", [])
            ]
            role_qualifiers = {
                term.casefold()
                for role_span in role_spans
                for term in WORD_RE.findall(role_span)
                if term.casefold() in QUALIFIER_TERMS
            }
            role_query = item["span_query"] + " " + " ".join(role_spans)
            matching_unit = max(
                draft_units,
                key=lambda unit: lexical_overlap(role_query, unit),
                default=draft,
            )
            missing = sorted(
                term
                for term in role_qualifiers
                if not qualifier_preserved(term, matching_unit)
            )
            if missing:
                issues.append(
                    f"draft omits meaning-bearing qualifiers for {item['role']}: "
                    + ", ".join(missing)
                )
        supported_edges = [
            edge
            for item in causal_scaffold
            for edge in item.get("outgoing_edges", [])
            if edge.get("status") == "reasonable_inference"
        ]
        if not supported_edges:
            unsupported_assertions = sorted(
                set(
                    match.casefold()
                    for match in STRONG_CROSS_NODE_CAUSAL_RE.findall(draft)
                )
            )
            if unsupported_assertions:
                issues.append(
                    "draft asserts cross-node causality without a supported edge: "
                    + ", ".join(unsupported_assertions)
                )
    return issues


def _writing_features(text: str) -> dict[str, float]:
    words = WORD_RE.findall(text)
    word_count = max(1, len(words))
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if WORD_RE.findall(sentence)
    ]
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if WORD_RE.findall(part)]
    first_person = len(re.findall(r"\b(I|I'm|I've|my|me|we|our|us)\b", text, re.I))
    uncertainty = len(
        re.findall(
            r"\b(maybe|could|likely|appears|suggests|uncertain|not sure|may|might)\b",
            text,
            re.I,
        )
    )
    causal = len(CONNECTOR_RE.findall(text))
    contrast = len(
        re.findall(r"\b(but|however|instead|although|while|yet)\b", text, re.I)
    )
    sequence = len(
        re.findall(r"\b(first|second|then|after|before|finally|once)\b", text, re.I)
    )
    return {
        "sentence_words": word_count / max(1, len(sentences)),
        "paragraph_sentences": len(sentences) / max(1, len(paragraphs)),
        "first_person_per_100": 100.0 * first_person / word_count,
        "questions_per_10_sentences": 10.0 * text.count("?") / max(1, len(sentences)),
        "punctuation_per_100": 100.0 * len(re.findall(r"[:;—]", text)) / word_count,
        "causal_connectors_per_100": 100.0 * causal / word_count,
        "qualifiers_per_100": 100.0 * uncertainty / word_count,
        "contrast_per_100": 100.0 * contrast / word_count,
        "sequence_per_100": 100.0 * sequence / word_count,
    }


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _empirical_centrality(value: float, values: list[float]) -> float:
    if not values:
        return 1.0
    below = sum(candidate < value for candidate in values)
    equal = sum(candidate == value for candidate in values)
    percentile = (below + 0.5 * equal) / len(values)
    return max(0.0, 1.0 - abs(2.0 * percentile - 1.0))


def _distribution_probability(
    features: dict[str, float],
    calibration: dict[str, Any],
) -> float:
    keys = calibration["feature_keys"]
    if not keys:
        return 1.0
    probabilities = [
        _empirical_centrality(features[key], calibration["feature_values"][key])
        for key in keys
    ]
    return sum(probabilities) / len(probabilities)


def _build_distribution_calibration(
    texts: list[str],
    feature_keys: tuple[str, ...],
) -> dict[str, Any]:
    feature_rows = [_writing_features(text) for text in texts]
    feature_values = {key: [row[key] for row in feature_rows] for key in feature_keys}
    self_scores = [
        sum(
            _empirical_centrality(row[key], feature_values[key]) for key in feature_keys
        )
        / len(feature_keys)
        for row in feature_rows
    ]
    accept_threshold = _quantile(self_scores, 0.10) if self_scores else 0.85
    warning_threshold = _quantile(self_scores, 0.02) if self_scores else 0.65
    return {
        "sample_count": len(feature_rows),
        "feature_keys": list(feature_keys),
        "feature_values": feature_values,
        "accept_threshold": accept_threshold,
        "warning_threshold": warning_threshold,
        "feature_medians": {
            key: _quantile(values, 0.50) for key, values in feature_values.items()
        },
    }


def build_writing_calibration(
    style_documents: list[dict[str, Any]],
    causal_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calibrate style and causal expectations from non-holdout user writing."""
    style_texts = []
    weighted_seen: dict[float, int] = {}
    for document in style_documents:
        metadata = document.get("metadata", {})
        weight = float(metadata.get("calibration_weight", 1.0))
        if (
            document.get("collection") != "style_memory"
            or metadata.get("style_owner") != "self"
            or metadata.get("style_unit") == "document_profile"
            or weight <= 0.0
            or len(WORD_RE.findall(document.get("text", ""))) < 30
        ):
            continue
        if weight < 1.0:
            stride = max(2, round(1.0 / weight))
            weighted_seen[weight] = weighted_seen.get(weight, 0) + 1
            if weighted_seen[weight] % stride:
                continue
        style_texts.append(document["text"])
    causal_texts = [
        document["text"]
        for document in causal_documents
        if document.get("collection") == "content_memory"
        and len(WORD_RE.findall(document.get("text", ""))) >= 30
    ]
    return {
        "style": _build_distribution_calibration(
            style_texts,
            (
                "sentence_words",
                "paragraph_sentences",
                "first_person_per_100",
                "questions_per_10_sentences",
                "punctuation_per_100",
            ),
        ),
        "causality": _build_distribution_calibration(
            causal_texts,
            (
                "causal_connectors_per_100",
                "qualifiers_per_100",
                "contrast_per_100",
                "sequence_per_100",
            ),
        ),
        "causal_edges": build_edge_calibration(causal_documents),
    }


def calibration_summary(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "sample_count": distribution["sample_count"],
            "feature_keys": distribution["feature_keys"],
            "accept_threshold": round(distribution["accept_threshold"], 4),
            "warning_threshold": round(distribution["warning_threshold"], 4),
            **(
                {
                    "document_count": distribution["document_count"],
                    "chain_accept_threshold": round(
                        distribution["chain_accept_threshold"], 4
                    ),
                    "chain_warning_threshold": round(
                        distribution["chain_warning_threshold"], 4
                    ),
                }
                if "chain_accept_threshold" in distribution
                else {}
            ),
            "feature_medians": {
                key: round(value, 4)
                for key, value in distribution["feature_medians"].items()
            },
        }
        for name, distribution in calibration.items()
    }


def assess_draft_confidence(
    draft: str,
    *,
    content_results: list[dict[str, Any]] | None = None,
    style_results: list[dict[str, Any]] | None = None,
    causal_scaffold: list[dict[str, Any]] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate acceptability against the user's empirical writing distributions."""
    content_results = content_results or []
    style_results = style_results or []
    causal_scaffold = causal_scaffold or []
    calibration = calibration or build_writing_calibration(
        style_results, content_results
    )
    draft_terms = {term.casefold() for term in WORD_RE.findall(draft)}

    source_terms = {
        term.casefold()
        for result in content_results
        for term in WORD_RE.findall(result["document"]["text"])
        if len(term) >= 4 and term.casefold() not in STYLE_STOPWORDS
    }
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", draft)
        if len(WORD_RE.findall(sentence)) >= 6
    ]
    grounded_sentences = sum(
        len({term.casefold() for term in WORD_RE.findall(sentence)} & source_terms) >= 2
        for sentence in sentences
    )
    grounding = grounded_sentences / max(1, len(sentences)) if content_results else 1.0

    retrieved_roles = [
        item for item in causal_scaffold if item.get("status") == "retrieved"
    ]
    covered_roles = 0
    required_qualifiers: set[str] = set()
    for item in retrieved_roles:
        role_terms: set[str] = set()
        for result in item.get("evidence", []):
            role_span = best_source_span(
                result.get("source_span") or result["document"]["text"],
                item["span_query"],
            )
            role_terms.update(
                term.casefold()
                for term in WORD_RE.findall(role_span)
                if len(term) >= 4 and term.casefold() not in STYLE_STOPWORDS
            )
            required_qualifiers.update(
                term.casefold()
                for term in WORD_RE.findall(role_span)
                if term.casefold() in QUALIFIER_TERMS
            )
        required_overlap = min(2, len(role_terms))
        if required_overlap == 0 or len(role_terms & draft_terms) >= required_overlap:
            covered_roles += 1
    role_coverage = (
        covered_roles / max(1, len(retrieved_roles)) if retrieved_roles else 1.0
    )
    preserved_qualifiers = {
        term for term in required_qualifiers if qualifier_preserved(term, draft)
    }
    qualifier_preservation = (
        len(preserved_qualifiers) / len(required_qualifiers)
        if required_qualifiers
        else 1.0
    )

    supported_edge_count = sum(
        edge.get("status") == "reasonable_inference"
        for item in causal_scaffold
        for edge in item.get("outgoing_edges", [])
    )
    chain_assessment = next(
        (
            item["chain_assessment"]
            for item in causal_scaffold
            if item.get("chain_assessment")
        ),
        {"probability": 1.0, "band": "accepted", "score": 1.0},
    )
    chain_edge_probability = chain_assessment["probability"]
    unsupported_assertions = sorted(
        set(match.casefold() for match in STRONG_CROSS_NODE_CAUSAL_RE.findall(draft))
    )
    evidence_causal_support = (
        0.75 + 0.25 * chain_edge_probability
        if not unsupported_assertions
        else max(
            0.15,
            (0.55 ** len(unsupported_assertions))
            * (0.50 + 0.50 * chain_edge_probability)
            + min(0.25, supported_edge_count * 0.05),
        )
    )

    style_phrases = {
        " ".join(words[index : index + 5]).casefold()
        for result in style_results
        for words in [
            [word.casefold() for word in WORD_RE.findall(result["document"]["text"])]
        ]
        for index in range(max(0, len(words) - 4))
    }
    draft_words = [word.casefold() for word in WORD_RE.findall(draft)]
    copied_style_phrases = sum(
        " ".join(draft_words[index : index + 5]) in style_phrases
        for index in range(max(0, len(draft_words) - 4))
    )
    style_independence = 1.0 / (1.0 + copied_style_phrases)

    features = _writing_features(draft)
    empirical_style = _distribution_probability(features, calibration["style"])
    empirical_causality = _distribution_probability(features, calibration["causality"])
    style_probability = (empirical_style * style_independence) ** 0.5
    source_fidelity = (
        0.45 * evidence_causal_support
        + 0.35 * qualifier_preservation
        + 0.20 * role_coverage
    )
    causal_probability = (empirical_causality * source_fidelity) ** 0.5
    score = 0.40 * grounding + 0.40 * causal_probability + 0.20 * style_probability

    style_accept = calibration["style"]["accept_threshold"]
    style_warning = calibration["style"]["warning_threshold"]
    causal_accept = calibration["causality"]["accept_threshold"]
    causal_warning = calibration["causality"]["warning_threshold"]
    if grounding >= 0.80 and causal_probability >= causal_accept:
        decision = (
            "accept" if style_probability >= style_accept else "accept_with_warnings"
        )
    elif (
        grounding >= 0.60
        and causal_probability >= causal_warning
        and style_probability >= style_warning
    ):
        decision = "accept_with_warnings"
    elif grounding >= 0.50 and causal_probability >= causal_warning:
        decision = "demote_to_strict_fallback"
    else:
        decision = "reject"

    warnings = []
    if grounding < 0.80:
        warnings.append(f"grounding={grounding:.2f}")
    if style_probability < style_accept:
        warnings.append(
            f"style_probability={style_probability:.2f}<user_p10={style_accept:.2f}"
        )
    if causal_probability < causal_accept:
        warnings.append(
            f"causal_probability={causal_probability:.2f}<user_p10={causal_accept:.2f}"
        )
    if role_coverage < 0.80:
        warnings.append(f"role_coverage={role_coverage:.2f}")
    if qualifier_preservation < 0.80:
        warnings.append(f"qualifier_preservation={qualifier_preservation:.2f}")
    if unsupported_assertions:
        warnings.append(
            "unsupported_causal_phrases=" + ",".join(unsupported_assertions)
        )
    if chain_assessment["band"] == "demoted":
        warnings.append(
            "causal_chain_probability="
            f"{chain_edge_probability:.2f}<user_edge_chain_warning"
        )
    if copied_style_phrases:
        warnings.append(f"copied_style_phrases={copied_style_phrases}")

    return {
        "score": round(score, 4),
        "score_kind": "empirical_user_distribution_probability_proxy",
        "decision": decision,
        "probabilities": {
            "style": round(style_probability, 4),
            "causality": round(causal_probability, 4),
            "grounding": round(grounding, 4),
        },
        "empirical_thresholds": {
            "style_accept_p10": round(style_accept, 4),
            "style_warning_p02": round(style_warning, 4),
            "causality_accept_p10": round(causal_accept, 4),
            "causality_warning_p02": round(causal_warning, 4),
        },
        "components": {
            "empirical_style": round(empirical_style, 4),
            "style_independence": round(style_independence, 4),
            "empirical_causality": round(empirical_causality, 4),
            "evidence_causal_support": round(evidence_causal_support, 4),
            "causal_chain_probability": round(chain_edge_probability, 4),
            "causal_chain_score": round(chain_assessment["score"], 4),
            "causal_chain_band": chain_assessment["band"],
            "role_coverage": round(role_coverage, 4),
            "qualifier_preservation": round(qualifier_preservation, 4),
        },
        "warnings": warnings,
        "unsupported_causal_phrases": unsupported_assertions,
        "required_qualifiers": sorted(required_qualifiers),
        "preserved_qualifiers": sorted(preserved_qualifiers),
        "copied_style_phrases": copied_style_phrases,
    }


def should_select_correction(
    current_assessment: dict[str, Any],
    current_issues: list[str],
    candidate_assessment: dict[str, Any],
    candidate_issues: list[str],
    *,
    score_tolerance: float = 0.03,
) -> bool:
    """Prefer fewer invariant failures without accepting a material score collapse."""
    issue_improved = len(candidate_issues) < len(current_issues)
    score_delta = candidate_assessment["score"] - current_assessment["score"]
    return score_delta > 0 or (issue_improved and score_delta >= -score_tolerance)


def build_style_rewrite_prompt(draft: str, style_results: list[dict[str, Any]]) -> str:
    """Ask for a voice-only rewrite after factual drafting is complete."""
    return (
        "Rewrite the post for voice only. Preserve every factual claim, qualifier, "
        "number, named artifact, and causal relationship from the draft. Do not add "
        "examples, motives, experiences, explanations, or claims. Return only the post.\n\n"
        "Style guidance:\n"
        + format_style_profile(style_results)
        + "\n\nDraft whose meaning must remain unchanged:\n"
        + draft
    )


def _mlx_error(backend: str, exc: Exception) -> RuntimeError:
    package = "mlx-lm" if backend == "mlx-lm" else "mlx-vlm"
    return RuntimeError(
        f"{backend} generation requires {package} and a compatible Apple Silicon runtime. "
        "Install requirements-qwen.txt, verify the model is compatible with the selected backend, "
        "or use --backend prompt for context-only output."
    )


def _mlx_lm_generation_options(temperature: float) -> dict[str, Any]:
    """Adapt sampling arguments across mlx-lm API generations."""
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        return {"temp": temperature}
    return {"sampler": make_sampler(temperature)}


def require_metal_runtime(backend: str = "mlx-vlm") -> None:
    """Run a real MLX operation in a child process before loading a model."""
    module = "mlx_lm" if backend in {"mlx", "mlx-lm"} else "mlx_vlm"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import {module}; import mlx.core as mx; "
                "x=mx.array([1.0])+1; mx.eval(x); "
                "device=mx.default_device(); print('gpu_op_ok', device, x); "
                "raise SystemExit(0 if device == mx.gpu and mx.metal.is_available() else 2)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        message = detail[-1] if detail else f"MLX probe exited {probe.returncode}"
        raise RuntimeError(f"MLX requires an accessible Apple Metal device: {message}")


def format_mlx_lm_prompt(processor: Any, prompt: str) -> str:
    """Apply the tokenizer chat template exactly once for MLX-LM."""
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError(
            "MLX-LM tokenizer must provide apply_chat_template; "
            "refusing to send an unformatted prompt"
        )
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )


def generate_with_mlx_lm(
    prompt: str, model_name: str, max_tokens: int, temperature: float
) -> str:
    configure_model_cache()
    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise _mlx_error("mlx-lm", exc) from exc
    try:
        model, tokenizer = load(model_name)
        formatted_prompt = format_mlx_lm_prompt(tokenizer, prompt)
        return generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            **_mlx_lm_generation_options(temperature),
        )
    except Exception as exc:
        raise RuntimeError(f"Could not run MLX-LM model {model_name}: {exc}") from exc


def generate_with_mlx_vlm(
    prompt: str, model_name: str, max_tokens: int, temperature: float
) -> str:
    configure_model_cache()
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise _mlx_error("mlx-vlm", exc) from exc
    try:
        model, processor = load(model_name)
        config = load_config(model_name)
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images=0)
        result = generate(
            model,
            processor,
            formatted_prompt,
            [],
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        return getattr(result, "text", str(result))
    except Exception as exc:
        raise RuntimeError(f"Could not run MLX-VLM model {model_name}: {exc}") from exc


def generate_with_backend(
    prompt: str,
    backend: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    runtime: Any = None,
) -> str:
    if runtime is not None:
        return runtime.generate(prompt, max_tokens, temperature)
    if backend == "mlx":
        backend = "mlx-lm"
    if backend == "mlx-lm":
        return generate_with_mlx_lm(prompt, model_name, max_tokens, temperature)
    if backend == "mlx-vlm":
        return generate_with_mlx_vlm(prompt, model_name, max_tokens, temperature)
    raise ValueError(f"Unsupported generator backend: {backend}")


class GenerationRuntime:
    """Load one MLX model once and reuse it for outline, draft, and checks."""

    def __init__(
        self,
        backend: str,
        model_name: str,
        model_cache: str | None = None,
    ) -> None:
        configure_model_cache(model_cache)
        self.backend = "mlx-lm" if backend == "mlx" else backend
        self.model_name = model_name
        require_metal_runtime(self.backend)
        if self.backend == "mlx-lm":
            try:
                from mlx_lm import generate, load
            except ImportError as exc:
                raise _mlx_error("mlx-lm", exc) from exc
            self._generate = generate
            self.model, self.processor = load(model_name)
        elif self.backend == "mlx-vlm":
            try:
                from mlx_vlm import generate, load
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm.utils import load_config
            except ImportError as exc:
                raise _mlx_error("mlx-vlm", exc) from exc
            self._generate = generate
            self.apply_chat_template = apply_chat_template
            self.config = load_config(model_name)
            self.model, self.processor = load(model_name)
        else:
            raise ValueError(f"Unsupported generator backend: {backend}")

    def format_prompt(self, prompt: str) -> str:
        if self.backend == "mlx-lm":
            return format_mlx_lm_prompt(self.processor, prompt)
        return self.apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=0,
        )

    def count_tokens(self, prompt: str) -> int:
        return self.count_text_tokens(self.format_prompt(prompt))

    def count_text_tokens(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        encoded = tokenizer.encode(text)
        if isinstance(encoded, dict):
            encoded = encoded.get("input_ids", [])
        return len(encoded)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        formatted_prompt = self.format_prompt(prompt)
        if self.backend == "mlx-lm":
            return self._generate(
                self.model,
                self.processor,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                **_mlx_lm_generation_options(temperature),
            )
        result = self._generate(
            self.model,
            self.processor,
            formatted_prompt,
            [],
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        return getattr(result, "text", str(result))


def merge_content_results(
    content_results: list[dict[str, Any]],
    causal_scaffold: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Select a deduplicated content budget that maximizes supported-role coverage."""
    causal_results = [
        (item["role"], result)
        for item in causal_scaffold
        if item.get("status") == "retrieved"
        for result in item["evidence"]
    ]
    candidates = [result for _, result in causal_results] + content_results
    by_id: dict[str, dict[str, Any]] = {}
    role_coverage: dict[str, set[str]] = {}
    for role, result in causal_results:
        document_id = result["document"]["document_id"]
        role_coverage.setdefault(document_id, set()).add(role)
    for result in candidates:
        document_id = result["document"]["document_id"]
        current = by_id.get(document_id)
        result_rank = result.get("causal_adjusted_score", result["score"])
        current_rank = (
            current.get("causal_adjusted_score", current["score"])
            if current is not None
            else -1.0
        )
        if current is None or result_rank > current_rank:
            by_id[document_id] = result
    merged = sorted(
        by_id.values(),
        key=lambda result: (
            -result.get("causal_adjusted_score", result["score"]),
            result["document"]["document_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    uncovered_roles = {
        item["role"] for item in causal_scaffold if item.get("status") == "retrieved"
    }
    while uncovered_roles and len(selected) < top_k:
        choices = [
            result
            for result in merged
            if result["document"]["document_id"] not in selected_ids
            and role_coverage.get(result["document"]["document_id"], set())
            & uncovered_roles
        ]
        if not choices:
            break
        chosen = max(
            choices,
            key=lambda result: (
                len(
                    role_coverage.get(result["document"]["document_id"], set())
                    & uncovered_roles
                ),
                result.get("causal_adjusted_score", result["score"]),
            ),
        )
        document_id = chosen["document"]["document_id"]
        selected.append(chosen)
        selected_ids.add(document_id)
        uncovered_roles -= role_coverage.get(document_id, set())
    for result in merged:
        if len(selected) >= top_k:
            break
        document_id = result["document"]["document_id"]
        if document_id not in selected_ids:
            selected.append(result)
            selected_ids.add(document_id)
    for item in causal_scaffold:
        item.pop("outgoing_edges", None)
        item["evidence"] = [
            result
            for result in item.get("evidence", [])
            if result["document"]["document_id"] in selected_ids
        ]
        if not item["evidence"]:
            item["status"] = "unsupported"
    verify_causal_edges(causal_scaffold)
    return selected


def memory_provenance() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "peak_rss_raw": usage.ru_maxrss,
        "peak_rss_unit": "bytes_on_macos_or_kilobytes_on_linux",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--content-top-k",
        "--top-k",
        dest="content_top_k",
        type=int,
        default=5,
        choices=range(CONTENT_MIN, CONTENT_MAX + 1),
    )
    parser.add_argument(
        "--view", choices=("content", "summary", "reasoning", "knowledge")
    )
    parser.add_argument(
        "--style-query",
        default="direct systems explanation with evidence, assumptions, constraints, limitations, and uncertainty",
    )
    parser.add_argument(
        "--style-top-k", type=int, default=STYLE_QUOTA, choices=(STYLE_QUOTA,)
    )
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--vectors", default="data/embeddings/vectors.json")
    parser.add_argument("--metadata", default="data/embeddings/metadata.json")
    parser.add_argument(
        "--article-plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build an opt-in source hydration and authority plan from the existing "
            "Qwen content results. Retrieval and the default generation path are unchanged."
        ),
    )
    parser.add_argument(
        "--canonical-messages",
        default="data/normalized/messages.json",
        help="Canonical messages used only when --article-plan is enabled.",
    )
    parser.add_argument(
        "--author-assertions",
        help="Optional JSON list of author_assertion_id/text records for --article-plan.",
    )
    parser.add_argument(
        "--article-plan-output",
        help="Optional JSON output path for the validated article authority/provenance plan.",
    )
    parser.add_argument("--style-documents")
    parser.add_argument("--style-vectors")
    parser.add_argument("--style-metadata")
    parser.add_argument("--additional-style-documents")
    parser.add_argument("--additional-style-vectors")
    parser.add_argument("--additional-style-metadata")
    parser.add_argument("--output", default="data/generated/article_context.md")
    parser.add_argument(
        "--backend", choices=("prompt", "mlx", "mlx-lm", "mlx-vlm"), default="prompt"
    )
    parser.add_argument("--model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument(
        "--model-cache", help="Hugging Face home for downloaded model weights."
    )
    parser.add_argument(
        "--embedding-device",
        help="Override the device recorded in embedding metadata, for example cpu.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--outline-max-tokens", type=int, default=900)
    parser.add_argument("--min-outline-roles", type=int, default=0)
    parser.add_argument("--min-draft-words", type=int, default=0)
    parser.add_argument("--min-transitions", type=int, default=0)
    parser.add_argument("--max-corrections", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--context-token-limit",
        type=int,
        default=12000,
        help="Reject prompts above the measured starting context budget.",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--style-template", default="prompts/style_prompt.txt")
    parser.add_argument(
        "--evidence-trace-output",
        help="Optional path for the explicit [START: ...]/[END: ...] retrieval trace.",
    )
    parser.add_argument(
        "--document-type",
        choices=DOCUMENT_TYPES,
        default="investigative",
        help="Composition family used for outline role guidance and narrative defaults.",
    )
    parser.add_argument(
        "--narrative-pattern",
        default=None,
        help="Optional narrative contract; defaults according to --document-type.",
    )
    parser.add_argument(
        "--narrative", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--causal-chain", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--causal-top-k", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--two-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--paragraph-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate evidence-scoped paragraph groups and repair only failed groups.",
    )
    parser.add_argument("--intent-top-k", type=int, default=3, choices=(2, 3, 4))
    parser.add_argument(
        "--intent-group-size",
        type=int,
        default=1,
        choices=(1, 2, 3, 4),
        help="Number of intents per verifier group; 1 keeps evidence and sentence scope local.",
    )
    parser.add_argument("--intent-minimum-score", type=float, default=0.20)
    parser.add_argument(
        "--intent-retrieval-strategy",
        choices=("baseline", "faceted"),
        default="baseline",
        help="Use baseline hybrid retrieval or relevance-gated source-span facets for intents.",
    )
    parser.add_argument("--facet-anchor-pool-k", type=int, default=10)
    parser.add_argument("--facet-anchor-k", type=int, default=2)
    parser.add_argument("--facet-top-k", type=int, default=20)
    parser.add_argument("--facet-base-weight", type=float, default=1.0)
    parser.add_argument("--facet-weight", type=float, default=1.0)
    parser.add_argument("--facet-min-anchor-lexical-score", type=float, default=0.3)
    parser.add_argument("--facet-min-anchor-term-hits", type=int, default=2)
    parser.add_argument(
        "--max-retrieval-retries",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Retry retrieval once for paragraphs whose claim audit finds weak support.",
    )
    parser.add_argument(
        "--style-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run an experimental post-hoc voice rewrite. Disabled by default because "
            "lexical grounding checks cannot prove semantic equivalence."
        ),
    )
    parser.add_argument(
        "--run-metadata", default="data/generated/generation_metadata.json"
    )
    args = parser.parse_args()
    model_cache = configure_model_cache(args.model_cache)
    if not 0.0 <= args.intent_minimum_score <= 1.0:
        parser.error("--intent-minimum-score must be between 0 and 1")
    intent_facet_config = (
        FacetRetrievalConfig(
            top_k=max(args.intent_top_k * 3, 8),
            anchor_pool_k=args.facet_anchor_pool_k,
            anchor_k=args.facet_anchor_k,
            facet_top_k=args.facet_top_k,
            base_weight=args.facet_base_weight,
            facet_weight=args.facet_weight,
            min_anchor_lexical_score=args.facet_min_anchor_lexical_score,
            min_anchor_term_hits=args.facet_min_anchor_term_hits,
        )
        if args.intent_retrieval_strategy == "faceted"
        else None
    )

    documents = read_json(args.documents)
    vectors = read_json(args.vectors)
    embedding_metadata = read_json(args.metadata)
    article_plan: dict[str, Any] | None = None
    style_documents = read_json(args.style_documents or args.documents)
    style_vectors = read_json(args.style_vectors or args.vectors)
    style_embedding_metadata = read_json(args.style_metadata or args.metadata)
    if args.embedding_device:
        embedding_metadata = {
            **embedding_metadata,
            "device": args.embedding_device,
        }
        style_embedding_metadata = {
            **style_embedding_metadata,
            "device": args.embedding_device,
        }
    if any(
        (
            args.additional_style_documents,
            args.additional_style_vectors,
            args.additional_style_metadata,
        )
    ):
        if not all(
            (
                args.additional_style_documents,
                args.additional_style_vectors,
                args.additional_style_metadata,
            )
        ):
            parser.error(
                "all three --additional-style-* paths must be supplied together"
            )
        additional_metadata = read_json(args.additional_style_metadata)
        compatibility_keys = (
            "embedding_backend",
            "embedding_model",
            "dimension",
            "normalized",
        )
        if any(
            additional_metadata.get(key) != style_embedding_metadata.get(key)
            for key in compatibility_keys
        ):
            raise ValueError(
                "Additional style index must use the same embedding model and shape"
            )
        additional_documents = read_json(args.additional_style_documents)
        additional_vectors = read_json(args.additional_style_vectors)
        existing_ids = {document["document_id"] for document in style_documents}
        duplicate_ids = existing_ids & {
            document["document_id"] for document in additional_documents
        }
        if duplicate_ids:
            raise ValueError(
                f"Additional style index has duplicate IDs: {sorted(duplicate_ids)[:3]}"
            )
        style_documents = style_documents + additional_documents
        style_vectors = style_vectors + additional_vectors
    style_structure_documents = [
        document
        for document in style_documents
        if document.get("metadata", {}).get("style_unit") == "document_profile"
    ]
    style_example_documents = [
        document
        for document in style_documents
        if document.get("metadata", {}).get("style_unit") != "document_profile"
    ]
    content_embedding_session = EmbeddingSession(vectors, embedding_metadata)
    shared_encoder = (
        embedding_metadata.get("embedding_backend")
        == style_embedding_metadata.get("embedding_backend")
        and embedding_metadata.get("embedding_model")
        == style_embedding_metadata.get("embedding_model")
        and embedding_metadata.get("device") == style_embedding_metadata.get("device")
    )
    style_embedding_session = (
        content_embedding_session
        if (args.style_vectors or args.vectors) == args.vectors
        and (args.style_metadata or args.metadata) == args.metadata
        and not args.additional_style_vectors
        else EmbeddingSession(
            style_vectors,
            style_embedding_metadata,
            encoder_session=content_embedding_session if shared_encoder else None,
        )
    )
    template = (PROJECT_ROOT / args.template).read_text(encoding="utf-8")
    style_template = (PROJECT_ROOT / args.style_template).read_text(encoding="utf-8")
    narrative_pattern = (
        args.narrative_pattern or DEFAULT_NARRATIVE_PATTERNS[args.document_type]
    )
    narrative_template = (PROJECT_ROOT / narrative_pattern).read_text(encoding="utf-8")
    if args.narrative:
        content_results = retrieve_narrative(
            args.query,
            documents,
            vectors,
            top_k=args.content_top_k,
            embedding_metadata=embedding_metadata,
            session=content_embedding_session,
        )
    else:
        content_results = retrieve(
            args.query,
            documents,
            vectors,
            top_k=args.content_top_k,
            view=args.view,
            collection="content_memory",
            embedding_metadata=embedding_metadata,
            session=content_embedding_session,
        )
    style_results = retrieve_style_examples(
        args.style_query,
        style_example_documents,
        style_vectors,
        style_embedding_metadata,
        top_k=args.style_top_k,
        session=style_embedding_session,
    )
    validate_retrieval_contract(
        content_results,
        style_results,
        require_quota=True,
        content_source_scopes={"content", "mdx_content"} if args.article_plan else None,
    )
    writing_calibration = build_writing_calibration(style_documents, documents)
    causal_scaffold = (
        fill_causal_scaffold(
            args.query,
            documents,
            vectors,
            embedding_metadata,
            top_k=args.causal_top_k,
            session=content_embedding_session,
            edge_calibration=writing_calibration["causal_edges"],
        )
        if args.causal_chain
        else []
    )
    if causal_scaffold:
        content_results = merge_content_results(
            content_results,
            causal_scaffold,
            top_k=args.content_top_k,
        )
        validate_retrieval_contract(
            content_results,
            style_results,
            require_quota=True,
            content_source_scopes={"content", "mdx_content"} if args.article_plan else None,
        )
    if args.article_plan:
        canonical_messages = read_json(args.canonical_messages)
        canonical_by_id = {
            str(item["message_id"]): item
            for item in canonical_messages
            if item.get("message_id")
        }
        author_assertions = (
            read_json(args.author_assertions) if args.author_assertions else []
        )
        if not isinstance(author_assertions, list):
            raise ValueError("--author-assertions must contain a JSON list")
        article_plan = build_article_plan(
            content_results,
            canonical_by_id,
            author_assertions=author_assertions,
            plan_id=f"generate:{args.output}",
        )
        plan_errors = validate_article_plan(article_plan)
        if plan_errors:
            raise ValueError(
                "article plan validation failed: " + "; ".join(plan_errors)
            )
        if not args.article_plan_output:
            args.article_plan_output = str(Path(args.output).with_suffix(".article-plan.json"))
        write_json(args.article_plan_output, article_plan)
    planning_inventory = ""
    planning_document_ids: set[str] = set()
    paragraph_planning_source_results = content_results
    if args.paragraph_mode:
        planning_limit = min(
            48,
            sum(
                document.get("collection") == "content_memory"
                for document in documents
            ),
        )
        if args.intent_retrieval_strategy == "faceted":
            planning_config = replace(
                intent_facet_config or FacetRetrievalConfig(),
                top_k=planning_limit,
                anchor_pool_k=max(
                    (intent_facet_config or FacetRetrievalConfig()).anchor_pool_k,
                    planning_limit,
                ),
            )
            planning_pool = retrieve_faceted(
                args.query,
                documents,
                vectors,
                config=planning_config,
                collection="content_memory",
                embedding_metadata=embedding_metadata,
                session=content_embedding_session,
            )
        else:
            planning_pool = retrieve(
                args.query,
                documents,
                vectors,
                top_k=planning_limit,
                collection="content_memory",
                embedding_metadata=embedding_metadata,
                session=content_embedding_session,
            )
        planning_inventory, planning_document_ids = build_planning_inventory(
            planning_pool,
            query=args.query,
            span_selector=best_source_span,
        )
        # Paragraph mode has its own intent-scoped evidence ledger. Keep the
        # outline planner on the same direct retrieval family and out of the
        # broad causal scaffold, which is assessed separately below.
        paragraph_planning_source_results = planning_pool[: args.content_top_k]
    context_results = (
        compact_results_for_planning(paragraph_planning_source_results, args.query)
        if args.paragraph_mode
        else content_results
    )
    generation_context_scaffold = [] if args.paragraph_mode else causal_scaffold
    base_context = build_generation_context(
        args.query,
        context_results,
        template,
        style_results=style_results,
        style_template=style_template,
        narrative_template=narrative_template,
        narrative_packets=article_plan.get("narrative_packets", []) if article_plan else None,
        article_plan=article_plan,
        style_structure_documents=style_structure_documents,
    )
    runtime: GenerationRuntime | None = None
    runtime_error: str | None = None
    if args.backend != "prompt":
        try:
            runtime = GenerationRuntime(
                args.backend, args.model, model_cache=model_cache
            )
        except Exception as error:
            runtime_error = f"{type(error).__name__}: {error}"

    token_counter = runtime.count_tokens if runtime is not None else count_tokens
    output_token_counter = (
        runtime.count_text_tokens if runtime is not None else count_tokens
    )
    generation_calls: list[dict[str, object]] = []

    def invoke_generation(
        stage: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        *,
        fallback: str = "",
    ) -> GenerationCallResult:
        if runtime is None:
            result = GenerationCallResult(
                stage=stage,
                ok=False,
                text=fallback.strip(),
                attempts=0,
                error=runtime_error or "generator runtime unavailable",
                fallback_used=bool(fallback.strip()),
            )
        else:
            result = run_generation_call(
                runtime.generate,
                prompt,
                max_tokens,
                temperature,
                stage=stage,
                retries=1,
                fallback=fallback,
                retry_temperature=min(temperature, 0.1),
            )
        generation_calls.append(result.as_dict())
        return result

    def paragraph_generate(prompt: str, max_tokens: int, temperature: float) -> str:
        if runtime is None:
            raise RuntimeError(runtime_error or "generator runtime unavailable")
        return runtime.generate(prompt, max_tokens, temperature)

    causal_token_profile = causal_scaffold_token_profile(
        causal_scaffold,
        token_counter=output_token_counter,
    )
    token_count_encoding = (
        f"{args.model} runtime tokenizer after chat template"
        if runtime is not None
        else "tiktoken:gpt-4o-or-regex-fallback; approximate for Qwen"
    )
    base_context_tokens = token_counter(base_context)
    if args.context_token_limit <= 0:
        raise ValueError("--context-token-limit must be greater than zero")
    causal_excerpt_tokens = 80
    context = build_generation_context(
        args.query,
        context_results,
        template,
        style_results=style_results,
        style_template=style_template,
        narrative_template=narrative_template,
        narrative_packets=article_plan.get("narrative_packets", []) if article_plan else None,
        article_plan=article_plan,
        style_structure_documents=style_structure_documents,
        causal_scaffold=generation_context_scaffold,
        causal_excerpt_tokens=causal_excerpt_tokens,
        causal_token_counter=output_token_counter,
    )
    context_tokens = token_counter(context)
    for compact_excerpt_tokens in (40, 20):
        if (
            context_tokens <= args.context_token_limit
            or not generation_context_scaffold
        ):
            break
        causal_excerpt_tokens = compact_excerpt_tokens
        context = build_generation_context(
            args.query,
            context_results,
            template,
            style_results=style_results,
            style_template=style_template,
            narrative_template=narrative_template,
            narrative_packets=article_plan.get("narrative_packets", []) if article_plan else None,
            article_plan=article_plan,
            style_structure_documents=style_structure_documents,
            causal_scaffold=generation_context_scaffold,
            causal_excerpt_tokens=causal_excerpt_tokens,
            causal_token_counter=output_token_counter,
        )
        context_tokens = token_counter(context)
    if context_tokens > args.context_token_limit:
        raise ValueError(
            f"Generation context has {context_tokens} tokens; "
            f"the configured limit is {args.context_token_limit}."
        )

    target = PROJECT_ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outline = ""
    generated_tokens = 0
    outline_issues: list[str] = []
    outline_fallback_used = False
    draft_issues: list[str] = []
    draft_assessment: dict[str, Any] | None = None
    model_draft_assessment: dict[str, Any] | None = None
    corrected_draft_assessment: dict[str, Any] | None = None
    style_draft_assessment: dict[str, Any] | None = None
    generation_decision: str | None = None
    correction_reward = 0.0
    correction_selected = False
    correction_count = 0
    outline_repair_count = 0
    style_rewrite_count = 0
    style_rewrite_rejected = False
    strict_fallback_used = False
    rejected_draft_output: str | None = None
    paragraph_result: dict[str, Any] | None = None
    paragraph_mode_used = False
    paragraph_local_repair_count = 0
    paragraph_pass_count = 0
    claim_verification_failures: dict[str, list[str]] = {}
    paragraph_content_results: list[dict[str, Any]] = []
    if args.backend == "prompt":
        target.write_text(context, encoding="utf-8")
        runtime_used = False
        pass_count = 0
        print(f"Saved grounded context to {args.output}")
    else:
        valid_document_ids = {
            result["document"]["document_id"] for result in content_results
        } | planning_document_ids
        supported_outline_roles = (
            set(DOCUMENT_ROLE_ORDERS[args.document_type])
            if args.paragraph_mode
            else (
                {
                    item["role"]
                    for item in causal_scaffold
                    if item.get("status") == "retrieved"
                }
                | {"conclusion"}
                if causal_scaffold
                else set(NARRATIVE_ROLE_ORDER)
            )
        )
        if args.two_pass:
            outline_context = context
            if planning_inventory:
                outline_context += (
                    "\n\n## Planning-only source inventory\n\n"
                    "Use this inventory to decide paragraph coverage and formulate retrieval questions. "
                    "It is not draft evidence; paragraph claims will be checked against separately retrieved spans.\n\n"
                    + planning_inventory
                )
            outline_prompt = build_outline_prompt(
                outline_context, narrative_template, args.document_type
            )
            outline = invoke_generation(
                "outline",
                outline_prompt,
                args.outline_max_tokens,
                min(args.temperature, 0.5),
            ).text
            outline_issues = validate_outline(
                outline,
                minimum_roles=args.min_outline_roles,
                valid_document_ids=valid_document_ids,
                allowed_roles=supported_outline_roles,
                role_order=DOCUMENT_ROLE_ORDERS[args.document_type],
                max_intents=8 if args.paragraph_mode else None,
            )
            if outline_issues:
                repair_prompt = (
                    outline_prompt
                    + "\n\nPrevious outline validation failures:\n- "
                    + "\n- ".join(outline_issues)
                    + "\nReturn a corrected outline only.\nPrevious outline:\n"
                    + outline
                )
                outline = invoke_generation(
                    "outline_repair",
                    repair_prompt,
                    args.outline_max_tokens,
                    min(args.temperature, 0.5),
                    fallback=outline,
                ).text
                outline_repair_count = 1
                outline_issues = validate_outline(
                    outline,
                    minimum_roles=args.min_outline_roles,
                    valid_document_ids=valid_document_ids,
                    allowed_roles=supported_outline_roles,
                    role_order=DOCUMENT_ROLE_ORDERS[args.document_type],
                    max_intents=8 if args.paragraph_mode else None,
                )
            if args.paragraph_mode:
                paragraph_outline = outline
                if outline_issues:
                    # Preserve the model outline for intent parsing so an
                    # invalid outline cannot silently become generic editorial
                    # or scaffold intents. The validation error remains fatal.
                    paragraph_outline = outline
                paragraph_result = generate_paragraph_article(
                    query=args.query,
                    outline=paragraph_outline,
                    scaffold=causal_scaffold,
                    documents=documents,
                    vectors=vectors,
                    embedding_metadata=embedding_metadata,
                    session=content_embedding_session,
                    span_selector=best_source_span,
                    style_profile=format_style_profile(
                        style_results, style_structure_documents
                    ),
                    narrative_template=narrative_template,
                    generate=paragraph_generate,
                    count_prompt_tokens=token_counter,
                    context_token_limit=args.context_token_limit,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    evidence_top_k=args.intent_top_k,
                    minimum_score=args.intent_minimum_score,
                    retrieval_strategy=args.intent_retrieval_strategy,
                    facet_config=intent_facet_config,
                    group_size=args.intent_group_size,
                    max_repairs=args.max_corrections,
                    max_retrieval_retries=args.max_retrieval_retries,
                    minimum_total_words=args.min_draft_words,
                    editorial_template=template,
                    document_type=args.document_type,
                    article_plan=article_plan,
                    article_plan_canonical_by_id=(
                        canonical_by_id if article_plan else None
                    ),
                )
                generation_calls.extend(paragraph_result.get("generation_calls", []))
                generated = paragraph_result["generated"]
                if generated:
                    paragraph_mode_used = True
                    paragraph_pass_count = paragraph_result["pass_count"]
                    paragraph_local_repair_count = paragraph_result["repair_count"]
                    claim_verification_failures = paragraph_result[
                        "verification_failures"
                    ]
                    paragraph_content_results = ledger_content_results(
                        paragraph_result["ledger"]
                    )
                    if paragraph_content_results:
                        content_results = paragraph_content_results
                    if paragraph_result.get("article_plan") is not None:
                        article_plan = paragraph_result["article_plan"]
                        write_json(args.article_plan_output, article_plan)
                else:
                    claim_verification_failures = paragraph_result[
                        "verification_failures"
                    ]
            if not paragraph_mode_used:
                draft_prompt = build_draft_prompt(
                    context,
                    outline,
                    narrative_template,
                    minimum_words=args.min_draft_words,
                )
                generated = invoke_generation(
                    "draft",
                    draft_prompt,
                    args.max_new_tokens,
                    args.temperature,
                    fallback=(
                        build_strict_evidence_fallback(causal_scaffold)
                        if causal_scaffold
                        else build_content_evidence_fallback(content_results)
                    ),
                ).text
        else:
            draft_prompt = context
            generated = invoke_generation(
                "draft_single_pass",
                draft_prompt,
                args.max_new_tokens,
                args.temperature,
                fallback=(
                    build_strict_evidence_fallback(causal_scaffold)
                    if causal_scaffold
                    else build_content_evidence_fallback(content_results)
                ),
            ).text
        draft_issues = validate_draft(
            generated,
            minimum_words=args.min_draft_words,
            minimum_transitions=args.min_transitions,
            content_results=content_results,
            style_results=style_results,
            causal_scaffold=generation_context_scaffold,
        )
        if claim_verification_failures:
            draft_issues.extend(
                f"claim verification failed for {intent_id}: {'; '.join(reasons)}"
                for intent_id, reasons in claim_verification_failures.items()
            )
        draft_assessment = assess_draft_confidence(
            generated,
            content_results=content_results,
            style_results=style_results,
            causal_scaffold=generation_context_scaffold,
            calibration=writing_calibration,
        )
        model_draft_assessment = draft_assessment
        correction_reasons = list(
            dict.fromkeys(draft_issues + draft_assessment["warnings"])
        )
        if (
            args.two_pass
            and not paragraph_mode_used
            and (draft_issues or draft_assessment["decision"] != "accept")
            and args.max_corrections
        ):
            correction_prompt = build_correction_prompt(
                context,
                outline,
                generated,
                correction_reasons or ["raise the evidence-confidence score"],
                narrative_template,
                minimum_words=args.min_draft_words,
            )
            corrected = invoke_generation(
                "correction",
                correction_prompt,
                args.max_new_tokens,
                args.temperature,
                fallback=generated,
            ).text
            correction_count = 1
            corrected_issues = validate_draft(
                corrected,
                minimum_words=args.min_draft_words,
                minimum_transitions=args.min_transitions,
                content_results=content_results,
                style_results=style_results,
            causal_scaffold=generation_context_scaffold,
            )
            corrected_assessment = assess_draft_confidence(
                corrected,
                content_results=content_results,
                style_results=style_results,
            causal_scaffold=generation_context_scaffold,
                calibration=writing_calibration,
            )
            corrected_draft_assessment = corrected_assessment
            correction_reward = round(
                corrected_assessment["score"] - draft_assessment["score"], 4
            )
            if should_select_correction(
                draft_assessment,
                draft_issues,
                corrected_assessment,
                corrected_issues,
            ):
                generated = corrected
                draft_issues = corrected_issues
                draft_assessment = corrected_assessment
                correction_selected = True

        if (
            draft_assessment["decision"] in {"demote_to_strict_fallback", "reject"}
            and causal_scaffold
        ):
            rejected_target = target.with_name(f"{target.stem}.demoted{target.suffix}")
            rejected_target.write_text(generated, encoding="utf-8")
            rejected_draft_output = str(rejected_target)
            generated = build_strict_evidence_fallback(causal_scaffold)
            strict_fallback_used = True
            draft_issues = validate_draft(
                generated,
                content_results=content_results,
                style_results=style_results,
            causal_scaffold=generation_context_scaffold,
            )
            draft_assessment = assess_draft_confidence(
                generated,
                content_results=content_results,
                style_results=style_results,
            causal_scaffold=generation_context_scaffold,
                calibration=writing_calibration,
            )

        if args.style_pass and not strict_fallback_used:
            style_call = invoke_generation(
                "style_rewrite",
                build_style_rewrite_prompt(generated, style_results),
                args.max_new_tokens,
                args.temperature,
            )
            styled = style_call.text
            styled_issues = validate_draft(
                styled,
                content_results=content_results,
                style_results=style_results,
            causal_scaffold=generation_context_scaffold,
            )
            styled_assessment = assess_draft_confidence(
                styled,
                content_results=content_results,
                style_results=style_results,
            causal_scaffold=generation_context_scaffold,
                calibration=writing_calibration,
            )
            style_draft_assessment = styled_assessment
            style_rewrite_count = 1
            if (
                style_call.ok
                and styled_assessment["probabilities"]["style"]
                >= draft_assessment["probabilities"]["style"]
                and styled_assessment["probabilities"]["causality"]
                >= draft_assessment["probabilities"]["causality"] - 0.02
                and styled_assessment["decision"]
                not in {"demote_to_strict_fallback", "reject"}
            ):
                generated = styled
                draft_issues = styled_issues
                draft_assessment = styled_assessment
            else:
                style_rewrite_rejected = True

        generation_decision = (
            "demoted_to_strict_fallback"
            if strict_fallback_used and draft_assessment["decision"] != "reject"
            else draft_assessment["decision"]
        )
        if claim_verification_failures or (
            outline_issues and not outline_fallback_used
        ):
            generation_decision = "reject"
        elif draft_issues and generation_decision == "accept":
            generation_decision = "accept_with_warnings"
        target.write_text(generated, encoding="utf-8")
        runtime_used = runtime is not None
        pass_count = (
            (
                1 + paragraph_pass_count
                if paragraph_mode_used
                else (2 if args.two_pass else 1)
            )
            + outline_repair_count
            + correction_count
            + style_rewrite_count
        )
        generated_tokens = output_token_counter(generated)
        if generation_decision == "reject":
            print(
                f"Saved rejected draft from {args.backend} ({args.model}) to {args.output}"
            )
        elif strict_fallback_used:
            print(
                f"Demoted the {args.backend} draft to a strict evidence fallback "
                f"(confidence={draft_assessment['score']:.2f})"
            )
            print(f"Saved article to {args.output}")
        else:
            print(
                f"Generated article with {args.backend} ({args.model}); "
                f"decision={generation_decision}, confidence={draft_assessment['score']:.2f}"
            )
            print(f"Saved article to {args.output}")
    elapsed = time.perf_counter() - started
    pipeline_attempted = args.backend != "prompt"
    metadata = {
        "document_type": args.document_type,
        "generator_backend": args.backend,
        "model_cache": str(model_cache),
        "generator_model": args.model if runtime_used else None,
        "runtime_used": runtime_used,
        "pipeline_attempted": pipeline_attempted,
        "generator_runtime_error": runtime_error,
        "generation_calls": generation_calls,
        "generation_fallback_used": any(
            call.get("fallback_used") for call in generation_calls
        ),
        "two_pass": bool(args.two_pass and runtime_used),
        "pass_count": pass_count,
        "outline_repair_count": outline_repair_count if pipeline_attempted else 0,
        "correction_count": correction_count if pipeline_attempted else 0,
        "paragraph_mode_enabled": args.paragraph_mode,
        "paragraph_retrieval_strategy": args.intent_retrieval_strategy,
        "paragraph_planning_strategy": (
            args.intent_retrieval_strategy if args.paragraph_mode else None
        ),
        "paragraph_planning_context_isolated": bool(args.paragraph_mode),
        "paragraph_planning_document_ids": sorted(planning_document_ids),
        "paragraph_facet_config": (
            paragraph_result.get("facet_config")
            if paragraph_result
            else (
                {
                    "top_k": intent_facet_config.top_k,
                    "anchor_pool_k": intent_facet_config.anchor_pool_k,
                    "anchor_k": intent_facet_config.anchor_k,
                    "facet_top_k": intent_facet_config.facet_top_k,
                    "rrf_k": intent_facet_config.rrf_k,
                    "base_weight": intent_facet_config.base_weight,
                    "facet_weight": intent_facet_config.facet_weight,
                    "min_anchor_lexical_score": intent_facet_config.min_anchor_lexical_score,
                    "min_anchor_term_hits": intent_facet_config.min_anchor_term_hits,
                }
                if intent_facet_config
                else None
            )
        ),
        "paragraph_mode_used": paragraph_mode_used if pipeline_attempted else False,
        "paragraph_intent_count": len(paragraph_result["ledger"])
        if paragraph_result
        else 0,
        "paragraph_group_count": paragraph_result["group_count"]
        if paragraph_result
        else 0,
        "paragraph_local_repair_count": paragraph_local_repair_count
        if pipeline_attempted
        else 0,
        "paragraph_pass_count": paragraph_pass_count if pipeline_attempted else 0,
        "paragraph_prompt_tokens": paragraph_result["prompt_tokens"]
        if paragraph_result
        else [],
        "paragraph_retrieval_retries": (
            paragraph_result.get("retrieval_retries", []) if paragraph_result else []
        ),
        "paragraph_evidence_ledger": ledger_summary(paragraph_result["ledger"])
        if paragraph_result
        else [],
        "claim_verification_reports": paragraph_result["verification_reports"]
        if paragraph_result
        else [],
        "claim_verifier_raw_outputs": paragraph_result["verifier_outputs"]
        if paragraph_result
        else [],
        "verifier_protocol_diagnostics": (
            paragraph_result.get("verifier_protocol_diagnostics", [])
            if paragraph_result
            else []
        ),
        "paragraph_evidence_trace": (
            [
                {
                    "intent_id": item["intent_id"],
                    "retrieval_trace": item.get("retrieval_trace", []),
                }
                for item in paragraph_result.get("ledger", [])
            ]
            if paragraph_result
            else []
        ),
        "paragraph_evidence_trace_debug": (
            paragraph_result.get("evidence_trace_debug", "")
            if paragraph_result
            else ""
        ),
        "claim_verification_failures": claim_verification_failures
        if pipeline_attempted
        else {},
        "paragraph_invariant_audit": (
            paragraph_result.get("invariant_audit") if paragraph_result else None
        ),
        "paragraph_dependency_evidence_audit": (
            paragraph_result.get("dependency_evidence_audit")
            if paragraph_result
            else None
        ),
        "style_rewrite_count": style_rewrite_count if pipeline_attempted else 0,
        "style_rewrite_rejected": style_rewrite_rejected
        if pipeline_attempted
        else False,
        "strict_fallback_used": strict_fallback_used if pipeline_attempted else False,
        "rejected_draft_output": rejected_draft_output if pipeline_attempted else None,
        "outline_validation_issues": outline_issues if pipeline_attempted else [],
        "outline_fallback_used": outline_fallback_used if pipeline_attempted else False,
        "draft_validation_issues": draft_issues if pipeline_attempted else [],
        "generation_decision": generation_decision if pipeline_attempted else None,
        "draft_confidence": draft_assessment["score"] if draft_assessment else None,
        "draft_assessment": draft_assessment if pipeline_attempted else None,
        "writing_calibration": calibration_summary(writing_calibration),
        "model_draft_assessment": model_draft_assessment
        if pipeline_attempted
        else None,
        "corrected_draft_assessment": corrected_draft_assessment
        if pipeline_attempted
        else None,
        "style_draft_assessment": style_draft_assessment
        if pipeline_attempted
        else None,
        "correction_reward": correction_reward if pipeline_attempted else 0.0,
        "correction_selected": correction_selected if pipeline_attempted else False,
        "validation_passed": (
            not outline_issues
            and not draft_issues
            and not claim_verification_failures
            and generation_decision != "reject"
            if pipeline_attempted
            else None
        ),
        "context_tokens": context_tokens,
        "outline_tokens": output_token_counter(outline) if outline else 0,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "retrieved_content_items": len(content_results),
        "retrieved_style_items": len(style_results),
        "causal_chain_enabled": args.causal_chain,
        "causal_chain_slots": len(causal_scaffold),
        "causal_chain_evidence_items": causal_scaffold_evidence_count(causal_scaffold),
        "causal_chain_assessment": next(
            (
                item["chain_assessment"]
                for item in causal_scaffold
                if item.get("chain_assessment")
            ),
            None,
        ),
        "causal_edge_assessments": [
            edge for item in causal_scaffold for edge in item.get("outgoing_edges", [])
        ],
        "causal_excerpt_tokens": causal_excerpt_tokens,
        "base_context_tokens": base_context_tokens,
        "causal_context_token_delta": context_tokens - base_context_tokens,
        "causal_token_profile": causal_token_profile,
        "content_document_ids": [
            result["document"]["document_id"] for result in content_results
        ],
        "style_document_ids": [
            result["document"]["document_id"] for result in style_results
        ],
        "token_count_encoding": token_count_encoding,
        "narrative_enabled": args.narrative,
        "article_plan_enabled": bool(args.article_plan),
        "article_plan_status": article_plan.get("status") if article_plan else None,
        "article_plan_sha256": article_plan.get("plan_sha256") if article_plan else None,
        "article_plan_errors": article_plan.get("errors", []) if article_plan else [],
        "article_plan_output": args.article_plan_output if article_plan else None,
        **memory_provenance(),
    }
    write_json(args.run_metadata, metadata)
    if args.evidence_trace_output and paragraph_result:
        trace_target = PROJECT_ROOT / args.evidence_trace_output
        trace_target.parent.mkdir(parents=True, exist_ok=True)
        trace_target.write_text(
            paragraph_result.get("evidence_trace_debug", ""), encoding="utf-8"
        )
        print(f"Saved evidence trace to {args.evidence_trace_output}")
    print(f"Saved generation metadata to {args.run_metadata}")
    print(f"Retrieved content items: {len(content_results)}")
    print(f"Retrieved style items: {len(style_results)}")
    if (
        runtime_used
        and generation_decision == "reject"
        and not any(not bool(call.get("ok")) for call in generation_calls)
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
