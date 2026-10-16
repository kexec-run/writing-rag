"""Opt-in graph- and time-aware retrieval over existing retrieval documents.

This module deliberately wraps the current dense-plus-lexical retriever instead
of changing it.  It treats the existing document metadata as a lightweight
conversation graph: message overlap and explicit graph paths are strong links;
same-thread temporal proximity is a weaker link.  The result includes a trace
so structural expansion can be evaluated before it is connected to generation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from writing_rag.runtime.pipeline_common import read_json, write_json
from writing_rag.retrieve.retrieve import EmbeddingSession, retrieve_many


@dataclass(frozen=True)
class GraphRetrievalConfig:
    """Controls relevance-gated structural candidate expansion."""

    top_k: int = 8
    candidate_k: int = 32
    anchor_k: int = 8
    max_expanded_per_seed: int = 12
    max_time_gap_ms: int = 30 * 60 * 1000
    min_base_score_ratio: float = 0.85
    min_base_score: float = 0.2
    same_path_bonus: float = 0.08
    shared_message_bonus: float = 0.12
    temporal_bonus: float = 0.04
    max_structural_bonus: float = 0.05


def _metadata(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("metadata")
    return value if isinstance(value, dict) else {}


def _values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    if value is None:
        return set()
    return {str(value).strip()}


def _timestamp(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _interval_gap(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    left_meta = _metadata(left)
    right_meta = _metadata(right)
    left_start = _timestamp(left_meta, "start_timestamp_ms")
    left_end = _timestamp(left_meta, "end_timestamp_ms")
    right_start = _timestamp(right_meta, "start_timestamp_ms")
    right_end = _timestamp(right_meta, "end_timestamp_ms")
    if None in {left_start, left_end, right_start, right_end}:
        return None
    if left_end < right_start:
        return right_start - left_end
    if right_end < left_start:
        return left_start - right_end
    return 0


def relationship_features(
    seed: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_time_gap_ms: int,
) -> dict[str, Any]:
    """Return explainable links between two retrieval documents.

    Same-thread membership alone is intentionally not a link. A time bound
    prevents broad thread metadata from turning unrelated records into one
    connected component.
    """

    seed_meta = _metadata(seed)
    candidate_meta = _metadata(candidate)
    seed_thread = str(seed_meta.get("source_thread_id") or "")
    candidate_thread = str(candidate_meta.get("source_thread_id") or "")
    same_thread = bool(seed_thread and seed_thread == candidate_thread)

    seed_paths = _values(seed_meta.get("graph_path_id"))
    candidate_paths = _values(candidate_meta.get("graph_path_id"))
    same_path = bool(seed_paths & candidate_paths)

    shared_messages = _values(seed_meta.get("message_ids")) & _values(
        candidate_meta.get("message_ids")
    )
    gap_ms = _interval_gap(seed, candidate)
    temporally_adjacent = (
        same_thread and gap_ms is not None and gap_ms <= max_time_gap_ms
    )

    relation_types: list[str] = []
    if same_path:
        relation_types.append("same_graph_path")
    if shared_messages:
        relation_types.append("shared_message")
    if temporally_adjacent:
        relation_types.append("same_thread_time_window")

    return {
        "same_thread": same_thread,
        "same_graph_path": same_path,
        "shared_message_ids": sorted(shared_messages),
        "time_gap_ms": gap_ms,
        "temporally_adjacent": temporally_adjacent,
        "relation_types": relation_types,
    }


def _structural_score(
    features: dict[str, Any], config: GraphRetrievalConfig
) -> float:
    score = 0.0
    if features["same_graph_path"]:
        score += config.same_path_bonus
    if features["shared_message_ids"]:
        score += config.shared_message_bonus
    if features["temporally_adjacent"]:
        score += config.temporal_bonus
    return score


def _is_linked(features: dict[str, Any]) -> bool:
    return bool(
        features["same_graph_path"]
        or features["shared_message_ids"]
        or features["temporally_adjacent"]
    )


def _document_id(document: dict[str, Any]) -> str:
    return str(document.get("document_id") or "")


def _expansion_strategy(relation_types: set[str]) -> str:
    if relation_types == {"same_thread_time_window"}:
        return "graph_time_expansion"
    if relation_types == {"same_graph_path"}:
        return "graph_path_expansion"
    if relation_types == {"shared_message"}:
        return "graph_shared_message_expansion"
    return "graph_structural_expansion"


def graph_retrieve_many(
    queries: list[str],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    *,
    config: GraphRetrievalConfig | None = None,
    view: str | None = None,
    collection: str | None = None,
    embedding_metadata: dict[str, Any] | None = None,
    session: EmbeddingSession | None = None,
) -> list[list[dict[str, Any]]]:
    """Retrieve candidates and admit only relevant structural expansions.

    Structural links propose candidates. The base semantic/lexical score remains
    the admission floor, and every admitted expansion competes with direct
    candidates in one final ranking. No expansion quota is guaranteed.
    """

    config = config or GraphRetrievalConfig()
    if config.top_k <= 0 or not queries:
        return [[] for _ in queries]
    if config.candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    if config.anchor_k <= 0:
        raise ValueError("anchor_k must be greater than zero")
    if config.max_expanded_per_seed < 0:
        raise ValueError("max_expanded_per_seed cannot be negative")
    if not 0.0 <= config.min_base_score_ratio <= 1.0:
        raise ValueError("min_base_score_ratio must be between zero and one")
    if config.min_base_score < 0:
        raise ValueError("min_base_score cannot be negative")
    if config.max_structural_bonus < 0:
        raise ValueError("max_structural_bonus cannot be negative")

    baseline = retrieve_many(
        queries,
        documents,
        vectors,
        top_k=len(documents),
        view=view,
        collection=collection,
        embedding_metadata=embedding_metadata,
        session=session,
    )

    results: list[list[dict[str, Any]]] = []
    for ranked in baseline:
        if not ranked:
            results.append([])
            continue

        pool = ranked[: min(config.candidate_k, len(ranked))]
        anchors = pool[: min(config.anchor_k, len(pool))]
        pool_ids = {_document_id(item["document"]) for item in pool}
        anchor_ids = {_document_id(item["document"]) for item in anchors}
        anchor_floor = max(
            config.min_base_score,
            anchors[0]["score"] * config.min_base_score_ratio,
        )
        promotion_floor = (
            pool[min(config.top_k, len(pool)) - 1]["score"]
            if config.min_base_score_ratio > 0.0
            else config.min_base_score
        )
        expanded: dict[str, dict[str, Any]] = {}

        for seed in anchors:
            seed_document = seed["document"]
            related: list[
                tuple[float, float, int, str, dict[str, Any], dict[str, Any]]
            ] = []
            for item in ranked:
                candidate = item["document"]
                candidate_id = _document_id(candidate)
                if candidate_id in anchor_ids:
                    continue
                admission_floor = (
                    config.min_base_score
                    if candidate_id in pool_ids
                    else anchor_floor
                )
                if item["score"] < admission_floor:
                    continue
                features = relationship_features(
                    seed_document,
                    candidate,
                    max_time_gap_ms=config.max_time_gap_ms,
                )
                if not _is_linked(features):
                    continue
                structural = _structural_score(features, config)
                gap = features["time_gap_ms"]
                related.append(
                    (
                        -structural,
                        -item["score"],
                        gap if gap is not None else 2**63 - 1,
                        candidate_id,
                        item,
                        features,
                    )
                )

            related.sort(key=lambda row: row[:4])
            for _, _, _, candidate_id, item, features in related[
                : config.max_expanded_per_seed
            ]:
                raw_structural = _structural_score(features, config)
                trace = expanded.setdefault(
                    candidate_id,
                    {
                        "seed_document_ids": [],
                        "relation_types": set(),
                        "relations": [],
                        "item": item,
                    },
                )
                seed_id = _document_id(seed_document)
                if seed_id not in trace["seed_document_ids"]:
                    trace["seed_document_ids"].append(seed_id)
                trace["relation_types"].update(features["relation_types"])
                trace["relations"].append(
                    {
                        "seed_document_id": seed_id,
                        **features,
                        "structural_score": round(raw_structural, 6),
                    }
                )
                if item["score"] > trace["item"]["score"]:
                    trace["item"] = item

        selected_candidates: list[dict[str, Any]] = []
        for item in pool:
            candidate_id = _document_id(item["document"])
            trace = expanded.get(candidate_id)
            if trace is None or trace["item"]["score"] < promotion_floor:
                selected_candidates.append(
                    {
                        **item,
                        "base_score": item["score"],
                        "structural_score": 0.0,
                        "retrieval_strategy": "dense_lexical_seed",
                        "graph_trace": {
                            "seed_document_ids": [],
                            "relation_types": [],
                            "relations": [],
                            "selection_reason": "direct_candidate",
                        },
                    }
                )
                continue
            relation_types = set(trace["relation_types"])
            raw_structural = max(
                relation["structural_score"] for relation in trace["relations"]
            )
            graph_bonus = min(raw_structural, config.max_structural_bonus)
            selected_candidates.append(
                {
                    **trace["item"],
                    "base_score": trace["item"]["score"],
                    "score": round(trace["item"]["score"] + graph_bonus, 6),
                    "structural_score": round(raw_structural, 6),
                    "graph_bonus": round(graph_bonus, 6),
                    "retrieval_strategy": _expansion_strategy(relation_types),
                    "graph_trace": {
                        "seed_document_ids": sorted(trace["seed_document_ids"]),
                        "relation_types": sorted(relation_types),
                        "relations": trace["relations"],
                        "selection_reason": (
                            "base_score_above_min_floor_and_structurally_linked"
                        ),
                        "anchor_floor": round(anchor_floor, 6),
                        "candidate_pool": True,
                    },
                }
            )

        for candidate_id, trace in expanded.items():
            if candidate_id in pool_ids:
                continue
            item = trace["item"]
            if item["score"] < promotion_floor:
                continue
            relation_types = set(trace["relation_types"])
            raw_structural = max(
                relation["structural_score"] for relation in trace["relations"]
            )
            graph_bonus = min(raw_structural, config.max_structural_bonus)
            selected_candidates.append(
                {
                    **item,
                    "base_score": item["score"],
                    "score": round(item["score"] + graph_bonus, 6),
                    "structural_score": round(raw_structural, 6),
                    "graph_bonus": round(graph_bonus, 6),
                    "retrieval_strategy": _expansion_strategy(relation_types),
                    "graph_trace": {
                        "seed_document_ids": sorted(trace["seed_document_ids"]),
                        "relation_types": sorted(relation_types),
                        "relations": trace["relations"],
                        "selection_reason": (
                            "base_score_above_anchor_floor_and_structurally_linked"
                        ),
                        "anchor_floor": round(anchor_floor, 6),
                        "candidate_pool": False,
                    },
                }
            )

        selected_candidates.sort(
            key=lambda item: (-item["score"], _document_id(item["document"]))
        )
        selected = selected_candidates[: config.top_k]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        results.append(selected)


    return results


def graph_retrieve(
    query: str,
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Single-query convenience wrapper."""

    return graph_retrieve_many([query], documents, vectors, **kwargs)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--vectors", default="data/embeddings/vectors.json")
    parser.add_argument("--metadata", default="data/embeddings/metadata.json")
    parser.add_argument("--embedding-device", choices=("cpu", "mps"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--anchor-k", type=int, default=8)
    parser.add_argument("--min-base-score-ratio", type=float, default=0.85)
    parser.add_argument("--max-time-gap-minutes", type=int, default=30)
    parser.add_argument("--collection")
    parser.add_argument("--view")
    parser.add_argument("--output")
    args = parser.parse_args()

    documents = read_json(args.documents)
    vectors = read_json(args.vectors)
    metadata = read_json(args.metadata)
    if args.embedding_device:
        metadata = {**metadata, "device": args.embedding_device}
    config = GraphRetrievalConfig(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        anchor_k=args.anchor_k,
        min_base_score_ratio=args.min_base_score_ratio,
        max_time_gap_ms=args.max_time_gap_minutes * 60 * 1000,
    )
    results = graph_retrieve(
        args.query,
        documents,
        vectors,
        config=config,
        collection=args.collection,
        view=args.view,
        embedding_metadata=metadata,
    )
    payload = {"query": args.query, "results": results}
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
