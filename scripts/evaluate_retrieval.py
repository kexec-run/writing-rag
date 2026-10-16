"""Evaluate frozen retrieval labels against baseline and alternative rankings."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

from graph_retrieve import GraphRetrievalConfig, graph_retrieve
from facet_retrieve import FacetRetrievalConfig, retrieve_faceted
from model_paths import configure_model_cache
from pipeline_common import read_json, write_json
from retrieve import (
    EmbeddingSession,
    is_content_evidence_candidate,
    retrieve,
    retrieve_multi_query,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ranking_metrics(
    results: list[dict[str, Any]],
    relevant_document_ids: set[str],
    *,
    k: int,
) -> dict[str, float | int]:
    ranked_ids = [
        item["document"]["document_id"] for item in results[:k]
    ]
    hits = [document_id in relevant_document_ids for document_id in ranked_ids]
    relevant_hits = sum(hits)
    reciprocal_rank = next(
        (1.0 / index for index, hit in enumerate(hits, start=1) if hit),
        0.0,
    )
    dcg = sum(
        (1.0 / math.log2(index + 1))
        for index, hit in enumerate(hits, start=1)
        if hit
    )
    ideal_hits = min(k, len(relevant_document_ids))
    ideal_dcg = sum(
        1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1)
    )
    return {
        "precision_at_k": relevant_hits / max(1, k),
        "recall_at_k": relevant_hits / max(1, len(relevant_document_ids)),
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "relevant_hits": relevant_hits,
        "returned": len(ranked_ids),
    }


def negative_control_metrics(
    results: list[dict[str, Any]],
    document_ids: set[str],
    *,
    k: int,
) -> dict[str, Any]:
    """Report whether named negative controls enter the evaluated top-k."""
    ranks = {
        item["document"]["document_id"]: rank
        for rank, item in enumerate(results, start=1)
        if item["document"]["document_id"] in document_ids
    }
    top_k_ids = set(ranks).intersection(
        item["document"]["document_id"] for item in results[:k]
    )
    return {
        "document_ids": sorted(document_ids),
        "ranks": {document_id: ranks[document_id] for document_id in sorted(ranks)},
        "top_k_hits": sorted(top_k_ids),
        "top_k_rate": len(top_k_ids) / max(1, len(document_ids)),
        "first_rank": min(ranks.values(), default=None),
    }


def evaluate_query(
    query: str,
    relevant_document_ids: set[str],
    baseline: list[dict[str, Any]],
    graph_aware: list[dict[str, Any]],
    faceted: list[dict[str, Any]],
    *,
    k: int,
    hard_negative_document_ids: set[str] | None = None,
    control_groups: dict[str, set[str]] | None = None,
    multi_query: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hard_negative_document_ids = hard_negative_document_ids or set()
    control_groups = control_groups or {}
    baseline_ids = {
        item["document"]["document_id"] for item in baseline[:k]
    }
    graph_ids = {
        item["document"]["document_id"] for item in graph_aware[:k]
    }
    faceted_ids = {
        item["document"]["document_id"] for item in faceted[:k]
    }
    expansions = [
        item
        for item in graph_aware[:k]
        if item.get("retrieval_strategy", "").startswith("graph_")
    ]
    structural_only_noise = [
        item["document"]["document_id"]
        for item in expansions
        if item["document"]["document_id"] not in relevant_document_ids
    ]
    facet_expansions = [
        item
        for item in faceted[:k]
        if item.get("retrieval_strategy") == "facet_rrf"
    ]
    facet_anchor_ids = sorted(
        {
            anchor_id
            for item in facet_expansions
            for anchor_id in item.get("facet_trace", {}).get(
                "anchor_document_ids", []
            )
        }
    )

    def method_controls(results: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "hard_negative": negative_control_metrics(
                results,
                hard_negative_document_ids,
                k=k,
            ),
            "groups": {
                name: negative_control_metrics(results, ids, k=k)
                for name, ids in sorted(control_groups.items())
            },
        }

    multi_query_report = None
    if multi_query is not None:
        multi_query_report = {
            "document_ids": [
                item["document"]["document_id"] for item in multi_query[:k]
            ],
            "metrics": ranking_metrics(
                multi_query,
                relevant_document_ids,
                k=k,
            ),
            "controls": method_controls(multi_query),
        }

    return {
        "query": query,
        "relevant_document_ids": sorted(relevant_document_ids),
        "hard_negative_document_ids": sorted(hard_negative_document_ids),
        "control_groups": {
            name: sorted(ids) for name, ids in sorted(control_groups.items())
        },
        "multi_query": multi_query_report,
        "baseline": {
            "document_ids": [
                item["document"]["document_id"] for item in baseline[:k]
            ],
            "metrics": ranking_metrics(
                baseline,
                relevant_document_ids,
                k=k,
            ),
            "controls": method_controls(baseline),
        },
        "graph_aware": {
            "document_ids": [
                item["document"]["document_id"] for item in graph_aware[:k]
            ],
            "metrics": ranking_metrics(
                graph_aware,
                relevant_document_ids,
                k=k,
            ),
            "graph_expansions": len(expansions),
            "structural_only_noise": structural_only_noise,
            "controls": method_controls(graph_aware),
        },
        "faceted": {
            "document_ids": [
                item["document"]["document_id"] for item in faceted[:k]
            ],
            "metrics": ranking_metrics(
                faceted,
                relevant_document_ids,
                k=k,
            ),
            "facet_expansions": len(facet_expansions),
            "facet_anchor_ids": facet_anchor_ids,
            "facet_only_noise": [
                item["document"]["document_id"]
                for item in facet_expansions
                if item["document"]["document_id"] not in relevant_document_ids
            ],
            "controls": method_controls(faceted),
        },
        "relevant_promoted_into_top_k": sorted(
            (graph_ids - baseline_ids) & relevant_document_ids
        ),
        "relevant_faceted_promoted_into_top_k": sorted(
            (faceted_ids - baseline_ids) & relevant_document_ids
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", default="data/evals/retrieval-quality-v1.json"
    )
    parser.add_argument(
        "--documents", default="data/chroma_docs/chroma_documents.json"
    )
    parser.add_argument("--vectors", default="data/embeddings/vectors.json")
    parser.add_argument("--metadata", default="data/embeddings/metadata.json")
    parser.add_argument(
        "--query-expansions",
        help=(
            "Optional query-only expansion JSON. Never reads query_variants "
            "from the relevance-label file."
        ),
    )
    parser.add_argument("--embedding-device", choices=("cpu", "mps"))
    parser.add_argument("--model-cache", default="models/weights/huggingface")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="/private/tmp/retrieval-quality-v1.json")
    args = parser.parse_args()

    started = time.perf_counter()
    model_cache = configure_model_cache(args.model_cache)
    labels_path = Path(args.labels)
    labels = read_json(labels_path)
    documents = read_json(args.documents)
    vectors = read_json(args.vectors)
    metadata = read_json(args.metadata)
    expansion_by_id: dict[str, list[str]] = {}
    if args.query_expansions:
        expansion_data = read_json(args.query_expansions)
        expansion_by_id = {
            str(item["id"]): [str(value) for value in item["variants"]]
            for item in expansion_data.get("queries", [])
        }
        label_ids = {str(item.get("id", item["query"])) for item in labels["queries"]}
        unknown_expansion_ids = set(expansion_by_id) - label_ids
        if unknown_expansion_ids:
            raise ValueError(
                "query expansion IDs not present in labels: "
                + ", ".join(sorted(unknown_expansion_ids))
            )
    if args.embedding_device:
        metadata = {**metadata, "device": args.embedding_device}
    content_documents = [
        document
        for document in documents
        if document.get("collection") == "content_memory"
        and document.get("view") == "content"
    ]
    eligible_content_document_count = sum(
        is_content_evidence_candidate(document) for document in content_documents
    )

    known_document_ids = {
        document["document_id"] for document in content_documents
    }
    session = EmbeddingSession(vectors, metadata)
    output: dict[str, Any] = {
        "version": labels["version"],
        "k": args.k,
        "model_cache": str(model_cache),
        "embedding_metadata": metadata,
        "evaluation_provenance": {
            "labels_path": str(labels_path),
            "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "documents_path": str(Path(args.documents)),
            "vectors_path": str(Path(args.vectors)),
            "metadata_path": str(Path(args.metadata)),
            "python_executable": sys.executable,
            "source_filter": "exclude assistant-only ChatGPT content-view documents",
            "content_document_count": len(content_documents),
            "eligible_content_document_count": eligible_content_document_count,
            "excluded_content_document_count": (
                len(content_documents) - eligible_content_document_count
            ),
            "query_expansion": {
                "enabled": bool(args.query_expansions),
                "path": args.query_expansions,
                "mode": "query_only_frozen" if args.query_expansions else "disabled",
            },
        },
        "retrieval_configs": {
            "graph_aware": {"top_k": args.k},
            "faceted": {
                "top_k": args.k,
                "anchor_pool_k": 10,
                "anchor_k": 2,
                "facet_top_k": 20,
                "base_weight": 1.0,
                "facet_weight": 1.0,
                "min_anchor_lexical_score": 0.5,
            },
            "multi_query": (
                {"candidate_k": max(args.k * 12, 80)}
                if args.query_expansions
                else None
            ),
        },
        "queries": [],
    }
    for item in labels["queries"]:
        query_id = str(item.get("id", item["query"]))
        relevant_ids = set(item["relevant_document_ids"])
        hard_negative_ids = set(item.get("hard_negative_document_ids", []))
        if relevant_ids.intersection(hard_negative_ids):
            raise ValueError(
                f"label overlap for {item.get('id', item['query'])}"
            )
        control_groups = {
            name: set(ids)
            for name, ids in item.get("control_groups", {}).items()
        }
        unknown_ids = (
            relevant_ids
            | hard_negative_ids
            | set().union(*control_groups.values())
        ) - known_document_ids
        if unknown_ids:
            raise ValueError(
                f"unknown content document IDs for {item.get('id', item['query'])}: "
                f"{sorted(unknown_ids)}"
            )
        baseline = retrieve(
            item["query"],
            content_documents,
            vectors,
            top_k=args.k,
            collection="content_memory",
            view="content",
            embedding_metadata=metadata,
            session=session,
        )
        graph_aware = graph_retrieve(
            item["query"],
            content_documents,
            vectors,
            config=GraphRetrievalConfig(top_k=args.k),
            collection="content_memory",
            view="content",
            embedding_metadata=metadata,
            session=session,
        )
        faceted = retrieve_faceted(
            item["query"],
            content_documents,
            vectors,
            config=FacetRetrievalConfig(top_k=args.k),
            collection="content_memory",
            view="content",
            embedding_metadata=metadata,
            session=session,
        )
        multi_query = None
        if args.query_expansions:
            variants = expansion_by_id.get(query_id)
            if not variants:
                raise ValueError(f"missing query-only expansion for {query_id}")
            multi_query = retrieve_multi_query(
                variants,
                content_documents,
                vectors,
                top_k=args.k,
                candidate_k=max(args.k * 12, 80),
                collection="content_memory",
                view="content",
                embedding_metadata=metadata,
                session=session,
            )
        output["queries"].append(
            evaluate_query(
                item["query"],
                relevant_ids,
                baseline,
                graph_aware,
                faceted,
                k=args.k,
                hard_negative_document_ids=hard_negative_ids,
                control_groups=control_groups,
                multi_query=multi_query,
            )
        )
    output["evaluation_provenance"]["elapsed_seconds"] = round(
        time.perf_counter() - started, 3
    )
    output_path = Path(args.output)
    write_json(output_path, output)
    print(output_path)


if __name__ == "__main__":
    main()
