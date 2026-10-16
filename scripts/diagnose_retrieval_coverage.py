"""Measure corpus availability, retrieval recall, and evidence usage separately.

The reference is used only for diagnosis. It is never passed into generation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from compare_narrative import narrative_chunks
from pipeline_common import read_json, write_json
from retrieve import EmbeddingSession, retrieve_many


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
MDX_WIRING_RE = re.compile(r"(?m)^\s*(?:import|export)\s+[^\n]+\n?")
MDX_COMPONENT_RE = re.compile(r"(?m)^\s*<[A-Z][^>\n]*/>\s*\n?")


def clean_reference(path: Path) -> str:
    """Remove presentation wiring while retaining article prose and code."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".mdx":
        text = FRONTMATTER_RE.sub("", text, count=1)
        text = MDX_WIRING_RE.sub("", text)
        text = MDX_COMPONENT_RE.sub("", text)
    return text.strip()


def reference_units(text: str) -> list[dict[str, Any]]:
    """Return paragraph-sized reference units, excluding labels."""

    return [
        unit
        for unit in narrative_chunks(text)
        if unit.get("kind") != "label" and len(WORD_RE.findall(unit["text"])) >= 6
    ]


def unit_query(unit: dict[str, Any]) -> str:
    heading = str(unit.get("heading") or "").strip()
    return f"{heading}\n{unit['text']}".strip()


def generation_sets(metadata: dict[str, Any]) -> dict[str, Any]:
    """Recover document-level evidence sets from a generation metadata file."""

    retrieved: set[str] = set(metadata.get("content_document_ids", []))
    ledger = metadata.get("paragraph_evidence_ledger", [])
    for intent in ledger:
        retrieved.update(intent.get("evidence_ids", []))

    invariant = metadata.get("paragraph_invariant_audit") or {}
    differences = invariant.get("differences") or {}
    unused: set[str] = set()
    for ids in (differences.get("retrieved_refs_unused") or {}).values():
        unused.update(ids)
    used = retrieved - unused

    failures = metadata.get("claim_verification_failures") or {}
    unsupported_claims = sum(
        len(reasons)
        for intent_id, reasons in failures.items()
        if intent_id != "article"
    )
    unbacked_claims = sum(
        len(values) for values in (differences.get("unbacked_claims") or {}).values()
    )
    introduced_atoms = sum(
        len(values)
        for values in (differences.get("introduced_atoms_by_intent") or {}).values()
    )

    return {
        "usage_known": bool(ledger or differences),
        "retrieved_document_ids": sorted(retrieved),
        "used_document_ids": sorted(used),
        "unused_retrieved_document_ids": sorted(unused),
        "unsupported_claim_count": unsupported_claims,
        "unbacked_claim_count": unbacked_claims,
        "introduced_atom_count": introduced_atoms,
    }


def edge_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    edges = metadata.get("causal_edge_assessments") or []
    supported = [edge for edge in edges if edge.get("status") == "reasonable_inference"]
    evidence_backed = [edge for edge in edges if edge.get("evidence_refs")]
    return {
        "edge_count": len(edges),
        "supported_edge_count": len(supported),
        "evidence_backed_edge_count": len(evidence_backed),
        "demoted_edge_count": sum(
            edge.get("status") != "reasonable_inference" for edge in edges
        ),
        "probabilities": [edge.get("support_probability", 0.0) for edge in edges],
    }


def diagnose(
    *,
    reference: str,
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any],
    query: str,
    generation_metadata: dict[str, Any] | None = None,
    top_k: int = 5,
    availability_threshold: float = 0.45,
) -> dict[str, Any]:
    """Produce proxy set coverage and actual generation evidence accounting.

    ``reference_unit`` is the smallest available evaluation unit here, so corpus
    availability is necessarily a proxy: a unit is considered available when
    its best hybrid retrieval score crosses ``availability_threshold``. The
    report preserves the scores so the threshold can be recalibrated later.
    """

    content_documents = [
        document
        for document in documents
        if document.get("collection") == "content_memory"
    ]
    units = reference_units(reference)
    session = EmbeddingSession(vectors, embedding_metadata)
    result_sets = retrieve_many(
        [unit_query(unit) for unit in units],
        content_documents,
        vectors,
        top_k=max(top_k, 1),
        collection="content_memory",
        embedding_metadata=embedding_metadata,
        session=session,
    )

    actual = generation_sets(generation_metadata or {})
    retrieved_ids = set(actual["retrieved_document_ids"])
    used_ids = set(actual["used_document_ids"])
    rows: list[dict[str, Any]] = []
    for unit, results in zip(units, result_sets):
        best = results[0] if results else None
        candidate_ids = [result["document"]["document_id"] for result in results]
        available = bool(best and best["score"] >= availability_threshold)
        retrieved = bool(retrieved_ids & set(candidate_ids))
        used = bool(used_ids & set(candidate_ids))
        rows.append(
            {
                "reference_unit": unit,
                "query": unit_query(unit),
                "best_score": best["score"] if best else 0.0,
                "best_cosine_score": best["cosine_score"] if best else 0.0,
                "best_lexical_score": best["lexical_score"] if best else 0.0,
                "best_document_id": (best["document"]["document_id"] if best else None),
                "candidate_document_ids": candidate_ids,
                "available_proxy": available,
                "retrieved_by_generation": retrieved,
                "used_by_generation": used,
            }
        )

    reference_count = len(rows)
    available_count = sum(row["available_proxy"] for row in rows)
    retrieved_count = sum(row["retrieved_by_generation"] for row in rows)
    retrieved_available_count = sum(
        row["available_proxy"] and row["retrieved_by_generation"] for row in rows
    )
    used_count = sum(row["used_by_generation"] for row in rows)
    coverage = available_count / max(1, reference_count)
    retrieval_recall = retrieved_available_count / max(1, available_count)
    evidence_usage = used_count / max(1, retrieved_count)

    report = {
        "method": {
            "reference_units": "heading-aware paragraphs excluding labels",
            "retrieval_score": "0.65*dense_cosine + 0.35*query_term_overlap",
            "availability_threshold": availability_threshold,
            "reference_used_for_generation": False,
        },
        "query": query,
        "corpus": {
            "document_count": len(content_documents),
            "document_ids": [document["document_id"] for document in content_documents],
        },
        "set_accounting": {
            "reference_unit_count": reference_count,
            "available_proxy_count": available_count,
            "generation_retrieved_count": retrieved_count,
            "generation_retrieved_available_count": retrieved_available_count,
            "generation_used_count": used_count,
            "source_coverage_proxy": round(coverage, 4),
            "retrieval_recall_given_available_proxy": round(retrieval_recall, 4),
            "evidence_usage_given_retrieved": round(evidence_usage, 4),
        },
        "generation_sets": actual,
        "causal_edges": edge_summary(generation_metadata or {}),
        "units": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--generation-metadata")
    parser.add_argument("--output", default="data/generated/retrieval-coverage.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--availability-threshold", type=float, default=0.45)
    parser.add_argument(
        "--embedding-device",
        help="Override the device recorded in embedding metadata, for example cpu.",
    )
    args = parser.parse_args()

    generation_metadata = (
        read_json(args.generation_metadata) if args.generation_metadata else None
    )
    embedding_metadata = read_json(args.metadata)
    if args.embedding_device:
        embedding_metadata = {
            **embedding_metadata,
            "device": args.embedding_device,
        }
    report = diagnose(
        reference=clean_reference(Path(args.reference)),
        documents=read_json(args.documents),
        vectors=read_json(args.vectors),
        embedding_metadata=embedding_metadata,
        query=args.query,
        generation_metadata=generation_metadata,
        top_k=args.top_k,
        availability_threshold=args.availability_threshold,
    )
    write_json(args.output, report)
    summary = report["set_accounting"]
    print(f"Reference units: {summary['reference_unit_count']}")
    print(f"Available proxy: {summary['available_proxy_count']}")
    print(f"Retrieved by generation: {summary['generation_retrieved_count']}")
    print(f"Used by generation: {summary['generation_used_count']}")
    print(f"Source coverage proxy: {summary['source_coverage_proxy']:.3f}")
    print(
        "Retrieval recall given available proxy: "
        f"{summary['retrieval_recall_given_available_proxy']:.3f}"
    )
    print(
        f"Evidence usage given retrieved: {summary['evidence_usage_given_retrieved']:.3f}"
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
