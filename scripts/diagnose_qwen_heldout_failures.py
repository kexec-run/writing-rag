"""Classify Q0 held-out retrieval failures using frozen evidence and traces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline_common import read_json, write_json


def eligible_content_ids(documents: list[dict[str, Any]]) -> set[str]:
    from retrieve import is_content_evidence_candidate

    return {
        document["document_id"]
        for document in documents
        if document.get("collection") == "content_memory"
        and document.get("view") == "content"
        and is_content_evidence_candidate(document)
    }


def diagnose_query(
    label: dict[str, Any],
    *,
    dense_row: dict[str, Any],
    hybrid_row: dict[str, Any],
    eligible_ids: set[str],
    vector_ids: set[str],
) -> dict[str, Any]:
    query_id = str(label.get("id", label["query"]))
    if label.get("evidence_absent"):
        return {
            "id": query_id,
            "query": label["query"],
            "status": "evidence_absent",
            "classification": "A_evidence_absent_from_corpus",
            "model_change_plausible": False,
            "evidence_status": "no labeled supporting evidence",
        }

    relevant_ids = set(label["relevant_document_ids"])
    missing_from_docs = sorted(relevant_ids - eligible_ids)
    missing_from_vectors = sorted(relevant_ids - vector_ids)
    dense_rank = {row["document_id"]: row["rank"] for row in dense_row["ranking"]}
    hybrid_rank = {row["document_id"]: row["rank"] for row in hybrid_row["ranking"]}
    classifications: list[str] = []
    model_change = False
    details: list[dict[str, Any]] = []
    for document_id in sorted(relevant_ids):
        if document_id in missing_from_vectors:
            classification = "B_evidence_lost_during_ingestion"
            plausible = False
        elif document_id in missing_from_docs:
            classification = "H_source_filter_removed_relevant_chunk"
            plausible = False
        elif dense_rank.get(document_id) is None:
            classification = "I_relevant_chunk_below_rank_100"
            plausible = False
        elif dense_rank[document_id] <= 10 and (
            hybrid_rank.get(document_id) is None or hybrid_rank[document_id] > 10
        ):
            classification = "F_lexical_scoring_suppressed_dense_candidate"
            plausible = False
        elif hybrid_rank.get(document_id) is None or hybrid_rank[document_id] > 10:
            classification = "E_dense_semantic_ranking_failure"
            plausible = True
            model_change = True
        else:
            classification = "no_top10_failure_for_this_evidence"
            plausible = False
        classifications.append(classification)
        details.append(
            {
                "document_id": document_id,
                "dense_rank": dense_rank.get(document_id),
                "hybrid_rank": hybrid_rank.get(document_id),
                "classification": classification,
                "model_change_plausible": plausible,
            }
        )

    failed = all(item["classification"] != "no_top10_failure_for_this_evidence" for item in details)
    primary = next(
        (classification for classification in classifications if classification != "no_top10_failure_for_this_evidence"),
        "no_failure",
    )
    return {
        "id": query_id,
        "query": label["query"],
        "status": "failed_top10" if failed else "some_supporting_evidence_in_top10",
        "classification": primary,
        "classifications": sorted(set(classifications)),
        "model_change_plausible": model_change,
        "evidence_status": "labeled supporting evidence exists and is ingested",
        "evidence_details": details,
        "supporting_spans": label.get("supporting_spans", {}),
        "critical_markers": label.get("critical_markers", []),
    }


def build_report(diagnoses: list[dict[str, Any]]) -> str:
    lines = [
        "# Held-out Qwen failure diagnosis",
        "",
        "This classifies Q0 retrieval failures after the Qwen adoption gate. It distinguishes corpus, ingestion, filter, rank, and scoring causes; it does not treat every failure as an embedding-model problem.",
        "",
        "| Query | Status | Classification | Model change plausible? |",
        "|---|---|---|---:|",
    ]
    for item in diagnoses:
        lines.append(
            f"| {item['id']} | {item['status']} | {item['classification']} | {item['model_change_plausible']} |"
        )
    lines.extend(["", "## Details", ""])
    for item in diagnoses:
        lines.append(f"### {item['id']}")
        lines.append("")
        lines.append(item.get("query", ""))
        lines.append("")
        if item.get("evidence_details"):
            for detail in item["evidence_details"]:
                lines.append(
                    f"- `{detail['document_id']}`: dense rank `{detail['dense_rank']}`, hybrid rank `{detail['hybrid_rank']}`, `{detail['classification']}`."
                )
        else:
            lines.append(f"- {item['evidence_status']}.")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/evals/retrieval-quality-heldout-v1.json")
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--vectors", default="data/embeddings/vectors.json")
    parser.add_argument(
        "--results",
        default="data/generated/qwen-query-encoding-v1/held-out/Q0_plain/retrieval-results.json",
    )
    parser.add_argument(
        "--output",
        default="data/generated/qwen-query-encoding-v1/comparison/failure-diagnosis",
    )
    args = parser.parse_args()
    labels = read_json(args.labels)
    documents = read_json(args.documents)
    vectors = read_json(args.vectors)
    results = read_json(args.results)
    dense_rows = {row["id"]: row for row in results["modes"]["dense_only"]["queries"]}
    hybrid_rows = {row["id"]: row for row in results["modes"]["hybrid"]["queries"]}
    eligible_ids = eligible_content_ids(documents)
    vector_ids = {item["document_id"] for item in vectors}
    diagnoses = [
        diagnose_query(
            label,
            dense_row=dense_rows[str(label.get("id", label["query"]))],
            hybrid_row=hybrid_rows[str(label.get("id", label["query"]))],
            eligible_ids=eligible_ids,
            vector_ids=vector_ids,
        )
        for label in labels["queries"]
    ]
    output = {
        "version": "qwen-heldout-failure-diagnosis-v1",
        "source_results": str(args.results),
        "diagnoses": diagnoses,
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "failure-diagnosis.json", output)
    (output_dir / "failure-diagnosis.md").write_text(
        build_report(diagnoses), encoding="utf-8"
    )
    print(output_dir / "failure-diagnosis.md")


if __name__ == "__main__":
    main()
