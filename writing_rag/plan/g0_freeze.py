"""Frozen G0 evidence pool identity artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from writing_rag.plan.source_hydration import sha256_text

G0_SCHEMA = "g0-freeze-v0"


def query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def metadata_sha256(metadata: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(metadata), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def build_g0_artifact(
    query: str,
    candidates: list[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    budget: int,
) -> dict[str, Any]:
    candidate_records = []
    document_ids: list[str] = []
    allowed_witness_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        document_id = str(candidate.get("document_id", ""))
        text = str(candidate.get("text", ""))
        text_hash = str(candidate.get("text_sha256", "")) or sha256_text(text)
        span_id = str(candidate.get("span_id", ""))
        edge_id = str(candidate.get("edge_id", ""))
        document_ids.append(document_id)
        if document_id:
            allowed_witness_ids.append(document_id)
        if span_id:
            allowed_witness_ids.append(span_id)
        if edge_id:
            allowed_witness_ids.append(edge_id)
        candidate_records.append(
            {
                "order": index,
                "document_id": document_id,
                "text_sha256": text_hash,
                "span_id": span_id or None,
                "edge_id": edge_id or None,
            }
        )
    return {
        "schema_version": G0_SCHEMA,
        "query": query,
        "query_sha256": query_sha256(query),
        "qwen_metadata_sha256": metadata_sha256(metadata),
        "evidence_budget": budget,
        "candidate_document_ids": document_ids,
        "candidates": candidate_records,
        "allowed_witness_ids": sorted(set(allowed_witness_ids)),
    }


def validate_g0_identity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if (
        left.get("schema_version") != G0_SCHEMA
        or right.get("schema_version") != G0_SCHEMA
    ):
        errors.append("unsupported G0 schema")
    if left.get("query_sha256") != right.get("query_sha256"):
        errors.append("G0 query hash mismatch")
    if left.get("qwen_metadata_sha256") != right.get("qwen_metadata_sha256"):
        errors.append("G0 metadata hash mismatch")
    if left.get("evidence_budget") != right.get("evidence_budget"):
        errors.append("G0 evidence budget mismatch")
    if left.get("candidate_document_ids") != right.get("candidate_document_ids"):
        errors.append("G0 candidate order mismatch")
    left_hashes = [item.get("text_sha256") for item in left.get("candidates", [])]
    right_hashes = [item.get("text_sha256") for item in right.get("candidates", [])]
    if left_hashes != right_hashes:
        errors.append("G0 candidate text hash mismatch")
    return sorted(set(errors))


def qwen_retrieval_drift_fraction(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    if left.get("candidate_document_ids") == right.get("candidate_document_ids"):
        return 0.0
    return 1.0
