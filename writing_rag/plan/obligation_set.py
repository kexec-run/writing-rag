"""Canonical obligation-set compilation and compiler parity checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from writing_rag.plan.composition_plan import SCHEMA_VERSION as PLAN_SCHEMA

OBLIGATION_SET_SCHEMA = "obligation-set-v0"


def build_obligation_set(plan: Mapping[str, Any]) -> dict[str, Any]:
    obligations = [dict(item) for item in plan.get("evidence_obligations", [])]
    obligation_ids = sorted(
        {
            str(item.get("obligation_id", ""))
            for item in obligations
            if item.get("obligation_id")
        }
    )
    payload = {
        "schema_version": OBLIGATION_SET_SCHEMA,
        "plan_id": str(plan.get("plan_id", "")),
        "composition_plan_schema": str(plan.get("schema_version", "")),
        "obligation_ids": obligation_ids,
        "obligations": sorted(
            obligations, key=lambda item: str(item.get("obligation_id", ""))
        ),
    }
    payload["obligation_set_sha256"] = obligation_set_sha256(payload)
    return payload


def obligation_set_sha256(obligation_set: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in obligation_set.items()
        if key != "obligation_set_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def validate_obligation_parity(
    plan: Mapping[str, Any],
    oflat_manifest: Mapping[str, Any],
    stateful_manifest: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != PLAN_SCHEMA:
        errors.append("composition plan schema mismatch for parity check")
    canonical = build_obligation_set(plan)
    canonical_ids = set(canonical.get("obligation_ids", []))
    oflat_ids = set(oflat_manifest.get("rendered_obligation_ids", []))
    stateful_ids = set(stateful_manifest.get("rendered_obligation_ids", []))
    if canonical_ids != oflat_ids:
        errors.append("O-flat compiler obligation ID mismatch")
    if canonical_ids != stateful_ids:
        errors.append("Shape/Beat compiler obligation ID mismatch")
    if oflat_ids != stateful_ids:
        errors.append("compiler manifests disagree on obligation IDs")
    return sorted(set(errors))
