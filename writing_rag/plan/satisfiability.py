"""Exact-ID satisfiability audit for composition plans against frozen G0 pools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from writing_rag.plan.article_plan import AUTHORITIES


def _witness_ids_from_obligation(obligation: Mapping[str, Any]) -> list[str]:
    witness = obligation.get("witness", {})
    if not isinstance(witness, Mapping):
        return []
    ids: list[str] = []
    for key in ("document_id", "span_id", "edge_id", "author_assertion_id"):
        value = witness.get(key)
        if value:
            ids.append(str(value))
    return ids


def audit_plan_pool_satisfiability(
    plan: Mapping[str, Any],
    obligation_set: Mapping[str, Any],
    g0_pool: Mapping[str, Any],
    *,
    similarity_only: bool = False,
) -> dict[str, Any]:
    if similarity_only:
        return {
            "satisfiable": False,
            "errors": ["similarity-only satisfiability is not allowed"],
            "unsatisfied_obligation_ids": sorted(
                obligation_set.get("obligation_ids", [])
            ),
        }

    allowed = set(g0_pool.get("allowed_witness_ids", []))
    assertion_ids = {
        str(item.get("author_assertion_id", ""))
        for item in plan.get("author_assertions", [])
        if item.get("author_assertion_id")
    }
    errors: list[str] = []
    unsatisfied: list[str] = []

    for obligation in obligation_set.get("obligations", []):
        obligation_id = str(obligation.get("obligation_id", ""))
        authority = str(obligation.get("authority", ""))
        if authority not in AUTHORITIES:
            errors.append(f"unknown authority on obligation: {obligation_id}")
            unsatisfied.append(obligation_id)
            continue
        witness_ids = _witness_ids_from_obligation(obligation)
        if authority == "source_backed":
            if not witness_ids or not all(wid in allowed for wid in witness_ids):
                errors.append(
                    f"source_backed obligation not satisfiable: {obligation_id}"
                )
                unsatisfied.append(obligation_id)
        elif authority == "author_asserted":
            assertion_id = witness_ids[0] if witness_ids else ""
            supplied = assertion_ids or allowed
            if assertion_id not in supplied:
                errors.append(
                    f"author_asserted obligation not satisfiable: {obligation_id}"
                )
                unsatisfied.append(obligation_id)

    return {
        "satisfiable": not unsatisfied and not errors,
        "errors": sorted(set(errors)),
        "unsatisfied_obligation_ids": sorted(set(unsatisfied)),
        "allowed_witness_count": len(allowed),
    }
