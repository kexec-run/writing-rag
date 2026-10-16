"""Composition and evidence state stores for unit execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def empty_composition_state() -> dict[str, Any]:
    return {
        "closed_shapes": [],
        "active_shape_id": "",
        "grounded_concepts": [],
        "established_relations": [],
        "open_obligations": [],
        "accepted_tail": "",
    }


def empty_evidence_state() -> dict[str, Any]:
    return {
        "verified_claim_ids": [],
        "used_witness_ids": [],
        "open_evidence_obligations": [],
    }


def shape_entry_satisfied(
    plan: Mapping[str, Any], shape_id: str, composition_state: Mapping[str, Any]
) -> list[str]:
    from writing_rag.plan.composition_plan import shape_by_id

    shape = shape_by_id(plan).get(shape_id, {})
    entry = shape.get("entry", {})
    if not isinstance(entry, Mapping):
        return []
    grounded = set(composition_state.get("grounded_concepts", []))
    errors: list[str] = []
    for concept in entry.get("required", []):
        if concept not in grounded:
            errors.append(f"shape entry missing concept: {shape_id} requires {concept}")
    return errors


def unit_requires_satisfied(
    unit: Mapping[str, Any], composition_state: Mapping[str, Any]
) -> list[str]:
    grounded = set(composition_state.get("grounded_concepts", []))
    errors: list[str] = []
    for concept in unit.get("requires", []):
        if concept not in grounded:
            errors.append(f"unit requires ungrounded concept: {concept}")
    return errors


def apply_verifier_result(
    composition_state: dict[str, Any],
    evidence_state: dict[str, Any],
    unit: Mapping[str, Any],
    verifier_payload: Mapping[str, Any],
) -> None:
    if not verifier_payload.get("accepted"):
        return
    composition_state.setdefault("grounded_concepts", [])
    composition_state.setdefault("established_relations", [])
    composition_state["grounded_concepts"] = sorted(
        set(composition_state["grounded_concepts"])
        | set(verifier_payload.get("grounded_concepts", []))
    )
    composition_state["established_relations"] = sorted(
        set(composition_state["established_relations"])
        | set(verifier_payload.get("established_relations", []))
    )
    if verifier_payload.get("accepted_text"):
        composition_state["accepted_tail"] = str(
            verifier_payload.get("accepted_text", "")
        )
    for witness_id in verifier_payload.get("used_witness_ids", []):
        if witness_id not in evidence_state.setdefault("used_witness_ids", []):
            evidence_state["used_witness_ids"].append(witness_id)
    for claim_id in verifier_payload.get("verified_claim_ids", []):
        if claim_id not in evidence_state.setdefault("verified_claim_ids", []):
            evidence_state["verified_claim_ids"].append(claim_id)


def close_shape(composition_state: dict[str, Any], shape_id: str) -> None:
    closed = composition_state.setdefault("closed_shapes", [])
    if shape_id not in closed:
        closed.append(shape_id)
