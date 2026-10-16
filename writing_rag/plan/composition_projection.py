"""Deterministic forward and reverse projections for composition units."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from writing_rag.plan.composition_plan import beats_for_shape, shape_by_id


def project_for_generation(
    plan: Mapping[str, Any],
    unit: Mapping[str, Any],
    arm: str,
    composition_state: Mapping[str, Any],
    evidence_allowlist: Mapping[str, Any],
) -> dict[str, Any]:
    unit_kind = str(unit.get("unit_kind", ""))
    shape_id = str(unit.get("shape_id", ""))
    shapes = shape_by_id(plan)
    shape = shapes.get(shape_id, {})
    projection: dict[str, Any] = {
        "thesis": str(plan.get("thesis", "")),
        "arm": arm,
        "unit_kind": unit_kind,
        "unit_id": str(unit.get("unit_id", "")),
        "shape_id": shape_id,
        "closed_shapes": list(composition_state.get("closed_shapes", [])),
        "current_shape_entry": dict(shape.get("entry", {})),
        "current_shape_exit": dict(shape.get("exit", {})),
        "unit_purpose": str(unit.get("purpose", "")),
        "requires": list(unit.get("requires", [])),
        "forbidden": list(shape.get("exit", {}).get("forbidden", [])),
        "allowlist_ids": list(evidence_allowlist.get("allowed_witness_ids", [])),
        "grounded_concepts": list(composition_state.get("grounded_concepts", [])),
        "accepted_tail": str(composition_state.get("accepted_tail", "")),
    }
    if unit_kind == "beat":
        projection["beat_id"] = str(unit.get("unit_id", ""))
        projection["depends_on"] = list(unit.get("depends_on", []))
    else:
        projection["beat_id"] = None
    return projection


def project_for_verification(
    plan: Mapping[str, Any],
    shape_id: str,
    verified_units: Mapping[str, Mapping[str, Any]],
    allow_connective_waiver: bool = False,
) -> dict[str, Any]:
    shapes = shape_by_id(plan)
    shape = shapes.get(shape_id, {})
    if not shape:
        return {
            "shape_id": shape_id,
            "shape_status": "invalid",
            "missing_exit_obligations": ["unknown shape"],
            "observed_concepts": [],
            "observed_relations": [],
        }

    observed_concepts: set[str] = set()
    observed_relations: set[str] = set()
    qualifiers_preserved: list[str] = []

    shape_beats = beats_for_shape(plan, shape_id)
    beat_ids = {str(beat["id"]) for beat in shape_beats}
    if not beat_ids:
        unit_id = shape_id
        artifact = verified_units.get(unit_id, {})
    else:
        for beat_id in sorted(beat_ids):
            artifact = verified_units.get(beat_id, {})
            observed_concepts.update(artifact.get("grounded_concepts", []))
            observed_relations.update(artifact.get("established_relations", []))
            qualifiers_preserved.extend(artifact.get("qualifiers_preserved", []))
        artifact = {}

    if not beat_ids:
        observed_concepts.update(artifact.get("grounded_concepts", []))
        observed_relations.update(artifact.get("established_relations", []))
        qualifiers_preserved.extend(artifact.get("qualifiers_preserved", []))

    exit_block = shape.get("exit", {})
    required = (
        list(exit_block.get("required", [])) if isinstance(exit_block, Mapping) else []
    )
    forbidden = (
        list(exit_block.get("forbidden", [])) if isinstance(exit_block, Mapping) else []
    )

    missing_exit: list[str] = []
    for concept in required:
        if concept not in observed_concepts:
            missing_exit.append(f"missing required concept: {concept}")
    for relation in required:
        if relation not in observed_relations and relation not in observed_concepts:
            missing_exit.append(f"missing required relation: {relation}")
    for concept in forbidden:
        if concept in observed_concepts:
            missing_exit.append(f"forbidden concept present: {concept}")

    waived_exit: list[str] = []
    if allow_connective_waiver and missing_exit:
        connective_grounds: set[str] = set()
        for beat in shape_beats:
            artifact = verified_units.get(str(beat["id"]), {})
            if artifact.get("accepted") and not artifact.get("grounded_concepts"):
                connective_grounds.update(
                    str(item) for item in beat.get("grounds", [])
                )
                connective_grounds.update(
                    str(item) for item in beat.get("establishes", [])
                )
        if connective_grounds:
            remaining: list[str] = []
            for item in missing_exit:
                _, _, target = item.partition(": ")
                if target and target in connective_grounds:
                    waived_exit.append(item)
                else:
                    remaining.append(item)
            missing_exit = remaining

    if not missing_exit and waived_exit:
        shape_status = "closed_amended"
    elif not missing_exit:
        shape_status = "closed"
    else:
        shape_status = "open"
    return {
        "shape_id": shape_id,
        "shape_status": shape_status,
        "observed_concepts": sorted(observed_concepts),
        "observed_relations": sorted(observed_relations),
        "qualifiers_preserved": sorted(set(qualifiers_preserved)),
        "missing_exit_obligations": sorted(missing_exit),
        "waived_exit_obligations": sorted(waived_exit),
        "amendment": (
            "connective_only_beats_ride_uncredited" if waived_exit else None
        ),
    }
