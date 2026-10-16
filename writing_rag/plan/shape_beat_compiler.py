"""Compile composition plans into Shape or Beat execution units."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from writing_rag.plan.composition_plan import beats_for_shape
from writing_rag.plan.obligation_set import build_obligation_set


def _obligations_for_unit(
    obligation_set: Mapping[str, Any],
    shape_id: str,
    beat_id: str | None,
) -> list[str]:
    rendered: list[str] = []
    for obligation in obligation_set.get("obligations", []):
        binds = obligation.get("binds", {})
        if not isinstance(binds, Mapping):
            continue
        bound_shape = str(binds.get("shape_id", ""))
        bound_beat = str(binds.get("beat_id", ""))
        obligation_id = str(obligation.get("obligation_id", ""))
        if beat_id and bound_beat == beat_id or not beat_id and bound_shape == shape_id:
            rendered.append(obligation_id)
    return sorted(set(rendered))


def compile_shape_beat_units(
    plan: Mapping[str, Any],
    arm: Literal["S", "SB"],
) -> dict[str, Any]:
    obligation_set = build_obligation_set(plan)
    shapes = sorted(plan.get("shapes", []), key=lambda item: int(item.get("order", 0)))
    units: list[dict[str, Any]] = []
    rendered_ids: set[str] = set()

    if arm == "S":
        for shape in shapes:
            shape_id = str(shape.get("id", ""))
            unit_obligations = _obligations_for_unit(obligation_set, shape_id, None)
            rendered_ids.update(unit_obligations)
            units.append(
                {
                    "unit_id": shape_id,
                    "unit_kind": "shape",
                    "shape_id": shape_id,
                    "order": int(shape.get("order", 0)),
                    "purpose": str(shape.get("purpose", "")),
                    "entry": dict(shape.get("entry", {})),
                    "exit": dict(shape.get("exit", {})),
                    "rendered_obligation_ids": unit_obligations,
                }
            )
    else:
        for shape in shapes:
            shape_id = str(shape.get("id", ""))
            shape_beats = sorted(
                beats_for_shape(plan, shape_id),
                key=lambda item: (int(item.get("order", 0)), str(item.get("id", ""))),
            )
            for beat in shape_beats:
                beat_id = str(beat.get("id", ""))
                unit_obligations = _obligations_for_unit(
                    obligation_set, shape_id, beat_id
                )
                rendered_ids.update(unit_obligations)
                units.append(
                    {
                        "unit_id": beat_id,
                        "unit_kind": "beat",
                        "shape_id": shape_id,
                        "order": int(beat.get("order", 0)),
                        "purpose": str(beat.get("purpose", "")),
                        "depends_on": list(beat.get("depends_on", [])),
                        "requires": list(beat.get("requires", [])),
                        "grounds": list(beat.get("grounds", [])),
                        "establishes": list(beat.get("establishes", [])),
                        "rendered_obligation_ids": unit_obligations,
                    }
                )

    all_ids = set(obligation_set.get("obligation_ids", []))
    if rendered_ids != all_ids:
        missing = sorted(all_ids - rendered_ids)
        extra = sorted(rendered_ids - all_ids)
        raise ValueError(
            f"Shape/Beat compiler obligation coverage mismatch missing={missing} extra={extra}"
        )

    manifest = {
        "compiler": "shape_beat",
        "arm": arm,
        "rendered_obligation_ids": sorted(rendered_ids),
        "unit_count": len(units),
    }
    return {
        "units": units,
        "rendered_obligation_ids": sorted(rendered_ids),
        "manifest": manifest,
    }
