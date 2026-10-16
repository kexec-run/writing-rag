"""Deterministic scheduling for Shape and Beat execution units."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from writing_rag.plan.composition_plan import beats_for_shape


def schedule_units(plan: Mapping[str, Any], arm: str) -> list[str]:
    shapes = sorted(plan.get("shapes", []), key=lambda item: int(item.get("order", 0)))
    ordered: list[str] = []
    if arm == "S":
        for shape in shapes:
            ordered.append(str(shape.get("id", "")))
        return ordered

    for shape in shapes:
        shape_id = str(shape.get("id", ""))
        beats = sorted(
            beats_for_shape(plan, shape_id),
            key=lambda item: (int(item.get("order", 0)), str(item.get("id", ""))),
        )
        for beat in beats:
            ordered.append(str(beat.get("id", "")))
    return ordered


def reachable_units(
    plan: Mapping[str, Any],
    arm: str,
    completed: set[str],
) -> list[str]:
    scheduled = schedule_units(plan, arm)
    if arm == "S":
        return [unit_id for unit_id in scheduled if unit_id not in completed]

    beat_by_id = {str(beat["id"]): beat for beat in plan.get("beats", [])}
    reachable: list[str] = []
    for unit_id in scheduled:
        if unit_id in completed:
            continue
        beat = beat_by_id.get(unit_id, {})
        dependencies = [str(value) for value in beat.get("depends_on", [])]
        if all(dep in completed for dep in dependencies):
            reachable.append(unit_id)
    return reachable


def unit_record(plan: Mapping[str, Any], arm: str, unit_id: str) -> dict[str, Any]:
    from writing_rag.plan.shape_beat_compiler import compile_shape_beat_units

    units = compile_shape_beat_units(plan, "S" if arm == "S" else "SB")["units"]
    for unit in units:
        if str(unit.get("unit_id")) == unit_id:
            return dict(unit)
    raise KeyError(f"unknown unit id: {unit_id}")
