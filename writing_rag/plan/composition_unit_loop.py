"""Shared generate-verify-repair-freeze loop for Shape and Beat units."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from writing_rag.plan.composition_plan import beats_for_shape
from writing_rag.plan.composition_projection import (
    project_for_generation,
    project_for_verification,
)
from writing_rag.plan.composition_scheduler import reachable_units, unit_record
from writing_rag.plan.composition_state import (
    apply_verifier_result,
    close_shape,
    empty_composition_state,
    empty_evidence_state,
    shape_entry_satisfied,
    unit_requires_satisfied,
)


def run_unit_loop(
    plan: Mapping[str, Any],
    arm: str,
    generate: Callable[[dict[str, Any]], str],
    verify: Callable[[Mapping[str, Any], str], dict[str, Any]],
    *,
    max_repairs: int = 1,
    evidence_allowlist: Mapping[str, Any] | None = None,
    enable_gates: bool = False,
    repair: Callable[[dict[str, Any], str, list[str]], str] | None = None,
    allow_connective_waiver: bool = False,
) -> dict[str, Any]:
    composition_state = empty_composition_state()
    evidence_state = empty_evidence_state()
    allowlist = dict(evidence_allowlist or {})
    completed: set[str] = set()
    frozen_units: dict[str, dict[str, Any]] = {}
    verified_artifacts: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[str]] = {}
    repair_count = 0
    unfrozen_attempts: list[str] = []

    shapes = sorted(plan.get("shapes", []), key=lambda item: int(item.get("order", 0)))
    for shape in shapes:
        shape_id = str(shape.get("id", ""))
        entry_errors = shape_entry_satisfied(plan, shape_id, composition_state)
        if enable_gates and entry_errors:
            failures.setdefault(shape_id, []).extend(entry_errors)
            continue

        composition_state["active_shape_id"] = shape_id
        shape_unit_ids = (
            [shape_id]
            if arm == "S"
            else [
                str(beat["id"])
                for beat in sorted(
                    beats_for_shape(plan, shape_id),
                    key=lambda item: (
                        int(item.get("order", 0)),
                        str(item.get("id", "")),
                    ),
                )
            ]
        )

        for unit_id in shape_unit_ids:
            if unit_id in completed:
                unfrozen_attempts.append(unit_id)
                continue
            if unit_id not in reachable_units(plan, arm, completed):
                failures.setdefault(unit_id, []).append("unit unreachable")
                continue

            unit = unit_record(plan, arm, unit_id)
            if enable_gates:
                req_errors = unit_requires_satisfied(unit, composition_state)
                if req_errors:
                    failures.setdefault(unit_id, []).extend(req_errors)
                    continue

            projection = project_for_generation(
                plan, unit, arm, composition_state, allowlist
            )
            text = generate(projection)
            verify_result = verify(unit, text)
            unit_failures = list(verify_result.get("failures", []))

            repairs = 0
            while unit_failures and repairs < max_repairs:
                if repair is None:
                    break
                repair_count += 1
                repairs += 1
                repaired_text = repair(projection, text, unit_failures)
                repair_result = verify(unit, repaired_text)
                if repair_result.get("accepted"):
                    verify_result = repair_result
                    unit_failures = list(repair_result.get("failures", []))
                else:
                    unit_failures = list(repair_result.get("failures", []))

            if unit_failures:
                failures[unit_id] = unit_failures
                verified_artifacts[unit_id] = {
                    "accepted": False,
                    "grounded_concepts": [],
                    "established_relations": [],
                    "qualifiers_preserved": [],
                }
                continue

            accepted_text = str(verify_result.get("accepted_text", text))
            frozen_units[unit_id] = {
                "unit_id": unit_id,
                "text": accepted_text,
                "frozen": True,
            }
            verified_artifacts[unit_id] = {
                "accepted": True,
                "accepted_text": accepted_text,
                "grounded_concepts": list(verify_result.get("grounded_concepts", [])),
                "established_relations": list(
                    verify_result.get("established_relations", [])
                ),
                "qualifiers_preserved": list(
                    verify_result.get("qualifiers_preserved", [])
                ),
                "used_witness_ids": list(verify_result.get("used_witness_ids", [])),
                "verified_claim_ids": list(verify_result.get("verified_claim_ids", [])),
            }
            apply_verifier_result(
                composition_state, evidence_state, unit, verify_result
            )
            completed.add(unit_id)

        shape_verification = project_for_verification(
            plan, shape_id, verified_artifacts, allow_connective_waiver
        )
        if shape_verification.get("shape_status") in {"closed", "closed_amended"}:
            close_shape(composition_state, shape_id)
        else:
            failures.setdefault(shape_id, []).extend(
                shape_verification.get("missing_exit_obligations", [])
            )

    return {
        "arm": arm,
        "completed_units": sorted(completed),
        "frozen_units": frozen_units,
        "verified_artifacts": verified_artifacts,
        "composition_state": composition_state,
        "evidence_state": evidence_state,
        "failures": failures,
        "repair_count": repair_count,
        "unfrozen_attempts": unfrozen_attempts,
        "shape_verifications": {
            str(shape["id"]): project_for_verification(
                plan, str(shape["id"]), verified_artifacts, allow_connective_waiver
            )
            for shape in shapes
        },
    }
