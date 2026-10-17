"""Executable positive/negative checks for the Shape-Beat Axis 0 gate."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from writing_rag.plan.composition_plan import (
    apply_beat_contract_defaults,
    composition_plan_sha256,
    validate_composition_plan,
)
from writing_rag.plan.composition_state import (
    apply_verifier_result,
    empty_composition_state,
    empty_evidence_state,
)
from writing_rag.plan.composition_unit_loop import run_unit_loop
from writing_rag.plan.g0_freeze import query_sha256, validate_g0_identity
from writing_rag.plan.obligation_set import (
    build_obligation_set,
    validate_obligation_parity,
)
from writing_rag.plan.oflat_compiler import compile_oflat, validate_oflat_text
from writing_rag.plan.satisfiability import audit_plan_pool_satisfiability

EXPECTED_OBLIGATION_IDS = {
    "synthetic-minimal": ["OBL-001", "OBL-002"],
    "multi-beat": ["OBL-001", "OBL-002", "OBL-003", "OBL-004"],
}


def validate_public_obligation_contract(
    plan: dict[str, Any], expected_ids: list[str] | None = None
) -> list[str]:
    """Validate plan obligation bindings and, when supplied, its fixed ID set."""
    errors: list[str] = []
    shape_ids = {
        str(shape.get("id", ""))
        for shape in plan.get("shapes", [])
        if isinstance(shape, dict)
    }
    beats = {
        str(beat.get("id", "")): str(beat.get("shape_id", ""))
        for beat in plan.get("beats", [])
        if isinstance(beat, dict)
    }
    obligations = plan.get("evidence_obligations", [])
    if not isinstance(obligations, list) or not obligations:
        return ["composition plan must contain evidence obligations"]
    actual_ids: list[str] = []
    for obligation in obligations:
        if not isinstance(obligation, dict):
            errors.append("evidence obligation must be an object")
            continue
        obligation_id = str(obligation.get("obligation_id", ""))
        if obligation_id:
            actual_ids.append(obligation_id)
        binds = obligation.get("binds")
        if not isinstance(binds, dict):
            errors.append(f"obligation has invalid bindings: {obligation_id}")
            continue
        shape_id = str(binds.get("shape_id", ""))
        beat_id = str(binds.get("beat_id", ""))
        if shape_id not in shape_ids:
            errors.append(f"obligation binds to missing shape: {obligation_id}")
        if beat_id not in beats:
            errors.append(f"obligation binds to missing beat: {obligation_id}")
        elif beats[beat_id] != shape_id:
            errors.append(f"obligation shape/beat binding mismatch: {obligation_id}")
    if expected_ids is not None and actual_ids != expected_ids:
        errors.append("obligation IDs differ from the fixed case contract")
    return sorted(set(errors))


def _load_fixture(fixture_root: Path, name: str, filename: str) -> dict[str, Any]:
    value = json.loads((fixture_root / name / filename).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"fixture must be an object: {name}/{filename}")
    return value


def _finalize(plan: dict[str, Any]) -> dict[str, Any]:
    value = apply_beat_contract_defaults(copy.deepcopy(plan))
    value["plan_sha256"] = composition_plan_sha256(value)
    return value


def _check(name: str, mutation: str, action: Callable[[], bool]) -> dict[str, Any]:
    try:
        passed = bool(action())
        error = None
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - malformed fixture diagnostics
        passed = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "check_id": name,
        "mutation": mutation,
        "expected": "reject",
        "passed": passed,
        "error": error,
    }


def run_negative_checks(
    fixture_root: Path, fixture_name: str = "synthetic-minimal"
) -> list[dict[str, Any]]:
    """Run every registered Axis 0 negative mutation and return its evidence."""

    def minimal() -> dict[str, Any]:
        return _load_fixture(fixture_root, fixture_name, "composition-plan.json")

    def beat() -> dict[str, Any]:
        # Build cycle mutations from the selected public synthetic case.
        return minimal()

    def minimal_pool() -> dict[str, Any]:
        return _load_fixture(fixture_root, fixture_name, "g0-pool.json")

    def unsupported_schema() -> bool:
        plan = _finalize(minimal())
        plan["schema_version"] = "composition-plan-v9"
        return "unsupported composition plan schema" in validate_composition_plan(plan)

    def plan_hash() -> bool:
        plan = minimal()
        plan["thesis"] = "changed thesis"
        return "composition plan hash mismatch" in validate_composition_plan(plan)

    def duplicate_shape() -> bool:
        plan = _finalize(minimal())
        plan["shapes"].append(copy.deepcopy(plan["shapes"][0]))
        return any(
            "duplicate shape id" in item for item in validate_composition_plan(plan)
        )

    def invalid_beat() -> bool:
        plan = _finalize(minimal())
        plan["beats"][0]["id"] = "BAD"
        return any(
            "invalid beat id" in item for item in validate_composition_plan(plan)
        )

    def dependency_cycle() -> bool:
        plan = _finalize(beat())
        first_id = str(plan["beats"][0]["id"])
        second_id = str(plan["beats"][1]["id"])
        plan["beats"][0]["depends_on"] = [second_id]
        plan["beats"][1]["depends_on"] = [first_id]
        plan["plan_sha256"] = composition_plan_sha256(plan)
        return any(
            "dependency cycle" in item for item in validate_composition_plan(plan)
        )

    def unknown_authority() -> bool:
        plan = _finalize(minimal())
        plan["evidence_obligations"][0]["authority"] = "derived_inference"
        plan["plan_sha256"] = composition_plan_sha256(plan)
        return any(
            "unknown obligation authority" in item
            for item in validate_composition_plan(plan)
        )

    def missing_witness() -> bool:
        plan = _finalize(minimal())
        plan["evidence_obligations"][0]["witness"] = {}
        plan["plan_sha256"] = composition_plan_sha256(plan)
        return any(
            "missing witness id" in item for item in validate_composition_plan(plan)
        )

    def obligation_structure() -> bool:
        original = _finalize(minimal())
        expected_ids = EXPECTED_OBLIGATION_IDS.get(fixture_name)
        if expected_ids is None:
            return False

        missing = copy.deepcopy(original)
        missing["evidence_obligations"].pop()
        missing["plan_sha256"] = composition_plan_sha256(missing)

        duplicate = copy.deepcopy(original)
        duplicate["evidence_obligations"].append(
            copy.deepcopy(duplicate["evidence_obligations"][0])
        )
        duplicate["plan_sha256"] = composition_plan_sha256(duplicate)

        invalid_binding = copy.deepcopy(original)
        invalid_binding["evidence_obligations"][0]["binds"]["beat_id"] = "B99-99"
        invalid_binding["plan_sha256"] = composition_plan_sha256(invalid_binding)

        return (
            bool(validate_public_obligation_contract(missing, expected_ids))
            and bool(validate_composition_plan(duplicate))
            and bool(validate_public_obligation_contract(invalid_binding, expected_ids))
        )

    def forbidden_oflat_instruction() -> bool:
        return bool(validate_oflat_text("Continue until the gate closes."))

    def missing_parity_id() -> bool:
        plan = minimal()
        oflat = compile_oflat(plan)
        bad_manifest = {"rendered_obligation_ids": ["OBL-001"]}
        return bool(validate_obligation_parity(plan, oflat["manifest"], bad_manifest))

    def g0_order_drift() -> bool:
        left = minimal_pool()
        right = copy.deepcopy(left)
        right["candidate_document_ids"] = list(
            reversed(right["candidate_document_ids"])
        )
        return "G0 candidate order mismatch" in validate_g0_identity(left, right)

    def missing_pool_witness() -> bool:
        plan = minimal()
        obligations = build_obligation_set(plan)
        pool = minimal_pool()
        pool["allowed_witness_ids"] = []
        return (
            audit_plan_pool_satisfiability(plan, obligations, pool)["satisfiable"]
            is False
        )

    def similarity_only() -> bool:
        plan = minimal()
        obligations = build_obligation_set(plan)
        audit = audit_plan_pool_satisfiability(
            plan, obligations, minimal_pool(), similarity_only=True
        )
        return "similarity-only satisfiability is not allowed" in audit["errors"]

    def rejected_state_update() -> bool:
        composition_state = empty_composition_state()
        evidence_state = empty_evidence_state()
        apply_verifier_result(
            composition_state,
            evidence_state,
            {"grounds": ["uncertainty_frame"], "establishes": []},
            {"accepted": False, "grounded_concepts": ["uncertainty_frame"]},
        )
        return composition_state["grounded_concepts"] == []

    def failed_repair_stays_open() -> bool:
        plan = minimal()

        def generate(projection: dict[str, Any], repair: bool = False) -> str:
            return (
                f"generated {projection['unit_id']} {'repair' if repair else 'initial'}"
            )

        def verify(unit: dict[str, Any], text: str) -> dict[str, Any]:
            return {"accepted": False, "failures": ["claim not grounded"]}

        result = run_unit_loop(
            plan,
            "SB",
            lambda projection: generate(projection),
            verify,
            repair=lambda projection, text, failures: generate(projection, True),
            max_repairs=1,
        )
        return (
            "B01-01" in result["failures"] and "B01-01" not in result["completed_units"]
        )

    checks = [
        ("unsupported_schema", "change composition schema version", unsupported_schema),
        ("plan_hash", "change plan content without recomputing hash", plan_hash),
        ("duplicate_shape", "duplicate a Shape ID", duplicate_shape),
        ("invalid_beat", "replace a Beat ID with an invalid value", invalid_beat),
        ("dependency_cycle", "introduce a Beat dependency cycle", dependency_cycle),
        ("unknown_authority", "introduce a forbidden authority", unknown_authority),
        (
            "obligation_structure",
            "remove or duplicate obligations, corrupt a Shape/Beat binding, or remove a witness",
            lambda: missing_witness() and obligation_structure(),
        ),
        (
            "oflat_gate_instruction",
            "insert a state-loop instruction into O-flat",
            forbidden_oflat_instruction,
        ),
        (
            "obligation_parity",
            "remove an obligation from a compiler manifest",
            missing_parity_id,
        ),
        ("g0_order", "reverse frozen G0 candidate order", g0_order_drift),
        ("pool_witness", "remove all allowed witness IDs", missing_pool_witness),
        (
            "similarity_only",
            "authorize satisfiability from similarity only",
            similarity_only,
        ),
        (
            "rejected_state_update",
            "return grounded concepts with accepted=false",
            rejected_state_update,
        ),
        ("failed_repair", "fail the one allowed repair", failed_repair_stays_open),
        (
            "query_hash",
            "change the frozen query text",
            lambda: query_sha256("alpha") != query_sha256("beta"),
        ),
    ]
    return [_check(name, mutation, action) for name, mutation, action in checks]
