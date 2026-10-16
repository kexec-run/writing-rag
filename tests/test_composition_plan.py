from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from writing_rag.plan import (
    apply_beat_contract_defaults,
    apply_verifier_result,
    audit_plan_pool_satisfiability,
    build_g0_artifact,
    build_obligation_set,
    compile_oflat,
    compile_shape_beat_units,
    composition_plan_sha256,
    empty_composition_state,
    empty_evidence_state,
    project_for_generation,
    project_for_verification,
    query_sha256,
    reachable_units,
    run_negative_checks,
    run_unit_loop,
    schedule_units,
    validate_composition_plan,
    validate_g0_identity,
    validate_obligation_parity,
    validate_oflat_text,
)

pytestmark = pytest.mark.private_data

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments/active/shape-beat-axis0-v1/fixtures"
)


def _load_fixture(name: str) -> dict:
    path = FIXTURE_ROOT / name / "composition-plan.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _g0_pool(name: str) -> dict:
    path = FIXTURE_ROOT / name / "g0-pool.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _finalize(plan: dict) -> dict:
    finalized = apply_beat_contract_defaults(copy.deepcopy(plan))
    finalized["plan_sha256"] = composition_plan_sha256(finalized)
    return finalized


def test_positive_synthetic_plan_validates() -> None:
    plan = _load_fixture("synthetic-minimal")
    assert validate_composition_plan(plan) == []


def test_positive_plan_hash_is_stable() -> None:
    plan = _load_fixture("synthetic-minimal")
    assert composition_plan_sha256(plan) == plan["plan_sha256"]


def test_positive_obligation_set_builds_sorted_ids() -> None:
    plan = _load_fixture("synthetic-minimal")
    obligation_set = build_obligation_set(plan)
    assert obligation_set["obligation_ids"] == ["OBL-001", "OBL-002"]


def test_positive_oflat_compiles_without_gate_phrases() -> None:
    plan = _load_fixture("synthetic-minimal")
    oflat = compile_oflat(plan)
    assert validate_oflat_text(oflat["scaffold_text"]) == []
    assert "AUTHOR-SUPPLIED NARRATIVE SCAFFOLD" in oflat["scaffold_text"]


def test_positive_shape_beat_sb_units_cover_obligations() -> None:
    plan = _load_fixture("oracle-beat-heavy")
    compiled = compile_shape_beat_units(plan, "SB")
    assert len(compiled["units"]) == 4
    assert compiled["rendered_obligation_ids"] == [
        "OBL-BH-01",
        "OBL-BH-02",
        "OBL-BH-03",
        "OBL-BH-04",
    ]


def test_positive_shape_units_cover_beat_bound_obligations() -> None:
    plan = _load_fixture("oracle-shape-heavy")
    compiled = compile_shape_beat_units(plan, "S")
    assert len(compiled["units"]) == 4
    assert compiled["rendered_obligation_ids"] == [
        "OBL-SH-01",
        "OBL-SH-02",
        "OBL-SH-03",
        "OBL-SH-04",
    ]


def test_positive_obligation_parity_between_compilers() -> None:
    plan = _load_fixture("oracle-shape-heavy")
    oflat = compile_oflat(plan)
    sb = compile_shape_beat_units(plan, "SB")
    assert validate_obligation_parity(plan, oflat["manifest"], sb["manifest"]) == []


def test_positive_schedule_units_orders_shape_then_beat() -> None:
    plan = _load_fixture("oracle-beat-heavy")
    assert schedule_units(plan, "SB") == ["B01-01", "B02-01", "B02-02", "B02-03"]
    assert schedule_units(plan, "S") == ["S01", "S02"]


def test_positive_reachable_units_respects_beat_dependencies() -> None:
    plan = _load_fixture("oracle-beat-heavy")
    completed = {"B01-01", "B02-01", "B02-02"}
    assert reachable_units(plan, "SB", completed) == ["B02-03"]


def test_positive_projection_for_generation_is_json_serializable() -> None:
    plan = _load_fixture("synthetic-minimal")
    unit = compile_shape_beat_units(plan, "SB")["units"][0]
    projection = project_for_generation(
        plan,
        unit,
        "SB",
        empty_composition_state(),
        {"allowed_witness_ids": ["span-fixture-001"]},
    )
    assert projection["unit_kind"] == "beat"
    assert projection["beat_id"] == "B01-01"
    assert projection["allowlist_ids"] == ["span-fixture-001"]


def test_positive_projection_for_verification_closes_shape() -> None:
    plan = _load_fixture("synthetic-minimal")
    verified = {
        "B01-01": {
            "grounded_concepts": ["uncertainty_frame"],
            "established_relations": [],
        },
        "B02-01": {
            "grounded_concepts": ["decision_threshold"],
            "established_relations": [],
        },
    }
    result = project_for_verification(plan, "S02", verified)
    assert result["shape_status"] == "closed"


def test_positive_unit_loop_freezes_units_with_injected_verify() -> None:
    plan = _load_fixture("synthetic-minimal")

    def generate(projection: dict) -> str:
        return f"generated {projection['unit_id']}"

    def verify(unit: dict, text: str) -> dict:
        return {
            "accepted": True,
            "accepted_text": text,
            "grounded_concepts": list(unit.get("grounds", [])),
            "established_relations": list(unit.get("establishes", [])),
            "used_witness_ids": [],
            "verified_claim_ids": [],
            "failures": [],
        }

    result = run_unit_loop(
        plan, "SB", generate, verify, evidence_allowlist=_g0_pool("synthetic-minimal")
    )
    assert result["completed_units"] == ["B01-01", "B02-01"]
    assert result["frozen_units"]["B01-01"]["frozen"] is True


def test_positive_g0_identity_passes_for_matching_pools() -> None:
    left = _g0_pool("synthetic-minimal")
    right = json.loads(
        (FIXTURE_ROOT / "synthetic-minimal" / "g0-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert validate_g0_identity(left, right) == []


def test_positive_satisfiability_audit_passes_fixture_pool() -> None:
    plan = _load_fixture("oracle-shape-heavy")
    obligation_set = build_obligation_set(plan)
    audit = audit_plan_pool_satisfiability(
        plan, obligation_set, _g0_pool("oracle-shape-heavy")
    )
    assert audit["satisfiable"] is True


def test_negative_unsupported_schema_version() -> None:
    plan = _finalize(_load_fixture("synthetic-minimal"))
    plan["schema_version"] = "composition-plan-v9"
    errors = validate_composition_plan(plan)
    assert "unsupported composition plan schema" in errors


def test_negative_plan_hash_mismatch() -> None:
    plan = _load_fixture("synthetic-minimal")
    plan = copy.deepcopy(plan)
    plan["thesis"] = "changed thesis"
    assert "composition plan hash mismatch" in validate_composition_plan(plan)


def test_negative_duplicate_shape_id() -> None:
    plan = _finalize(_load_fixture("synthetic-minimal"))
    plan["shapes"].append(copy.deepcopy(plan["shapes"][0]))
    errors = validate_composition_plan(plan)
    assert any("duplicate shape id" in error for error in errors)


def test_negative_invalid_beat_id_pattern() -> None:
    plan = _finalize(_load_fixture("synthetic-minimal"))
    plan["beats"][0]["id"] = "BAD"
    errors = validate_composition_plan(plan)
    assert any("invalid beat id" in error for error in errors)


def test_negative_beat_dependency_cycle() -> None:
    plan = _finalize(_load_fixture("oracle-beat-heavy"))
    plan["beats"][2]["depends_on"] = ["B02-03"]
    plan["beats"][3]["depends_on"] = ["B02-02"]
    plan["plan_sha256"] = composition_plan_sha256(plan)
    errors = validate_composition_plan(plan)
    assert any("dependency cycle" in error for error in errors)


def test_negative_unknown_obligation_authority() -> None:
    plan = _finalize(_load_fixture("synthetic-minimal"))
    plan["evidence_obligations"][0]["authority"] = "derived_inference"
    plan["plan_sha256"] = composition_plan_sha256(plan)
    errors = validate_composition_plan(plan)
    assert any("unknown obligation authority" in error for error in errors)


def test_negative_source_backed_obligation_missing_witness() -> None:
    plan = _finalize(_load_fixture("synthetic-minimal"))
    plan["evidence_obligations"][0]["witness"] = {}
    plan["plan_sha256"] = composition_plan_sha256(plan)
    errors = validate_composition_plan(plan)
    assert any("missing witness id" in error for error in errors)


def test_negative_oflat_text_rejects_gate_loop_phrase() -> None:
    errors = validate_oflat_text("Continue until the gate closes.")
    assert errors


def test_negative_obligation_parity_detects_missing_ids() -> None:
    plan = _load_fixture("synthetic-minimal")
    oflat = compile_oflat(plan)
    bad_manifest = {"rendered_obligation_ids": ["OBL-001"]}
    assert validate_obligation_parity(plan, oflat["manifest"], bad_manifest)


def test_negative_g0_identity_detects_candidate_order_drift() -> None:
    left = _g0_pool("synthetic-minimal")
    right = copy.deepcopy(left)
    right["candidate_document_ids"] = list(reversed(right["candidate_document_ids"]))
    errors = validate_g0_identity(left, right)
    assert "G0 candidate order mismatch" in errors


def test_negative_satisfiability_rejects_missing_witness() -> None:
    plan = _load_fixture("synthetic-minimal")
    obligation_set = build_obligation_set(plan)
    pool = copy.deepcopy(_g0_pool("synthetic-minimal"))
    pool["allowed_witness_ids"] = []
    audit = audit_plan_pool_satisfiability(plan, obligation_set, pool)
    assert audit["satisfiable"] is False


def test_negative_similarity_only_hook_is_rejected() -> None:
    plan = _load_fixture("synthetic-minimal")
    obligation_set = build_obligation_set(plan)
    audit = audit_plan_pool_satisfiability(
        plan, obligation_set, _g0_pool("synthetic-minimal"), similarity_only=True
    )
    assert "similarity-only satisfiability is not allowed" in audit["errors"]


def test_negative_state_updates_require_verifier_acceptance() -> None:
    composition_state = empty_composition_state()
    evidence_state = empty_evidence_state()
    unit = {"grounds": ["uncertainty_frame"], "establishes": []}
    apply_verifier_result(
        composition_state,
        evidence_state,
        unit,
        {"accepted": False, "grounded_concepts": ["uncertainty_frame"]},
    )
    assert composition_state["grounded_concepts"] == []


def test_negative_declared_concepts_do_not_bypass_verifier() -> None:
    composition_state = empty_composition_state()
    evidence_state = empty_evidence_state()
    apply_verifier_result(
        composition_state,
        evidence_state,
        {"grounds": ["declared_only"], "establishes": []},
        {"accepted": True, "grounded_concepts": []},
    )
    assert composition_state["grounded_concepts"] == []


def test_negative_unit_loop_keeps_obligation_open_after_failed_repair() -> None:
    plan = _load_fixture("synthetic-minimal")

    def generate(projection: dict) -> str:
        return f"generated {projection['unit_id']}"

    def verify(unit: dict, text: str) -> dict:
        return {
            "accepted": False,
            "failures": ["claim not grounded"],
            "repair_text": text + " retry",
        }

    result = run_unit_loop(plan, "SB", generate, verify, max_repairs=1)
    assert "B01-01" in result["failures"]
    assert "B01-01" not in result["completed_units"]


def test_positive_unit_loop_uses_explicit_repair_callback() -> None:
    plan = _load_fixture("synthetic-minimal")
    repair_calls: list[str] = []

    def generate(projection: dict) -> str:
        return f"initial {projection['unit_id']}"

    def repair(projection: dict, text: str, failures: list[str]) -> str:
        repair_calls.append(projection["unit_id"])
        return f"repaired {projection['unit_id']}"

    def verify(unit: dict, text: str) -> dict:
        accepted = text.startswith("repaired")
        return {
            "accepted": accepted,
            "accepted_text": text,
            "grounded_concepts": list(unit.get("grounds", [])) if accepted else [],
            "established_relations": [],
            "failures": [] if accepted else ["initial failure"],
        }

    result = run_unit_loop(plan, "SB", generate, verify, repair=repair, max_repairs=1)
    assert result["repair_count"] == 2
    assert repair_calls == ["B01-01", "B02-01"]


def test_positive_negative_registry_is_executable() -> None:
    checks = run_negative_checks(FIXTURE_ROOT)
    assert len(checks) == 15
    assert all(item["passed"] for item in checks)


def test_negative_query_sha256_changes_with_query_text() -> None:
    assert query_sha256("alpha") != query_sha256("beta")


def test_positive_build_g0_artifact_records_candidate_hashes() -> None:
    artifact = build_g0_artifact(
        "fixture query",
        [{"document_id": "doc-a", "text": "alpha", "span_id": "span-a"}],
        {"retriever_backend": "qwen3"},
        3,
    )
    assert artifact["candidate_document_ids"] == ["doc-a"]
    assert "span-a" in artifact["allowed_witness_ids"]
