"""Run the public synthetic Shape–Beat validation cases without models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from writing_rag.plan import (
    audit_plan_pool_satisfiability,
    build_obligation_set,
    compile_oflat,
    compile_shape_beat_units,
    run_negative_checks,
    schedule_units,
    validate_composition_plan,
    validate_g0_identity,
    validate_obligation_parity,
    validate_oflat_text,
)
from writing_rag.plan.axis0_validation import (
    EXPECTED_OBLIGATION_IDS,
    validate_public_obligation_contract,
)
from writing_rag.plan.g0_freeze import query_sha256

CASE_DIRECTORIES = {
    "minimal": REPOSITORY_ROOT
    / "experiments/active/shape-beat-axis0-v1/fixtures/synthetic-minimal",
    "multi-beat": REPOSITORY_ROOT
    / "tests/fixtures/shape_beat_validation/multi-beat",
}
EXPECTED_SCHEDULES = {
    "minimal": {
        "S": ["S01", "S02"],
        "SB": ["B01-01", "B02-01"],
    },
    "multi-beat": {
        "S": ["S01", "S02"],
        "SB": ["B01-01", "B02-01", "B02-02", "B02-03"],
    },
}
EXPECTED_UNITS = {"minimal": {"S": 2, "SB": 2}, "multi-beat": {"S": 2, "SB": 4}}
EXPECTED_CASE_OBLIGATION_IDS = {
    "minimal": EXPECTED_OBLIGATION_IDS["synthetic-minimal"],
    "multi-beat": EXPECTED_OBLIGATION_IDS["multi-beat"],
}
INPUT_NAMES = ("composition-plan.json", "g0-pool.json", "g0-baseline.json")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_INPUT_SHA256 = {
    "minimal": {
        "composition-plan.json": "93984f343d631fa851f2a15b1748cb77e101a4ef32fa3787618bd712c014dec8",
        "g0-pool.json": "a00ddb820c21638502380418ab48e1aedf95db3954e4690601413d9ad2d4aa7c",
        "g0-baseline.json": "a00ddb820c21638502380418ab48e1aedf95db3954e4690601413d9ad2d4aa7c",
    },
    "multi-beat": {
        "composition-plan.json": "24b286da511f89135fc0b0a9adf1078cb2a67b20fa364543155ab0c7eff073f1",
        "g0-pool.json": "0ade3508be937d663eeb6d3581e43de41a69520cd68a8abf9481965ceb9c7daa",
        "g0-baseline.json": "0ade3508be937d663eeb6d3581e43de41a69520cd68a8abf9481965ceb9c7daa",
    },
}


def _check(check_id: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed), "detail": detail}


def _read_case_json(case_directory: Path, filename: str) -> dict[str, Any]:
    root = REPOSITORY_ROOT.resolve()
    directory = case_directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("fixture directory resolves outside the public repository")
    current = root
    for part in directory.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinked fixture path is prohibited: {filename}")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required fixture is missing or not a regular file: {filename}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"fixture root must be an object: {filename}")
    return value


def _validate_pool(pool: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if pool.get("schema_version") != "g0-freeze-v0":
        errors.append("unsupported G0 pool schema")
    query = pool.get("query")
    if not isinstance(query, str) or not query.strip():
        errors.append("G0 query is missing")
    elif pool.get("query_sha256") != query_sha256(query):
        errors.append("G0 query hash does not match query text")
    if not isinstance(pool.get("qwen_metadata_sha256"), str) or not HEX_SHA256.fullmatch(
        pool.get("qwen_metadata_sha256", "")
    ):
        errors.append("G0 metadata hash is invalid")
    budget = pool.get("evidence_budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        errors.append("G0 evidence budget must be a positive integer")

    candidates = pool.get("candidates")
    candidate_ids = pool.get("candidate_document_ids")
    if not isinstance(candidates, list) or not candidates:
        errors.append("G0 candidates are missing")
        candidates = []
    if not isinstance(candidate_ids, list) or any(
        not isinstance(item, str) or not item for item in candidate_ids
    ):
        errors.append("G0 candidate document IDs are invalid")
        candidate_ids = []
    rows = [item for item in candidates if isinstance(item, dict)]
    if len(rows) != len(candidates):
        errors.append("G0 candidate record is not an object")
    row_ids = [item.get("document_id") for item in rows]
    if row_ids != candidate_ids or len(row_ids) != len(set(row_ids)):
        errors.append("G0 candidate IDs do not match candidate records in order")
    if [item.get("order") for item in rows] != list(range(len(rows))):
        errors.append("G0 candidate order fields are not contiguous")

    witness_ids: set[str] = set()
    for item in rows:
        document_id = item.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            errors.append("G0 candidate document ID is invalid")
        else:
            witness_ids.add(document_id)
        digest = item.get("text_sha256")
        if not isinstance(digest, str) or not HEX_SHA256.fullmatch(digest):
            errors.append(f"G0 candidate text hash is invalid: {document_id}")
        for key in ("span_id", "edge_id"):
            witness = item.get(key)
            if witness is not None:
                if not isinstance(witness, str) or not witness:
                    errors.append(f"G0 candidate {key} is invalid: {document_id}")
                else:
                    witness_ids.add(witness)
    allowed = pool.get("allowed_witness_ids")
    if not isinstance(allowed, list) or allowed != sorted(witness_ids):
        errors.append("G0 allowlist does not match candidate document/span/edge IDs")
    errors.extend(validate_g0_identity(pool, baseline))
    if pool != baseline:
        errors.append("G0 pool differs from the complete supplied baseline")
    return sorted(set(errors))


def build_validation(case_name: str) -> dict[str, Any]:
    if case_name not in CASE_DIRECTORIES:
        raise ValueError(f"unsupported case: {case_name}")
    case_directory = CASE_DIRECTORIES[case_name]
    inputs = {name: _read_case_json(case_directory, name) for name in INPUT_NAMES}
    plan = inputs["composition-plan.json"]
    pool = inputs["g0-pool.json"]
    baseline = inputs["g0-baseline.json"]
    plan_errors = validate_composition_plan(plan)
    plan_errors.extend(
        validate_public_obligation_contract(
            plan, EXPECTED_CASE_OBLIGATION_IDS[case_name]
        )
    )
    if plan_errors:
        raise ValueError("invalid composition plan: " + "; ".join(plan_errors))
    pool_errors = _validate_pool(pool, baseline)
    if pool_errors:
        raise ValueError("invalid G0 pool: " + "; ".join(pool_errors))

    obligation_set = build_obligation_set(plan)
    oflat = compile_oflat(plan)
    shape = compile_shape_beat_units(plan, "S")
    beats = compile_shape_beat_units(plan, "SB")
    parity_shape = validate_obligation_parity(plan, oflat["manifest"], shape["manifest"])
    parity_beats = validate_obligation_parity(plan, oflat["manifest"], beats["manifest"])
    oflat_errors = validate_oflat_text(oflat["scaffold_text"])
    satisfiability = audit_plan_pool_satisfiability(plan, obligation_set, pool)
    schedules = {
        arm: schedule_units(plan, arm)
        for arm in ("S", "SB")
    }
    negative_checks = run_negative_checks(case_directory.parent, case_directory.name)
    check_rows = [
        _check("oflat_valid", not oflat_errors, "; ".join(oflat_errors)),
        _check(
            "obligation_parity_shape",
            not parity_shape,
            "; ".join(parity_shape),
        ),
        _check(
            "obligation_parity_beats",
            not parity_beats,
            "; ".join(parity_beats),
        ),
        _check(
            "witness_ids_satisfiable",
            satisfiability["satisfiable"],
            "; ".join(satisfiability["errors"]),
        ),
        _check(
            "expected_schedule",
            schedules == EXPECTED_SCHEDULES[case_name],
            json.dumps(schedules, sort_keys=True),
        ),
        _check(
            "expected_unit_counts",
            {arm: len(value["units"]) for arm, value in (("S", shape), ("SB", beats))}
            == EXPECTED_UNITS[case_name],
            json.dumps(
                {"S": len(shape["units"]), "SB": len(beats["units"])},
                sort_keys=True,
            ),
        ),
        _check(
            "negative_checks",
            len(negative_checks) == 15
            and len({row["check_id"] for row in negative_checks}) == 15
            and all(row.get("passed") is True for row in negative_checks),
            "all 15 registered mutations must be rejected",
        ),
    ]
    check_rows.extend(
        _check(
            f"negative:{row['check_id']}",
            row.get("passed") is True,
            row.get("error") or row.get("mutation", ""),
        )
        for row in negative_checks
    )
    failed = [row["check_id"] for row in check_rows if not row["passed"]]
    input_hashes = {
        name: hashlib.sha256((case_directory / name).read_bytes()).hexdigest()
        for name in INPUT_NAMES
    }
    if input_hashes != EXPECTED_INPUT_SHA256[case_name]:
        raise ValueError("fixture input hashes differ from the fixed public case manifest")
    artifacts = {
        "obligation-set.json": obligation_set,
        "oflat.txt": oflat["scaffold_text"],
        "compiled-units.json": {
            "schema_version": "shape-beat-public-compiled-units-v1",
            "S": shape,
            "SB": beats,
        },
        "schedules.json": {
            "schema_version": "shape-beat-public-schedules-v1",
            "S": schedules["S"],
            "SB": schedules["SB"],
        },
        "validation-report.json": {
            "schema_version": "shape-beat-public-validation-v1",
            "case": case_name,
            "status": "passed" if not failed else "failed",
            "model_executed": False,
            "input_sha256": input_hashes,
            "counts": {
                "required_checks": len(check_rows),
                "passed_checks": sum(row["passed"] for row in check_rows),
                "failed_checks": len(failed),
                "negative_checks_executed": len(negative_checks),
                "negative_checks_passed": sum(
                    row.get("passed") is True for row in negative_checks
                ),
            },
            "checks": check_rows,
            "satisfiability_scope": "exact witness-ID availability only",
            "failure_ids": failed,
        },
    }
    return artifacts


def _write_output(output: Path, artifacts: dict[str, Any]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output.parent}")
    output.mkdir()
    try:
        for name, value in artifacts.items():
            destination = output / name
            if name.endswith(".txt"):
                destination.write_text(value, encoding="utf-8")
            else:
                destination.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    except Exception:
        shutil.rmtree(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=tuple(CASE_DIRECTORIES))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        artifacts = build_validation(args.case)
        report = artifacts["validation-report.json"]
        if report["status"] != "passed":
            raise ValueError("validation failed: " + ", ".join(report["failure_ids"]))
        _write_output(args.output, artifacts)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Shape–Beat public validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "case": args.case,
                "status": "passed",
                "counts": report["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
