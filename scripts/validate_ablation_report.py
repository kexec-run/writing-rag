#!/usr/bin/env python3
"""Validate ablation artifacts against diagnosis-report-v1 / ablation-run-v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FAILURE_CLASSES = {"evidence", "qualifier", "scope", "structure", "protocol", "other"}
TAXONOMY_CLASSES = {
    "representation/schema failure",
    "retrieval/evidence acquisition failure",
    "generation/projection/scoping failure",
    "composition decomposition failure",
    "composition contract or terminal quality failure",
}
VERDICTS = {"accepted", "rejected", "inconclusive", "invalid_harness_state"}
HOLDOUT_STATUSES = {"target_removed", "target_absent", "not_run"}
REQUIRED_RUN_KEYS = {
    "word_count": int,
    "tokens_prompt": int,
    "tokens_output": int,
    "claim_verifier_failures": int,
    "qualifier_failures": int,
    "repeat_determinism": bool,
    "call_count": int,
    "repair_count": int,
}
REQUIRED_PROVENANCE = {
    "qwen_only": bool,
    "jina_used": bool,
    "production_index_changed": bool,
    "generator_model": str,
    "max_token_budget": int,
}


def _fail(errors: list[str], path: Path, message: str) -> None:
    errors.append(f"{path}: {message}")


def validate_diagnosis_report(data: Any, path: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"{path}: expected JSON object"]
    if data.get("schema") != "diagnosis-report-v1":
        _fail(errors, path, "missing schema=diagnosis-report-v1")
    if data.get("taxonomy_class") not in TAXONOMY_CLASSES:
        _fail(errors, path, f"invalid taxonomy_class: {data.get('taxonomy_class')}")
    if data.get("global_verdict") not in VERDICTS:
        _fail(errors, path, f"invalid global_verdict: {data.get('global_verdict')}")
    for key in ("parts_total", "parts_accepted", "parts_rejected"):
        value = data.get(key)
        if not isinstance(value, int) or value < 0:
            _fail(errors, path, f"invalid {key}: {value!r}")
    ratio = data.get("mean_support_ratio")
    if ratio is not None and not isinstance(ratio, (int, float)):
        _fail(errors, path, "mean_support_ratio must be numeric or null")
    totals = data.get("failure_class_totals") or {}
    unknown = set(totals) - FAILURE_CLASSES
    if unknown:
        _fail(errors, path, f"unknown failure classes: {sorted(unknown)}")
    for index, part in enumerate(data.get("parts") or []):
        verdict = part.get("verdict")
        if verdict not in {
            "accepted",
            "repaired_accepted",
            "rejected",
            "not_attempted",
            "inconclusive",
        }:
            _fail(errors, path, f"part[{index}] invalid verdict: {verdict!r}")
        part_classes = set(part.get("failure_classes") or {})
        if part_classes - FAILURE_CLASSES:
            _fail(errors, path, f"part[{index}] unknown class in {part_classes}")
    return errors


def _validate_run_row(row: dict[str, Any], path: Path, where: str) -> list[str]:
    errors: list[str] = []
    run_failed = row.get("run_failed")
    if run_failed is not None and not isinstance(run_failed, dict):
        _fail(errors, path, f"{where}: run_failed must be an object with return_code")
    if not isinstance(row.get("accepted"), bool) and run_failed is None:
        if row.get("acceptance_applicable") is False:
            pass
        else:
            _fail(errors, path, f"{where}: missing boolean 'accepted' (or run_failed)")
    for key, kind in REQUIRED_RUN_KEYS.items():
        value = row.get(key)
        if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
            _fail(errors, path, f"{where}: invalid {key}: {value!r}")
    holdout = row.get("holdout_status")
    if holdout not in HOLDOUT_STATUSES:
        _fail(errors, path, f"{where}: invalid holdout_status: {holdout!r}")
    for key, kind in REQUIRED_PROVENANCE.items():
        value = row.get(key)
        if kind is bool:
            ok = isinstance(value, bool)
        else:
            ok = isinstance(value, kind) and not isinstance(value, bool)
        if not ok:
            _fail(errors, path, f"{where}: invalid provenance {key}: {value!r}")
    classes = row.get("failure_class_totals") or {}
    if set(classes) - FAILURE_CLASSES:
        _fail(
            errors,
            path,
            f"{where}: unknown failure classes {sorted(set(classes) - FAILURE_CLASSES)}",
        )
    failures = row.get("failures")
    if failures is not None and not isinstance(failures, list):
        _fail(errors, path, f"{where}: failures must be a list")
    return errors


def validate_run_report(data: Any, path: Path) -> list[str]:
    errors: list[str] = []
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, dict) and data.get("schema") == "ablation-run-v1":
        runs = data.get("runs") or {}
        if not isinstance(runs, dict):
            return [f"{path}: ablation-run-v1.runs must be an object"]
        rows = list(runs.items())
        if not rows:
            if data.get("decision") in {
                "confirmed",
                "not_confirmed",
                "inconclusive_structural",
            }:
                return []
            return [
                f"{path}: no ablation-run-v1 rows detected"
            ]
    elif isinstance(data, dict):
        if isinstance(data.get("runs"), dict):
            rows = list(data["runs"].items())
        else:
            rows = [
                (key, value)
                for key, value in data.items()
                if isinstance(value, dict)
                and ("accepted" in value or "run_failed" in value)
            ]
        if not rows:
            if data.get("decision") in {
                "confirmed",
                "not_confirmed",
                "inconclusive_structural",
            }:
                return []
            return [f"{path}: no ablation-run-v1 rows detected"]
    else:
        return [f"{path}: expected JSON object"]

    provenance_checked = False
    for name, row in rows:
        errors.extend(_validate_run_row(row, path, str(name)))
        if REQUIRED_PROVENANCE.keys() & row.keys():
            provenance_checked = True
    if not provenance_checked:
        _fail(errors, path, "no provenance flags found on any row")
    return errors


def validate_file(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc})"]
    if isinstance(data, dict) and data.get("schema") == "diagnosis-report-v1":
        return validate_diagnosis_report(data, path)
    return validate_run_report(data, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", dest="reports", required=True)
    args = parser.parse_args()
    errors: list[str] = []
    for name in args.reports:
        path = Path(name)
        if not path.is_file():
            errors.append(f"{path}: not found")
            continue
        errors.extend(validate_file(path))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"validated {len(args.reports)} report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
