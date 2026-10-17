"""Verify exact ranking repeatability for two frozen Jina result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments/active/qwen-top50-reranker-v2"))


def _load_validated_result(path: Path, substrate_path: Path | None) -> dict[str, Any]:
    from replay_frozen_rankings import load_validated_result

    return load_validated_result(path, substrate_path)


def verify(
    left_path: Path,
    right_path: Path,
    conditions: tuple[str, ...] = ("baseline", "c1_full", "jina_selected"),
    substrate_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        left = _load_validated_result(left_path, substrate_path)
        right = _load_validated_result(right_path, substrate_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    left_mode = left.get("execution_mode", "model_execution")
    right_mode = right.get("execution_mode", "model_execution")
    if left_mode != right_mode:
        errors.append("execution_mode differs")
    if left.get("substrate_sha256") != right.get("substrate_sha256"):
        errors.append("substrate_sha256 differs")
    for condition in conditions:
        left_condition = left.get("conditions", {}).get(condition, {})
        right_condition = right.get("conditions", {}).get(condition, {})
        left_trace = left_condition.get("trace_rows")
        right_trace = right_condition.get("trace_rows")
        if not isinstance(left_trace, list) or not isinstance(right_trace, list):
            errors.append(f"ranking trace missing for {condition}")
            continue
        left_ids = [row.get("query_id") for row in left_trace]
        right_ids = [row.get("query_id") for row in right_trace]
        if len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
            errors.append(f"duplicate query IDs for {condition}")
            continue
        left_rows = {row["query_id"]: row for row in left_trace}
        right_rows = {row["query_id"]: row for row in right_trace}
        if set(left_rows) != set(right_rows):
            errors.append(f"query IDs differ for {condition}")
            continue
        for query_id in sorted(left_rows):
            left_ranking = left_rows[query_id].get("reranked_ids")
            right_ranking = right_rows[query_id].get("reranked_ids")
            if not isinstance(left_ranking, list) or not isinstance(right_ranking, list):
                errors.append(f"complete ranking missing: {condition}/{query_id}")
            elif len(left_ranking) != len(set(left_ranking)) or len(right_ranking) != len(set(right_ranking)):
                errors.append(f"duplicate candidate IDs: {condition}/{query_id}")
            elif left_ranking != right_ranking:
                errors.append(f"complete ranking differs: {condition}/{query_id}")
    return {"passed": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--substrate")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("baseline", "c1_full", "jina_selected"),
        default=("baseline", "c1_full", "jina_selected"),
    )
    args = parser.parse_args()
    result = verify(
        Path(args.left),
        Path(args.right),
        tuple(args.conditions),
        Path(args.substrate) if args.substrate else None,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
