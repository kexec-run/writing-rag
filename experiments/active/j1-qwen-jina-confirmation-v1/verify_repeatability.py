"""Verify exact ranking repeatability for two frozen Jina result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(
    left_path: Path,
    right_path: Path,
    conditions: tuple[str, ...] = ("baseline", "c1_full", "jina_selected"),
) -> dict[str, Any]:
    left = _load(left_path)
    right = _load(right_path)
    errors: list[str] = []
    if left.get("substrate_sha256") != right.get("substrate_sha256"):
        errors.append("substrate_sha256 differs")
    for condition in conditions:
        left_rows = {row["query_id"]: row for row in left["conditions"][condition]["trace_rows"]}
        right_rows = {row["query_id"]: row for row in right["conditions"][condition]["trace_rows"]}
        if set(left_rows) != set(right_rows):
            errors.append(f"query IDs differ for {condition}")
            continue
        for query_id in sorted(left_rows):
            if left_rows[query_id].get("reranked_ids") != right_rows[query_id].get("reranked_ids"):
                errors.append(f"complete ranking differs: {condition}/{query_id}")
    return {"passed": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("baseline", "c1_full", "jina_selected"),
        default=("baseline", "c1_full", "jina_selected"),
    )
    args = parser.parse_args()
    result = verify(Path(args.left), Path(args.right), tuple(args.conditions))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
