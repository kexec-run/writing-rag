"""Fail-closed validation for zembed retrieval artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    contract = read_json(root / "model-contract/contract.json")
    result_path = root / "retrieval/results.json"
    check("contract_passed", contract.get("passed") is True, "card contract must pass")
    model_name = contract.get("contract", {}).get("model_name")
    resolved = contract.get("contract", {}).get("resolved_model", "")
    check("mlx_model_only", model_name == "lexrivera/zembed-1-embedding-mlx-6Bit" and "zeroentropy" not in resolved, "only the requested MLX checkpoint may execute")
    check("no_fallback", contract.get("contract", {}).get("fallback_used") is False, "fallback is forbidden")
    check("dimension", contract.get("contract", {}).get("dimension") == 2560, "zembed output must be 2560-dimensional")
    if result_path.is_file():
        result = read_json(result_path)
        candidate = result.get("candidate", {})
        check("both_modes", set(candidate.get("modes", {})) == {"dense_only", "hybrid"}, "dense and hybrid modes are required")
        check("candidate_count", candidate.get("eligible_document_count", 0) > 0, "eligible corpus cannot be empty")
        check("comparison_present", set(result.get("comparison", {})) == {"dense_only", "hybrid"}, "both comparisons are required")
    else:
        check("retrieval_result_present", False, "retrieval results are missing")
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/zembed-retrieval-validity-v1")
    args = parser.parse_args()
    result = validate(Path(args.input))
    output = Path(args.input) / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
