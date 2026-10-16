"""Compare deterministic payloads from two accepted Jina Q2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DETERMINISTIC_FILES = (
    "cases/diagnostic-cases.json",
    "cases/confirmation-cases.json",
    "model-contract/resolved-contract.json",
    "retrieval/passages.json",
    "retrieval/query-vectors.json",
    "retrieval/q2-local-results.json",
    "retrieval/q2-global-results.json",
    "comparison/q2-vs-qwen-candidate-trace.json",
    "comparison/q2-vs-qwen-case-summary.json",
    "comparison/q2-vs-qwen-summary.json",
)


def canonical_json(path: Path) -> bytes:
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compare(first: Path, second: Path) -> dict[str, Any]:
    files = []
    for relative in DETERMINISTIC_FILES:
        left = canonical_json(first / relative)
        right = canonical_json(second / relative)
        files.append({"path": relative, "identical": left == right, "first_sha256": sha256(left), "second_sha256": sha256(right)})
    metadata_first = json.loads((first / "run-metadata.json").read_text(encoding="utf-8"))
    metadata_second = json.loads((second / "run-metadata.json").read_text(encoding="utf-8"))
    stable_fields = ("experiment", "model_name", "task", "query_prompt_name", "document_prompt_name", "case_count", "batch_size", "device", "python_executable", "python_version")
    stable = {field: metadata_first.get(field) == metadata_second.get(field) for field in stable_fields}
    timing_fields = {"first_total_elapsed_seconds": metadata_first.get("timings", {}).get("total_elapsed_seconds"), "second_total_elapsed_seconds": metadata_second.get("timings", {}).get("total_elapsed_seconds")}
    return {"status": "passed" if all(item["identical"] for item in files) and all(stable.values()) else "failed", "deterministic_payload_identical": all(item["identical"] for item in files), "canonical_metadata_equivalent": all(stable.values()), "volatile_timing": timing_fields, "stable_metadata_fields": stable, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(Path(args.first), Path(args.second))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "repeatability.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "repeatability.md").write_text("# Jina Q2 repeatability\n\n" + json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
