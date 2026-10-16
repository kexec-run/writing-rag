"""Compare deterministic payloads from two Jina text-matching runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DETERMINISTIC_FILES = (
    "cases/diagnostic-cases.json",
    "cases/confirmation-cases.json",
    "retrieval/vectors.json",
    "retrieval/canonical-passages.json",
    "comparison/q3-local-results.json",
    "comparison/q3-global-results.json",
    "comparison/q3-vs-qwen-candidate-trace.json",
    "comparison/summary.json",
    "model-contract/resolved-contract.json",
    "model-contract/smoke-test.json",
)
VOLATILE_KEYS = {"encoding_seconds", "elapsed_seconds", "platform", "python_executable"}


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonical(child) for key, child in sorted(value.items()) if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [canonical(child) for child in value]
    return value


def read_canonical(path: Path) -> Any:
    return canonical(json.loads(path.read_text(encoding="utf-8")))


def read_raw(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(first: Path, second: Path) -> dict[str, Any]:
    files = []
    for relative in DETERMINISTIC_FILES:
        left = read_canonical(first / relative)
        right = read_canonical(second / relative)
        files.append({"path": relative, "identical": left == right})
    first_meta = read_canonical(first / "run-metadata.json")
    second_meta = read_canonical(second / "run-metadata.json")
    first_raw_meta = read_raw(first / "run-metadata.json")
    second_raw_meta = read_raw(second / "run-metadata.json")
    stable_fields = (
        "experiment",
        "model_name",
        "resolved_model",
        "requested_device",
        "batch_size",
        "task",
        "prompt_name",
        "case_count",
        "canonical_passage_count",
        "unique_vector_count",
        "vector_dimension",
        "raw_output_dtype",
        "analysis_dtype",
        "qwen_reference_hashes",
    )
    stable = {field: first_meta.get(field) == second_meta.get(field) for field in stable_fields}
    deterministic = all(item["identical"] for item in files)
    metadata_equivalent = all(stable.values())
    return {
        "status": "passed" if deterministic and metadata_equivalent else "failed",
        "deterministic_payload_identical": deterministic,
        "canonical_metadata_equivalent": metadata_equivalent,
        "stable_metadata_fields": stable,
        "volatile_runtime_present": first_raw_meta.get("timings") != second_raw_meta.get("timings"),
        "files": files,
    }


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
    (output / "repeatability.md").write_text("# Jina text-matching repeatability\n\n" + json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
