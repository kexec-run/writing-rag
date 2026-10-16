"""Compare deterministic payloads from two Jina multi-mode pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DETERMINISTIC_FILES = (
    "cases/diagnostic-cases.json",
    "cases/confirmation-cases.json",
    "labelled-collections/samples.json",
    "labelled-collections/classification-splits.json",
    "model-contract/resolved-contract.json",
    "modes/text-matching/vectors.json",
    "modes/text-matching/results.json",
    "modes/clustering/vectors.json",
    "modes/clustering/results.json",
    "modes/classification/vectors.json",
    "modes/classification/results.json",
    "retrieval-control/summary.json",
)
VOLATILE_KEYS = {"elapsed_seconds", "rss_before_bytes", "rss_after_bytes"}


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonical(child) for key, child in sorted(value.items()) if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [canonical(child) for child in value]
    return value


def read_canonical(path: Path) -> Any:
    return canonical(json.loads(path.read_text(encoding="utf-8")))


def compare(first: Path, second: Path) -> dict[str, Any]:
    files = []
    for relative in DETERMINISTIC_FILES:
        left = read_canonical(first / relative)
        right = read_canonical(second / relative)
        files.append({"path": relative, "identical": left == right})
    first_meta = read_canonical(first / "run-metadata.json")
    second_meta = read_canonical(second / "run-metadata.json")
    stable_fields = (
        "experiment",
        "model_name",
        "device",
        "batch_size",
        "case_count",
        "dependency_versions",
        "qwen_reference_hashes",
        "q2_reference_hashes",
        "classification_split_hash",
        "mode_vector_counts",
    )
    stable = {field: first_meta.get(field) == second_meta.get(field) for field in stable_fields}
    return {
        "status": "passed" if all(item["identical"] for item in files) and all(stable.values()) else "failed",
        "deterministic_payload_identical": all(item["identical"] for item in files),
        "canonical_metadata_equivalent": all(stable.values()),
        "stable_metadata_fields": stable,
        "volatile_runtime_present": bool(first_meta.get("mode_runtime")) and bool(second_meta.get("mode_runtime")),
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
    (output / "repeatability.md").write_text("# Jina mode repeatability\n\n" + json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
