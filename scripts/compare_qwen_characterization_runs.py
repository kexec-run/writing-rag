"""Compare two corrected Qwen characterization runs by contract and payload."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_common import read_json, write_json  # noqa: E402
from validate_qwen_embedding_characterization import ensure_fresh_output  # noqa: E402


NUMERIC_TOLERANCE = 1e-5
DETERMINISTIC_FILES = (
    "cases/diagnostic-cases.json",
    "cases/confirmation-cases.json",
    "cases/case-freeze-metadata.json",
    "passages/passage-vectors.json",
    "Q0_plain/query-vectors.json",
    "Q1_card/query-vectors.json",
    "Q0_plain/local-results.json",
    "Q1_card/local-results.json",
    "Q0_plain/global-pool-results.json",
    "Q1_card/global-pool-results.json",
    "comparison/per-case-comparison.json",
    "comparison/category-summary.json",
    "comparison/score-distributions.json",
    "comparison/hub-analysis.json",
)
VOLATILE_METADATA_FIELDS = {"timings", "elapsed_seconds"}


def compare_values(left: Any, right: Any, path: str = "", differences: list[str] | None = None) -> list[str]:
    differences = [] if differences is None else differences
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            differences.append(path)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE):
            differences.append(path)
    elif type(left) is not type(right):
        differences.append(path)
    elif isinstance(left, dict):
        if set(left) != set(right):
            differences.append(path + ".keys")
        for key in sorted(set(left) & set(right)):
            compare_values(left[key], right[key], f"{path}.{key}".strip("."), differences)
    elif isinstance(left, list):
        if len(left) != len(right):
            differences.append(path + ".length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(left_item, right_item, f"{path}[{index}]", differences)
    elif left != right:
        differences.append(path)
    return differences


def canonical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key not in VOLATILE_METADATA_FIELDS}


def compare_runs(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    payload_differences: dict[str, list[str]] = {}
    missing_files: list[str] = []
    for relative in DETERMINISTIC_FILES:
        left_path = left_dir / relative
        right_path = right_dir / relative
        if not left_path.is_file() or not right_path.is_file():
            missing_files.append(relative)
            continue
        differences = compare_values(read_json(left_path), read_json(right_path))
        if differences:
            payload_differences[relative] = differences[:20]
    left_metadata = read_json(left_dir / "run-metadata.json")
    right_metadata = read_json(right_dir / "run-metadata.json")
    metadata_differences = compare_values(canonical_metadata(left_metadata), canonical_metadata(right_metadata))
    timing_differences = compare_values(left_metadata.get("timings"), right_metadata.get("timings"))
    elapsed_differences = compare_values(left_metadata.get("elapsed_seconds"), right_metadata.get("elapsed_seconds"))
    ranking_files = (
        "Q0_plain/local-results.json",
        "Q1_card/local-results.json",
        "Q0_plain/global-pool-results.json",
        "Q1_card/global-pool-results.json",
        "comparison/per-case-comparison.json",
    )
    ranking_differences = {file: payload_differences[file] for file in ranking_files if file in payload_differences}
    result = {
        "left_run": str(left_dir),
        "right_run": str(right_dir),
        "deterministic_payload_equivalent": not missing_files and not payload_differences,
        "numeric_equivalence_within_tolerance": not missing_files and not payload_differences,
        "ranking_equivalent": not missing_files and not ranking_differences,
        "contract_equivalent": not metadata_differences,
        "volatile_metadata_different": bool(timing_differences or elapsed_differences),
        "volatile_metadata_differences": {
            "timings": timing_differences,
            "elapsed_seconds": elapsed_differences,
        },
        "missing_files": missing_files,
        "payload_differences": payload_differences,
        "contract_differences": metadata_differences,
    }
    result["repeatability_passed"] = all(
        (
            result["deterministic_payload_equivalent"],
            result["numeric_equivalence_within_tolerance"],
            result["ranking_equivalent"],
            result["contract_equivalent"],
        )
    )
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen characterization repeatability",
        "",
        f"Repeatability passed: `{result['repeatability_passed']}`",
        "",
        "| Dimension | Result |",
        "|---|---:|",
        f"| Deterministic payload equivalence | {result['deterministic_payload_equivalent']} |",
        f"| Numeric equivalence within tolerance | {result['numeric_equivalence_within_tolerance']} |",
        f"| Ranking equivalence | {result['ranking_equivalent']} |",
        f"| Contract equivalence | {result['contract_equivalent']} |",
        f"| Volatile metadata differs | {result['volatile_metadata_different']} |",
        "",
        "Vectors, norms, scores, rankings, margins, candidate traces, flags, summaries, case hashes, and model contract fields are compared separately from elapsed time and timing metadata.",
        "",
        "Elapsed time and timing fields are permitted to differ between runs and are reported as volatile metadata rather than silently ignored.",
    ]
    if result["missing_files"]:
        lines.extend(["", "## Missing files", "", *[f"- `{item}`" for item in result["missing_files"]]])
    if result["payload_differences"]:
        lines.extend(["", "## Payload differences", "", *[f"- `{item}`: {values[:3]}" for item, values in result["payload_differences"].items()]])
    if result["contract_differences"]:
        lines.extend(["", "## Contract differences", "", *[f"- `{item}`" for item in result["contract_differences"][:20]]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", default="data/generated/qwen-embedding-characterization-v1-repeatability")
    args = parser.parse_args()
    result = compare_runs(Path(args.left), Path(args.right))
    output_dir = Path(args.output)
    ensure_fresh_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "repeatability.json", result)
    (output_dir / "repeatability.md").write_text(render_report(result), encoding="utf-8")
    print(output_dir / "repeatability.md")
    if not result["repeatability_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
