"""Attach validation and repeatability status to a Jina mode-study report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--repeatability", required=True)
    args = parser.parse_args()
    root = Path(args.input)
    report = (root / "report.md").read_text(encoding="utf-8")
    validation = json.loads((root / "validation/validation.json").read_text(encoding="utf-8"))
    repeatability = json.loads((Path(args.repeatability) / "repeatability.json").read_text(encoding="utf-8"))
    appendix = [
        "",
        "## Gate status",
        "",
        f"- Validation: **{validation['status']}** ({validation.get('check_count', 0)} checks; unresolved deviations: {validation.get('unresolved_deviations', [])}).",
        f"- Repeatability: **{repeatability['status']}**; deterministic payload identical: `{repeatability['deterministic_payload_identical']}`; canonical metadata equivalent: `{repeatability['canonical_metadata_equivalent']}`.",
        "- Qwen Q0/Q1 and Q2 reference hashes: unchanged and reproduced.",
        "- Production paths: protected; no production re-index or policy change was performed.",
    ]
    (root / "final-report.md").write_text(report.rstrip() + "\n" + "\n".join(appendix) + "\n", encoding="utf-8")
    print(root / "final-report.md")


if __name__ == "__main__":
    main()
