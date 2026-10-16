"""Validate the bounded article-plan shadow gate from persisted artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from article_plan import validate_article_plan


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _target_summary(path: Path, expected_family: str) -> dict[str, Any]:
    plan = _read(path)
    validation_errors = validate_article_plan(plan)
    packets = list(plan.get("source_packets", []))
    families = Counter(str(packet.get("source_family", "")) for packet in packets)
    coverage = plan.get("coverage", {})
    style_escapes = [
        str(packet.get("document_id", ""))
        for packet in packets
        if str(packet.get("metadata", {}).get("collection", "")) == "style_memory"
        or str(packet.get("source_scope", "")).startswith("style_")
    ]
    passed = (
        not validation_errors
        and plan.get("status") == "ready"
        and families.get(expected_family, 0) == len(packets)
        and bool(packets)
        and float(coverage.get("source_span_coverage", -1.0)) == 1.0
        and not style_escapes
    )
    return {
        "path": str(path),
        "expected_family": expected_family,
        "status": plan.get("status"),
        "packet_count": len(packets),
        "source_family_counts": dict(sorted(families.items())),
        "source_span_count": coverage.get("source_span_count"),
        "hashed_source_span_count": coverage.get("hashed_source_span_count"),
        "source_span_coverage": coverage.get("source_span_coverage"),
        "validation_errors": validation_errors,
        "style_escape_document_ids": style_escapes,
        "passed": passed,
    }


def _retrieval_comparison(message_metadata: Path, baseline_metadata: Path) -> dict[str, Any]:
    current = _read(message_metadata)
    baseline = _read(baseline_metadata)
    current_content = list(current.get("content_document_ids", []))
    baseline_content = list(baseline.get("content_document_ids", []))
    current_style = list(current.get("style_document_ids", []))
    baseline_style = list(baseline.get("style_document_ids", []))
    return {
        "content_ids_equal": current_content == baseline_content,
        "content_order_equal": current_content == baseline_content,
        "style_ids_equal": current_style == baseline_style,
        "qwen_retrieval_drift_fraction": 0.0 if current_content == baseline_content else 1.0,
        "current_content_document_ids": current_content,
        "baseline_content_document_ids": baseline_content,
    }


def build_gate_report(args: argparse.Namespace) -> dict[str, Any]:
    message = _target_summary(Path(args.message_plan), "message")
    mdx = _target_summary(Path(args.mdx_plan), "document")
    retrieval = _retrieval_comparison(
        Path(args.message_metadata),
        Path(args.baseline_metadata),
    )
    fixture_gate = {
        "positive_fixture_count": args.positive_fixture_count,
        "negative_fixture_count": args.negative_fixture_count,
        "minimum_positive_fixture_count": 8,
        "minimum_negative_fixture_count": 8,
        "passed": args.positive_fixture_count >= 8 and args.negative_fixture_count >= 8,
    }
    passed = (
        message["passed"]
        and mdx["passed"]
        and retrieval["qwen_retrieval_drift_fraction"] == 0.0
        and fixture_gate["passed"]
    )
    return {
        "schema_version": "article-plan-gate-v1",
        "status": "passed" if passed else "failed",
        "targets": {"message": message, "mdx": mdx},
        "qwen_retrieval_comparison": retrieval,
        "fixture_gate": fixture_gate,
        "style_authority_escape_count": len(message["style_escape_document_ids"])
        + len(mdx["style_escape_document_ids"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message-plan", required=True)
    parser.add_argument("--mdx-plan", required=True)
    parser.add_argument("--message-metadata", required=True)
    parser.add_argument("--baseline-metadata", required=True)
    parser.add_argument("--positive-fixture-count", type=int, required=True)
    parser.add_argument("--negative-fixture-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_gate_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
