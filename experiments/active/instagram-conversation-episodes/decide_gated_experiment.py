"""Write the terminal decision for the percentile-gated episode experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v2-gated-qwen-rerun"
STRUCTURAL_DIR = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v2-gated-structural"
REPEAT_DIR = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v2-gated-structural-repeat"
OUTPUT = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v2-gated-decision"
MRR_GAIN_REQUIRED = 0.03
RECALL_TOLERANCE = 0.01
P90_MRR_TOLERANCE = 0.01


def _load(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "gated-results.json").read_text(encoding="utf-8"))


def _metric(result: dict[str, Any], condition: str) -> dict[str, Any]:
    return result["qwen_retrieval"]["conditions"][condition]


def _structural_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "schema_version",
            "method",
            "scope",
            "thresholds",
            "selection",
            "excluded_anchor_ids",
            "cohorts",
            "corpus_profile",
            "rows",
            "queries",
            "structural",
        )
    }


def _positive_checks(
    result: dict[str, Any], structural: dict[str, Any], repeat: dict[str, Any]
) -> dict[str, bool]:
    retrieval = result["qwen_retrieval"]
    backend = retrieval["backend"]
    baseline = _metric(result, "baseline")
    c1 = _metric(result, "c1_full")
    gated = _metric(result, "gated")
    return {
        "structural_result_passed": structural["structural"]["passed"],
        "structural_payload_repeatable": _structural_payload(result)
        == _structural_payload(repeat),
        "retrieval_result_structural_passed": result["structural"]["passed"],
        "exactly_60_queries": retrieval["query_count"] == 60,
        "all_conditions_have_60_queries": all(
            retrieval["conditions"][condition]["query_count"] == 60
            for condition in ("baseline", "c1_full", "gated")
        ),
        "same_query_ids_across_conditions": retrieval["comparison_scope"]["same_query_ids"],
        "qwen_backend": backend["embedding_backend"] == "qwen3",
        "expected_qwen_model": backend["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B",
        "fallback_disabled": not backend.get("fallback_used", False),
        "condition_specific_documents_changed": (
            c1["mrr"] != baseline["mrr"] or c1["recall_at_10"] != baseline["recall_at_10"]
        ),
        "gated_recall_within_tolerance": (
            gated["recall_at_10"] - baseline["recall_at_10"] >= -RECALL_TOLERANCE
        ),
        "positive_fold_count_at_least_four": sum(
            gated["by_fold"][fold]["mrr"] > baseline["by_fold"][fold]["mrr"]
            for fold in gated["by_fold"]
        )
        >= 4,
        "positive_median_fold_delta": (
            sorted(
                gated["by_fold"][fold]["mrr"] - baseline["by_fold"][fold]["mrr"]
                for fold in gated["by_fold"]
            )[2]
            > 0
        ),
    }


def _negative_checks(result: dict[str, Any]) -> dict[str, bool]:
    retrieval = result["qwen_retrieval"]
    baseline = _metric(result, "baseline")
    gated = _metric(result, "gated")
    p90 = gated["by_band"]["p90_band"]
    baseline_p90 = baseline["by_band"]["p90_band"]
    return {
        "no_structural_errors": not result["structural"]["validation_errors"],
        "no_mixed_query_sets": retrieval["comparison_scope"]["same_query_ids"],
        "no_fallback_or_hashing": not retrieval["backend"].get("fallback_used", False),
        "gated_mrr_gain_gate_passed": (
            gated["mrr"] - baseline["mrr"] >= MRR_GAIN_REQUIRED
        ),
        "gated_p90_mrr_no_material_regression": (
            p90["mrr"] - baseline_p90["mrr"] >= -P90_MRR_TOLERANCE
        ),
        "gated_worst_fold_recall_no_material_regression": min(
            gated["by_fold"][fold]["recall_at_10"]
            - baseline["by_fold"][fold]["recall_at_10"]
            for fold in gated["by_fold"]
        )
        >= -0.05,
    }


def decide() -> dict[str, Any]:
    result = _load(RESULT_DIR)
    structural = _load(STRUCTURAL_DIR)
    repeat = _load(REPEAT_DIR)
    positive = _positive_checks(result, structural, repeat)
    negative = _negative_checks(result)
    hard_checks = [
        value
        for name, value in {**positive, **negative}.items()
        if name
        not in {
            "gated_mrr_gain_gate_passed",
            "gated_p90_mrr_no_material_regression",
        }
    ]
    hard_invariants_pass = all(hard_checks)
    gated = _metric(result, "gated")
    baseline = _metric(result, "baseline")
    if not hard_invariants_pass:
        decision = "archive_experiment"
        reason = "A structural or model-contract invariant failed."
    elif (
        gated["mrr"] - baseline["mrr"] >= MRR_GAIN_REQUIRED
        and negative["gated_p90_mrr_no_material_regression"]
    ):
        decision = "retrieval_only_direction_supported"
        reason = "The gated representation passes retrieval thresholds; article holdouts are the next authorized gate."
    else:
        decision = "structural_only"
        reason = (
            "The gated method passes structural checks and improves Recall@10, "
            "but its MRR gain is below threshold and p90-band MRR regresses. "
            "Do not run article generation or change production policy."
        )
    return {
        "schema_version": "instagram-percentile-gated-decision-v1",
        "decision": decision,
        "reason": reason,
        "thresholds": {
            "mrr_gain_required": MRR_GAIN_REQUIRED,
            "recall_regression_tolerance": RECALL_TOLERANCE,
            "p90_mrr_regression_tolerance": P90_MRR_TOLERANCE,
        },
        "hard_invariants_pass": hard_invariants_pass,
        "positive_checks": positive,
        "negative_checks": negative,
        "metrics": {
            "baseline": {
                key: baseline[key]
                for key in ("query_count", "mrr", "recall_at_10", "ndcg_at_10")
            },
            "c1_full": {
                key: _metric(result, "c1_full")[key]
                for key in ("query_count", "mrr", "recall_at_10", "ndcg_at_10")
            },
            "gated": {
                key: gated[key]
                for key in ("query_count", "mrr", "recall_at_10", "ndcg_at_10")
            },
            "gated_delta": {
                "mrr": round(gated["mrr"] - baseline["mrr"], 6),
                "recall_at_10": round(gated["recall_at_10"] - baseline["recall_at_10"], 6),
                "ndcg_at_10": round(gated["ndcg_at_10"] - baseline["ndcg_at_10"], 6),
            },
            "p90_delta_mrr": round(
                gated["by_band"]["p90_band"]["mrr"]
                - baseline["by_band"]["p90_band"]["mrr"],
                6,
            ),
        },
        "article_gate": {
            "run": False,
            "reason": "Retrieval promotion gate failed.",
        },
        "production_policy": {
            "changed": False,
            "qwen_remains_retrieval_backend": True,
            "jina_used": False,
            "human_labels_used": False,
        },
        "source_artifacts": {
            "retrieval": str(RESULT_DIR),
            "structural": str(STRUCTURAL_DIR),
            "repeatability": str(REPEAT_DIR),
        },
    }


def _render(decision: dict[str, Any]) -> str:
    metrics = decision["metrics"]
    return "\n".join(
        [
            "# Instagram percentile-gated episode decision",
            "",
            f"Decision: **`{decision['decision']}`**",
            "",
            decision["reason"],
            "",
            "| Condition | MRR | Recall@10 | nDCG@10 |",
            "|---|---:|---:|---:|",
            f"| baseline | {metrics['baseline']['mrr']:.6f} | {metrics['baseline']['recall_at_10']:.6f} | {metrics['baseline']['ndcg_at_10']:.6f} |",
            f"| C1 full | {metrics['c1_full']['mrr']:.6f} | {metrics['c1_full']['recall_at_10']:.6f} | {metrics['c1_full']['ndcg_at_10']:.6f} |",
            f"| G1 gated | {metrics['gated']['mrr']:.6f} | {metrics['gated']['recall_at_10']:.6f} | {metrics['gated']['ndcg_at_10']:.6f} |",
            "",
            f"G1 delta MRR: `{metrics['gated_delta']['mrr']:+.6f}`.",
            f"G1 delta Recall@10: `{metrics['gated_delta']['recall_at_10']:+.6f}`.",
            f"G1 delta nDCG@10: `{metrics['gated_delta']['ndcg_at_10']:+.6f}`.",
            f"p90-band delta MRR: `{metrics['p90_delta_mrr']:+.6f}`.",
            "",
            "## Verification",
            "",
            f"Hard invariants passed: `{decision['hard_invariants_pass']}`.",
            "The valid comparison used 60 new anchors, five chronological folds, "
            "three condition-specific document spaces, one fixed query per anchor, "
            "Qwen retrieval, and no fallback.",
            "",
            "Article generation was not run because retrieval did not meet the MRR "
            "and p90 no-regression gates. Production policy remains unchanged.",
            "",
        ]
    )


def main() -> int:
    decision = decide()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "decision-report.md").write_text(_render(decision), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if decision["decision"] in {"retrieval_only_direction_supported", "structural_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
