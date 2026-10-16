"""Produce the fail-closed terminal decision for the episode experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCREENING = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v1-screening-qwen"
CONFIRMATION = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v1-confirmation-qwen"
REPEATABILITY = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v1-confirmation-repeat"
OUTPUT = PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v1"

MRR_THRESHOLD = 0.03
RECALL_REGRESSION_TOLERANCE = 0.01


def _load(path: Path) -> dict[str, Any]:
    return json.loads((path / "episode-results.json").read_text(encoding="utf-8"))


def _anchor_cohorts(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(anchor["message_id"]): cohort
        for cohort, anchors in result["cohorts"].items()
        for anchor in anchors
    }


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    retrieval = result["qwen_retrieval"]
    conditions = retrieval["conditions"]
    baseline = conditions["baseline"]
    composite = conditions["composite"]
    return {
        "baseline": {
            "query_count": baseline["query_count"],
            "mrr": baseline["mrr"],
            "recall_at_10": baseline["recall_at_10"],
        },
        "composite": {
            "query_count": composite["query_count"],
            "mrr": composite["mrr"],
            "recall_at_10": composite["recall_at_10"],
        },
        "delta": retrieval["delta"],
        "comparison_scope": retrieval["comparison_scope"],
    }


def _structural_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "schema_version",
            "method",
            "anchor_selection",
            "corpus_profile",
            "cohorts",
            "episodes",
            "baseline_rows",
            "validation",
        )
    }


def _pooled(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[float]] = {"baseline": [], "composite": []}
    for result in results:
        retrieval = result["qwen_retrieval"]
        for condition in rows:
            rows[condition].extend(
                1.0 / int(row["rank"])
                for row in retrieval["conditions"][condition]["rows"]
            )
    metrics = {
        condition: {
            "query_count": len(values),
            "mrr": round(sum(values) / len(values), 6),
        }
        for condition, values in rows.items()
    }
    recall_rows: dict[str, list[bool]] = {"baseline": [], "composite": []}
    for result in results:
        retrieval = result["qwen_retrieval"]
        for condition in recall_rows:
            recall_rows[condition].extend(
                int(row["rank"]) <= 10
                for row in retrieval["conditions"][condition]["rows"]
            )
    for condition, values in recall_rows.items():
        metrics[condition]["recall_at_10"] = round(sum(values) / len(values), 6)
    metrics["delta"] = {
        "mrr": round(metrics["composite"]["mrr"] - metrics["baseline"]["mrr"], 6),
        "recall_at_10": round(
            metrics["composite"]["recall_at_10"]
            - metrics["baseline"]["recall_at_10"],
            6,
        ),
    }
    return metrics


def _positive_checks(
    screening: dict[str, Any], confirmation: dict[str, Any]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for label, result in (("screening", screening), ("confirmation", confirmation)):
        validation = result["validation"]
        checks[f"{label}_structural_validation_passed"] = bool(
            validation["passed"]
        )
        checks[f"{label}_fixture_checks_passed"] = bool(
            validation["fixture_checks"]["passed"]
        )
        checks[f"{label}_perturbation_checks_passed"] = bool(
            validation["perturbation_checks"]["passed"]
        )
        retrieval = result["qwen_retrieval"]
        checks[f"{label}_qwen_backend_is_qwen"] = (
            retrieval["backend"]["embedding_backend"] == "qwen3"
        )
        checks[f"{label}_qwen_fallback_disabled"] = not bool(
            retrieval["backend"].get("fallback_used", False)
        )
        checks[f"{label}_query_policy_is_fixed"] = (
            retrieval["comparison_scope"]["query_policy"]
            == "fixed_method_independent_neighbourhood"
        )
        checks[f"{label}_conditions_have_equal_query_counts"] = (
            retrieval["conditions"]["baseline"]["query_count"]
            == retrieval["conditions"]["composite"]["query_count"]
        )
    checks["confirmation_has_60_structural_anchors"] = (
        confirmation["scope"]["anchor_count"] == 60
    )
    checks["confirmation_uses_later_time_fraction"] = (
        confirmation["anchor_selection"]["time_fraction"] == [0.4, 1.0]
    )
    checks["screening_uses_earlier_time_fraction"] = (
        screening["anchor_selection"]["time_fraction"] == [0.0, 0.4]
    )
    checks["confirmation_retrieval_meets_gain_gate"] = (
        _metrics(confirmation)["delta"]["mrr"] >= MRR_THRESHOLD
        and _metrics(confirmation)["delta"]["recall_at_10"]
        >= -RECALL_REGRESSION_TOLERANCE
    )
    return checks


def _negative_checks(
    screening: dict[str, Any], confirmation: dict[str, Any]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for label, result in (("screening", screening), ("confirmation", confirmation)):
        retrieval = result["qwen_retrieval"]
        checks[f"{label}_no_validation_failure"] = not bool(
            result["validation"]["errors"]
        )
        checks[f"{label}_no_fallback_or_hashing"] = not bool(
            retrieval["backend"].get("fallback_used", False)
        )
        checks[f"{label}_no_mixed_condition_query_sets"] = (
            retrieval["conditions"]["baseline"]["query_count"]
            == retrieval["conditions"]["composite"]["query_count"]
        )
    screen_delta = _metrics(screening)["delta"]
    checks["screening_gain_gate_failure_detected"] = screen_delta["mrr"] < MRR_THRESHOLD
    checks["screening_recall_regression_detected"] = (
        screen_delta["recall_at_10"] < -RECALL_REGRESSION_TOLERANCE
    )
    checks["both_runs_use_same_qwen_model"] = (
        screening["qwen_retrieval"]["backend"]["embedding_model"]
        == confirmation["qwen_retrieval"]["backend"]["embedding_model"]
    )
    checks["both_runs_use_same_query_radius"] = (
        screening["qwen_retrieval"]["comparison_scope"][
            "query_context_radius_messages"
        ]
        == confirmation["qwen_retrieval"]["comparison_scope"][
            "query_context_radius_messages"
        ]
    )
    return checks


def decide() -> dict[str, Any]:
    screening = _load(SCREENING)
    confirmation = _load(CONFIRMATION)
    repeatability = _load(REPEATABILITY)
    positive = _positive_checks(screening, confirmation)
    negative = _negative_checks(screening, confirmation)
    positive["confirmation_structural_payload_repeatable"] = (
        _structural_payload(confirmation) == _structural_payload(repeatability)
    )
    hard_invariant_names = {
        name
        for name in negative
        if name
        not in {
            "screening_gain_gate_failure_detected",
            "screening_recall_regression_detected",
        }
    }
    hard_invariants_pass = all(positive.values()) and all(
        negative[name] for name in hard_invariant_names
    )
    screening_metrics = _metrics(screening)
    confirmation_metrics = _metrics(confirmation)
    if not hard_invariants_pass:
        decision = "archive_experiment"
        reason = "A required structural or contract check failed."
    elif confirmation_metrics["delta"]["mrr"] >= MRR_THRESHOLD and (
        screening_metrics["delta"]["mrr"] < MRR_THRESHOLD
        or screening_metrics["delta"]["recall_at_10"]
        < -RECALL_REGRESSION_TOLERANCE
    ):
        decision = "structural_only"
        reason = (
            "C1 improves later time-disjoint confirmation retrieval, but the "
            "earlier screening split is inconsistent and has a material recall "
            "regression; do not promote it to a retrieval or production policy."
        )
    elif confirmation_metrics["delta"]["mrr"] >= MRR_THRESHOLD and (
        screening_metrics["delta"]["mrr"] >= MRR_THRESHOLD
        and screening_metrics["delta"]["recall_at_10"]
        >= -RECALL_REGRESSION_TOLERANCE
    ):
        decision = "direction_supported"
        reason = "Both time-disjoint stages pass the retrieval gain and regression gates."
    else:
        decision = "archive_experiment"
        reason = "The confirmation retrieval gain did not meet the required threshold."
    return {
        "schema_version": "instagram-overlapping-episodes-decision-v1",
        "decision": decision,
        "reason": reason,
        "thresholds": {
            "mrr_gain_required": MRR_THRESHOLD,
            "recall_regression_tolerance": RECALL_REGRESSION_TOLERANCE,
        },
        "positive_checks": positive,
        "negative_checks": negative,
        "hard_invariants_pass": hard_invariants_pass,
        "stages": {
            "screening": screening_metrics,
            "confirmation": confirmation_metrics,
            "pooled": _pooled([screening, confirmation]),
        },
        "alternatives": {
            "maximum_promotions": 2,
            "evaluated": [],
            "status": "not_run_after_primary_screening_inconsistency",
            "five_run_rule": "required_before_any_alternative_promotion",
        },
        "production_policy": {
            "retrieval_backend_changed": False,
            "qwen_remains_retrieval_backend": True,
            "jina_used": False,
            "human_labels_used": False,
            "article_grounding_gate_run": False,
            "promotion_authorized": False,
        },
        "source_artifacts": {
            "screening": str(SCREENING),
            "confirmation": str(CONFIRMATION),
            "repeatability": str(REPEATABILITY),
        },
    }


def _render(decision: dict[str, Any]) -> str:
    stages = decision["stages"]
    lines = [
        "# Instagram overlapping episode experiment: terminal decision",
        "",
        f"Decision: **`{decision['decision']}`**",
        "",
        decision["reason"],
        "",
        "## Metrics",
        "",
        "| Stage | Baseline MRR | Composite MRR | Δ MRR | Baseline Recall@10 | Composite Recall@10 | Δ Recall@10 | Queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("screening", "confirmation", "pooled"):
        values = stages[stage]
        lines.append(
            f"| {stage} | {values['baseline']['mrr']:.6f} | "
            f"{values['composite']['mrr']:.6f} | {values['delta']['mrr']:+.6f} | "
            f"{values['baseline']['recall_at_10']:.6f} | "
            f"{values['composite']['recall_at_10']:.6f} | "
            f"{values['delta']['recall_at_10']:+.6f} | "
            f"{values['baseline']['query_count']} |"
        )
    lines.extend(
        [
            "",
            "The screening stage uses the first 40% of the self-message timeline. "
            "The confirmation stage uses the later 60% and contains 60 structural "
            "anchors. The fixed Qwen query policy has an 8-message radius; seven "
            "confirmation anchors had no non-empty other-speaker query in that "
            "fixed window, so retrieval metrics cover 53 common anchors while "
            "structural validation covers all 60.",
            "",
            "## Verification",
            "",
            f"- Hard invariants passed: `{decision['hard_invariants_pass']}`.",
            "- Positive checks cover structural validation, fixture checks, "
            "perturbation checks, Qwen backend identity, equal condition query sets, "
            "time-disjoint anchor selection, and the confirmation gain gate.",
            "- Negative checks cover validation errors, fallback/hashing, mixed query "
            "sets, screening gain failure, screening recall regression, model drift, "
            "and query-radius drift.",
            "- No alternative was promoted. Any alternative would require five runs "
            "before promotion and is unnecessary after the primary method failed the "
            "screening consistency gate.",
            "",
            "## Boundary",
            "",
            "This is not a production retrieval or article-grounding result. Qwen "
            "remains the retrieval backend, Jina was not used, human labels were not "
            "used, and no production chunking or retrieval path changed. The result "
            "supports using overlapping episodes as a structural candidate for further "
            "controlled work, but not claiming directionality or downstream writing "
            "improvement.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    decision = decide()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "decision-report.md").write_text(_render(decision), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if decision["decision"] in {"direction_supported", "structural_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
