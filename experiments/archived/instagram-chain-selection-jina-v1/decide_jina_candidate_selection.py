"""Apply the bounded decision gates for the Jina candidate selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


MIN_MRR_GAIN_VS_C1 = 0.03
MIN_POSITIVE_FOLDS = 4
MIN_RECALL_DELTA_VS_C1 = -0.01
MAX_BAND_MRR_REGRESSION_VS_C1 = -0.01


def _delta(left: float, right: float) -> float:
    return round(left - right, 6)


def decide(result: dict[str, Any]) -> dict[str, Any]:
    structural = result.get("structural", {})
    contract = result.get("jina_contract", {})
    selection_contract = result.get("selection_contract", {})
    qwen = result.get("qwen_retrieval", {})
    qwen_backend = qwen.get("backend", {})
    conditions = qwen.get("conditions", {})
    baseline = conditions.get("baseline", {})
    c1 = conditions.get("c1_full", {})
    jina = conditions.get("jina_selected", {})

    positive_folds = {
        fold: _delta(jina["by_fold"][fold]["mrr"], c1["by_fold"][fold]["mrr"])
        for fold in c1.get("by_fold", {})
        if fold in jina.get("by_fold", {})
    }
    band_deltas = {
        band: _delta(jina["by_band"][band]["mrr"], c1["by_band"][band]["mrr"])
        for band in c1.get("by_band", {})
        if band in jina.get("by_band", {})
    }

    positive = {
        "structural_checks_passed": structural.get("passed") is True,
        "jina_mrr_gain_vs_c1_at_least_0_03": _delta(jina.get("mrr", 0.0), c1.get("mrr", 0.0))
        >= MIN_MRR_GAIN_VS_C1,
        "jina_recall_not_worse_than_c1": _delta(
            jina.get("recall_at_10", 0.0), c1.get("recall_at_10", 0.0)
        )
        >= MIN_RECALL_DELTA_VS_C1,
        "positive_mrr_folds_at_least_4_of_5": sum(value > 0 for value in positive_folds.values())
        >= MIN_POSITIVE_FOLDS,
        "median_fold_mrr_gain_positive": bool(positive_folds)
        and median(positive_folds.values()) > 0,
        "no_band_mrr_regression_worse_than_0_01": all(
            value >= MAX_BAND_MRR_REGRESSION_VS_C1 for value in band_deltas.values()
        ),
    }

    negative = {
        "jina_text_matching_task": contract.get("task") == "text-matching"
        and contract.get("task_type") == "text-matching",
        "jina_has_no_query_prompt": contract.get("prompt_name") is None,
        "jina_batch_size_one": contract.get("batch_size") == 1,
        "jina_metal_without_fallback": contract.get("backend_device") == "metal"
        and contract.get("fallback_used") is False,
        "jina_separate_vector_space": contract.get("vector_space")
        == "jina-v5-text-matching-mlx",
        "jina_did_not_receive_qwen_or_article_inputs": selection_contract.get(
            "qwen_query_used"
        )
        is False
        and selection_contract.get("article_prompt_used") is False,
        "qwen_retrieval_contract_preserved": qwen_backend.get("embedding_backend") == "qwen3"
        and qwen_backend.get("embedding_model") == "Qwen/Qwen3-Embedding-0.6B"
        and qwen_backend.get("fallback_used") is False,
        "same_query_count_across_conditions": {
            baseline.get("query_count"),
            c1.get("query_count"),
            jina.get("query_count"),
        }
        == {60},
        "symmetry_exact": result.get("jina_symmetry", {}).get(
            "reverse_order_max_abs_difference"
        )
        == 0.0,
    }

    if not all(negative.values()):
        decision = "archive_experiment"
    elif all(positive.values()):
        decision = "direction_supported"
    else:
        decision = "structural_only"

    return {
        "schema_version": "instagram-jina-candidate-selection-decision-v1",
        "decision": decision,
        "article_holdout_authorized": decision == "direction_supported",
        "thresholds": {
            "min_mrr_gain_vs_c1": MIN_MRR_GAIN_VS_C1,
            "min_positive_folds": MIN_POSITIVE_FOLDS,
            "min_recall_delta_vs_c1": MIN_RECALL_DELTA_VS_C1,
            "max_band_mrr_regression_vs_c1": MAX_BAND_MRR_REGRESSION_VS_C1,
        },
        "positive_checks": positive,
        "negative_checks": negative,
        "metrics": {
            "baseline": {
                key: baseline.get(key) for key in ("mrr", "recall_at_10", "ndcg_at_10")
            },
            "c1_full": {key: c1.get(key) for key in ("mrr", "recall_at_10", "ndcg_at_10")},
            "jina_selected": {
                key: jina.get(key) for key in ("mrr", "recall_at_10", "ndcg_at_10")
            },
            "jina_delta_vs_c1": {
                "mrr": _delta(jina.get("mrr", 0.0), c1.get("mrr", 0.0)),
                "recall_at_10": _delta(
                    jina.get("recall_at_10", 0.0), c1.get("recall_at_10", 0.0)
                ),
                "ndcg_at_10": _delta(
                    jina.get("ndcg_at_10", 0.0), c1.get("ndcg_at_10", 0.0)
                ),
            },
            "fold_mrr_delta_vs_c1": positive_folds,
            "band_mrr_delta_vs_c1": band_deltas,
        },
    }


def render(decision: dict[str, Any]) -> str:
    metrics = decision["metrics"]
    lines = [
        "# Instagram Jina candidate-selection decision",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        "The comparison is Jina local text matching for candidate selection followed by unchanged Qwen retrieval.",
        "Jina is not a retrieval model in this experiment.",
        "",
        "| Condition | MRR | Recall@10 | nDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("baseline", "c1_full", "jina_selected"):
        values = metrics[name]
        lines.append(
            f"| {name} | {values['mrr']:.6f} | {values['recall_at_10']:.6f} | {values['ndcg_at_10']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Jina delta versus C1: MRR `{metrics['jina_delta_vs_c1']['mrr']:+.6f}`, "
            f"Recall@10 `{metrics['jina_delta_vs_c1']['recall_at_10']:+.6f}`, "
            f"nDCG@10 `{metrics['jina_delta_vs_c1']['ndcg_at_10']:+.6f}`.",
            "",
            "## Positive checks",
            "",
        ]
    )
    lines.extend(f"- `{key}`: `{value}`" for key, value in decision["positive_checks"].items())
    lines.extend(["", "## Negative checks", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in decision["negative_checks"].items())
    lines.extend(
        [
            "",
            "The aggregate gain is not sufficient for promotion when fold and band stability gates fail.",
            "Article holdout is authorized only by `direction_supported`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/generated/instagram-jina-candidate-selection-v1/jina-selection-results.json",
    )
    parser.add_argument(
        "--output", default="data/generated/instagram-jina-candidate-selection-v1-decision"
    )
    args = parser.parse_args()
    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    decision = decide(result)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (output / "decision-report.md").write_text(render(decision), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
