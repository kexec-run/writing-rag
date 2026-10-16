"""Apply the predeclared gate to a fresh Instagram zembed confirmation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIRMATION = ROOT / "data/generated/zembed-instagram-confirmation-v1/confirmation-cases.json"
DEFAULT_VALIDATION = ROOT / "data/generated/zembed-instagram-confirmation-v1/validation/validation.json"
DEFAULT_OUTPUT = ROOT / "data/generated/zembed-instagram-confirmation-v1/decision"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric_delta(validation: dict[str, Any], metric: str) -> float:
    c1 = validation["conditions"]["c1_full"]
    return round(float(c1["zembed"][metric]) - float(c1["qwen"][metric]), 6)


def decide(confirmation: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    c1 = validation["conditions"]["c1_full"]
    qwen = c1["qwen"]
    zembed = c1["zembed"]
    comparison = c1["comparison"]["summary"]
    fold_deltas = {
        fold: round(
            float(zembed["by_fold"][fold]["mrr"])
            - float(qwen["by_fold"][fold]["mrr"]),
            6,
        )
        for fold in qwen["by_fold"]
    }
    band_deltas = {
        band: {
            metric: round(
                float(zembed["by_band"][band][metric])
                - float(qwen["by_band"][band][metric]),
                6,
            )
            for metric in ("mrr", "recall_at_10", "recall_at_50")
        }
        for band in qwen["by_band"]
    }
    positive = {
        "structural_passed": confirmation["structural"]["passed"],
        "exactly_60_fresh_anchors": confirmation["scope"]["anchor_count"] == 60,
        "five_folds": confirmation["scope"]["fold_count"] == 5,
        "same_queries_and_documents": validation["scope"]["same_documents_and_queries"],
        "zembed_model_contract": (
            validation["models"]["zembed"]["metadata"].get("model_name")
            == "lexrivera/zembed-1-embedding-mlx-6Bit"
            and validation["models"]["zembed"]["metadata"].get("dimension") == 2560
            and validation["models"]["zembed"]["batch_size"] == 1
        ),
        "mrr_gain_at_least_003": _metric_delta(validation, "mrr") >= 0.03,
        "ndcg_gain_at_least_003": _metric_delta(validation, "ndcg_at_10") >= 0.03,
        "recall10_guardrail": _metric_delta(validation, "recall_at_10") >= -0.01,
        "recall50_guardrail": _metric_delta(validation, "recall_at_50") >= -0.01,
        "four_of_five_folds_mrr_nonnegative": sum(delta >= 0 for delta in fold_deltas.values()) >= 4,
        "rescue_damage_margin_at_least_3": comparison["top10_rescues"] - comparison["top10_damages"] >= 3,
        "no_band_mrr_regression_over_001": all(values["mrr"] >= -0.01 for values in band_deltas.values()),
    }
    negative = {
        "no_p99_mrr_regression": band_deltas.get("p99_band", {}).get("mrr", -1.0) >= 0,
        "no_band_recall50_regression_over_005": all(values["recall_at_50"] >= -0.05 for values in band_deltas.values()),
        "no_fallback": validation["models"]["zembed"]["metadata"].get("fallback_used") is False,
        "no_jina_or_reranker": confirmation["method"]["no_jina"] and confirmation["method"]["no_reranker"],
    }
    status = (
        "direction_supported_for_instagram_c1"
        if all(positive.values()) and all(negative.values())
        else "not_confirmed"
    )
    return {
        "schema_version": "zembed-instagram-confirmation-decision-v1",
        "decision": status,
        "positive_checks": positive,
        "negative_checks": negative,
        "metrics": {
            "c1_qwen": {key: qwen[key] for key in ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_50")},
            "c1_zembed": {key: zembed[key] for key in ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_50")},
            "deltas": {key: _metric_delta(validation, key) for key in ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_50")},
            "fold_mrr_deltas": fold_deltas,
            "band_deltas": band_deltas,
            "transition_summary": comparison,
        },
    }


def _render(decision: dict[str, Any]) -> str:
    lines = [
        "# Instagram zembed confirmation decision",
        "",
        f"Decision: **`{decision['decision']}`**",
        "",
        "## C1 metrics",
        "",
        "| Model | MRR | nDCG@10 | Recall@10 | Recall@20 | Recall@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_key, label in (("c1_qwen", "Qwen"), ("c1_zembed", "zembed")):
        row = decision["metrics"][model_key]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(label, *(f"{row[key]:.6f}" for key in ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_50"))))
    lines.extend(["", "## Positive checks", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in decision["positive_checks"].items())
    lines.extend(["", "## Negative checks", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in decision["negative_checks"].items())
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default=str(DEFAULT_CONFIRMATION))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    decision = decide(_load(Path(args.confirmation)), _load(Path(args.validation)))
    output = Path(args.output)
    _write(output / "decision.json", decision)
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.md").write_text(_render(decision), encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "output": str(output)}, indent=2))
    return 0 if decision["decision"] == "direction_supported_for_instagram_c1" else 2


if __name__ == "__main__":
    raise SystemExit(main())
