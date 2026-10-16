"""Evaluate paired native-vs-Jina confirmation metrics and gates."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/j1-qwen-jina-confirmation-v1/evaluation"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _metric(ranks: list[int], cutoff: int) -> float:
    return _mean([1.0 if rank <= cutoff else 0.0 for rank in ranks])


def _mrr(ranks: list[int]) -> float:
    return _mean([1.0 / rank if rank <= 50 else 0.0 for rank in ranks])


def _ndcg(ranks: list[int]) -> float:
    return _mean([1.0 / math.log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks])


def _bootstrap(values: list[float], *, seed: int, resamples: int = 10000) -> dict[str, float]:
    if not values:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}
    rng = random.Random(seed)
    samples = [
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(resamples)
    ]
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return {"lower": round(lower, 6), "upper": round(upper, 6), "mean": round(_mean(values), 6)}


def _mcnemar(rescues: int, damages: int) -> float:
    total = rescues + damages
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(0, min(rescues, damages) + 1)) / 2**total
    return min(1.0, 2.0 * tail)


def _transition(native: int, jina: int) -> str:
    if native > 10 and jina <= 5:
        return "strong_rescue"
    if native > 10 and jina <= 10:
        return "rescue"
    if native <= 10 and jina < native:
        return "internal_improvement"
    if native == jina:
        return "unchanged"
    if native <= 10 and jina <= 10:
        return "internal_worsening"
    if native <= 5 and jina > 10:
        return "catastrophic_damage"
    return "damage"


def _condition(result: dict[str, Any], name: str) -> dict[str, Any]:
    return result["conditions"][name]


def _evaluate_rows(native: dict[str, Any], jina: dict[str, Any]) -> dict[str, Any]:
    native_rows = {row["query_id"]: row for row in native["conditions"]["jina_selected"]["trace_rows"]}
    jina_rows = {row["query_id"]: row for row in jina["conditions"]["jina_selected"]["trace_rows"]}
    if set(native_rows) != set(jina_rows):
        raise ValueError("native and Jina traces do not contain the same query IDs")
    pairs = []
    for query_id in sorted(native_rows):
        left = native_rows[query_id]
        right = jina_rows[query_id]
        pairs.append(
            {
                "query_id": query_id,
                "fold": left["fold"],
                "band": left["band"],
                "native_rank": left["native_rank"],
                "jina_rank": right["reranked_rank"],
                "transition": _transition(left["native_rank"], right["reranked_rank"]),
            }
        )
    native_ranks = [row["native_rank"] for row in pairs]
    jina_ranks = [row["jina_rank"] for row in pairs]
    values = {
        "recall_at_5": [float(j <= 5) - float(n <= 5) for n, j in zip(native_ranks, jina_ranks)],
        "recall_at_10": [float(j <= 10) - float(n <= 10) for n, j in zip(native_ranks, jina_ranks)],
        "mrr_at_50": [
            (1.0 / j if j <= 50 else 0.0) - (1.0 / n if n <= 50 else 0.0)
            for n, j in zip(native_ranks, jina_ranks)
        ],
        "ndcg_at_10": [
            (1.0 / math.log2(j + 1) if j <= 10 else 0.0)
            - (1.0 / math.log2(n + 1) if n <= 10 else 0.0)
            for n, j in zip(native_ranks, jina_ranks)
        ],
    }
    rescues = sum(row["native_rank"] > 10 and row["jina_rank"] <= 10 for row in pairs)
    damages = sum(row["native_rank"] <= 10 and row["jina_rank"] > 10 for row in pairs)
    native_metrics = {
        "recall_at_5": round(_metric(native_ranks, 5), 6),
        "recall_at_10": round(_metric(native_ranks, 10), 6),
        "recall_at_50": round(_metric(native_ranks, 50), 6),
        "mrr_at_50": round(_mrr(native_ranks), 6),
        "ndcg_at_10": round(_ndcg(native_ranks), 6),
    }
    jina_metrics = {
        "recall_at_5": round(_metric(jina_ranks, 5), 6),
        "recall_at_10": round(_metric(jina_ranks, 10), 6),
        "recall_at_50": round(_metric(jina_ranks, 50), 6),
        "mrr_at_50": round(_mrr(jina_ranks), 6),
        "ndcg_at_10": round(_ndcg(jina_ranks), 6),
    }
    intervals = {
        key: _bootstrap(values[key], seed=20260808 + index)
        for index, key in enumerate(values)
    }
    return {
        "query_count": len(pairs),
        "native_metrics": native_metrics,
        "jina_metrics": jina_metrics,
        "deltas": {key: round(jina_metrics[key] - native_metrics[key], 6) for key in native_metrics},
        "paired_bootstrap_95": intervals,
        "transitions": {
            "rescues_at_10": rescues,
            "damages_at_10": damages,
            "net_rescue_at_10": rescues - damages,
            "rescue_damage_mcnemar_p": round(_mcnemar(rescues, damages), 6),
            "counts": {
                label: sum(row["transition"] == label for row in pairs)
                for label in (
                    "strong_rescue",
                    "rescue",
                    "internal_improvement",
                    "unchanged",
                    "internal_worsening",
                    "damage",
                    "catastrophic_damage",
                )
            },
        },
        "pairs": pairs,
    }


def _groups(evaluation: dict[str, Any], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in evaluation["pairs"]:
        groups.setdefault(str(row[field]), []).append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        native = [row["native_rank"] for row in rows]
        jina = [row["jina_rank"] for row in rows]
        result[name] = {
            "query_count": len(rows),
            "delta_recall_at_5": round(_metric(jina, 5) - _metric(native, 5), 6),
            "delta_recall_at_10": round(_metric(jina, 10) - _metric(native, 10), 6),
            "delta_mrr_at_50": round(_mrr(jina) - _mrr(native), 6),
            "delta_ndcg_at_10": round(_ndcg(jina) - _ndcg(native), 6),
        }
    return result


def evaluate(native_path: Path, jina_path: Path, *, substrate_hash: str | None = None) -> dict[str, Any]:
    native = _load(native_path)
    jina = _load(jina_path)
    if native.get("substrate_sha256") != jina.get("substrate_sha256"):
        raise ValueError("native and Jina results use different substrates")
    if substrate_hash is not None and native.get("substrate_sha256") != substrate_hash:
        raise ValueError("results do not match the frozen confirmation substrate")
    evaluation = _evaluate_rows(native, jina)
    evaluation["subgroups"] = {
        "fold": _groups(evaluation, "fold"),
        "band": _groups(evaluation, "band"),
    }
    delta10 = evaluation["deltas"]["recall_at_10"]
    delta5 = evaluation["deltas"]["recall_at_5"]
    lower10 = evaluation["paired_bootstrap_95"]["recall_at_10"]["lower"]
    no_large_subgroup_regression = all(
        values["delta_recall_at_10"] >= -0.05
        for values in evaluation["subgroups"]["band"].values()
    )
    positive_secondary = (
        evaluation["deltas"]["mrr_at_50"] > 0
        and evaluation["deltas"]["ndcg_at_10"] > 0
    )
    strong = (
        delta10 >= 0.07
        and delta5 >= 0.05
        and lower10 > 0
        and evaluation["transitions"]["rescues_at_10"] > evaluation["transitions"]["damages_at_10"]
        and positive_secondary
        and no_large_subgroup_regression
        and evaluation["transitions"]["counts"]["catastrophic_damage"] == 0
    )
    moderate = (
        0.04 <= delta10 < 0.07
        and delta5 >= 0
        and evaluation["transitions"]["rescues_at_10"] > evaluation["transitions"]["damages_at_10"]
        and positive_secondary
        and no_large_subgroup_regression
    )
    evaluation["decision"] = "confirmed" if strong else "inconclusive" if moderate else "not_confirmed"
    evaluation["gate_checks"] = {
        "strong": strong,
        "moderate": moderate,
        "lower_ci_recall_at_10_positive": lower10 > 0,
        "no_large_band_regression": no_large_subgroup_regression,
    }
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--jina", required=True)
    parser.add_argument("--substrate-hash")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = evaluate(Path(args.native), Path(args.jina), substrate_hash=args.substrate_hash)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "confirmation-evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
