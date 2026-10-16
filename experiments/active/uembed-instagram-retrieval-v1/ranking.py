"""Shared ranking primitives for the UEmbed Instagram experiment."""

from __future__ import annotations

import math
from typing import Any, Sequence

RRF_K = 60


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def rrf(rank_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion over ordered id lists; deterministic tie-break by id."""

    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for position, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=lambda item_id: (-scores[item_id], item_id))


def metrics(ranks: Sequence[int]) -> dict[str, float | int]:
    if not ranks:
        return {
            "query_count": 0,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
        }
    count = len(ranks)
    return {
        "query_count": count,
        "mrr": round(sum(1.0 / r for r in ranks) / count, 6),
        "ndcg_at_10": round(
            sum(1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in ranks) / count, 6
        ),
        "recall_at_10": round(sum(r <= 10 for r in ranks) / count, 6),
        "recall_at_20": round(sum(r <= 20 for r in ranks) / count, 6),
        "recall_at_50": round(sum(r <= 50 for r in ranks) / count, 6),
    }


def grouped_metrics(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(int(row["rank"]))
    return {name: metrics(values) for name, values in sorted(groups.items())}
