"""Scarcity-first variant of select_five_fold_anchors.

Identical rules (bands, query availability, minimum_separation, spacing);
only the cell processing ORDER changes: rarest eligible pool first, so scarce
cells claim spaced slots before abundant cells fragment the timeline.
"""

from __future__ import annotations

from typing import Any

BANDS = ("p90_band", "p95_band", "p99_band")


def _choose_spaced(candidates, count, minimum_separation):
    if len(candidates) < count:
        return []
    chosen = []
    target_positions = [
        round(index * (len(candidates) - 1) / max(1, count - 1))
        for index in range(count)
    ]
    for target in target_positions:
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (abs(candidates.index(item) - target), item[0]),
        )
        selected = next(
            (
                item
                for item in ordered_candidates
                if all(abs(item[0] - prev[0]) >= minimum_separation for prev in chosen)
            ),
            None,
        )
        if selected is None:
            return []
        chosen.append(selected)
    return chosen


def select_scarcity_first(
    ordered: list[dict[str, Any]],
    *,
    thresholds: dict[str, int],
    excluded_ids: set[str],
    fold_count: int = 5,
    per_band_fold: int = 4,
    query_radius: int = 8,
    minimum_separation: int = 40,
    other_query,
    count_tokens,
    assign_band,
    fold_for_index,
):
    candidates: dict[tuple[int, str], list[tuple[int, dict[str, Any]]]] = {
        (fold, band): []
        for fold in range(fold_count)
        for band in BANDS
    }
    for index, message in enumerate(ordered):
        if message.get("speaker") != "self":
            continue
        message_id = str(message.get("message_id"))
        if message_id in excluded_ids:
            continue
        tokens = count_tokens(str(message.get("content", "")))
        band = assign_band(tokens, thresholds)
        if band is None or not other_query(ordered, index, query_radius):
            continue
        fold = fold_for_index(index, len(ordered), fold_count)
        candidates[(fold, band)].append((index, message))

    cells = sorted(candidates, key=lambda cell: (len(candidates[cell]), cell))
    selected: dict[str, list[dict[str, Any]]] = {str(f): [] for f in range(fold_count)}
    selected_indices: list[int] = []
    selection_errors: list[str] = []
    eligible_counts: dict[str, int] = {}
    for fold, band in cells:
        key = f"{fold}_{band}"
        band_candidates = candidates[(fold, band)]
        eligible_counts[key] = len(band_candidates)
        free = [
            item
            for item in band_candidates
            if all(abs(item[0] - prev) >= minimum_separation for prev in selected_indices)
        ]
        choices = _choose_spaced(free, per_band_fold, minimum_separation)
        if len(choices) != per_band_fold:
            selection_errors.append(f"insufficient_{key}")
            continue
        for index, message in choices:
            selected_indices.append(index)
            selected[str(fold)].append(
                {
                    "message_id": str(message["message_id"]),
                    "timestamp_ms": int(message.get("timestamp_ms", 0)),
                    "token_count": count_tokens(str(message.get("content", ""))),
                    "band": band,
                    "fold": fold,
                    "query_context_radius": query_radius,
                    "query_available": True,
                }
            )
    for fold in selected:
        selected[fold].sort(
            key=lambda a: (a["timestamp_ms"], a["message_id"])
        )
    metadata = {
        "fold_count": fold_count,
        "per_band_fold": per_band_fold,
        "minimum_separation_messages": minimum_separation,
        "query_radius_messages": query_radius,
        "eligible_counts": eligible_counts,
        "selection_errors": selection_errors,
        "selected_count": sum(len(v) for v in selected.values()),
        "cell_order": [f"{f}_{b}" for f, b in cells],
    }
    return selected, metadata
