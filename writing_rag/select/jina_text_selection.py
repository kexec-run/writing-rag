"""Contract helpers for bounded Jina text selection after Qwen retrieval.

This module deliberately does not load a model or search a corpus. The caller
must provide a fixed Qwen candidate pool and Jina-native scores for those same
candidate IDs. That keeps the future Jina ablation separate from production
retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class TextSelectionContractError(ValueError):
    """Raised when a selector attempts to escape the Qwen candidate boundary."""


def _candidate_id(item: Mapping[str, Any]) -> str:
    document = item.get("document")
    if isinstance(document, Mapping):
        value = document.get("document_id")
    else:
        value = item.get("document_id")
    if not isinstance(value, str) or not value:
        raise TextSelectionContractError("Qwen candidate is missing document_id")
    return value


def validate_jina_contract(metadata: Mapping[str, Any]) -> None:
    """Require the isolated symmetric Jina text-matching contract."""

    required = {
        "embedding_backend": "jina-v5-mlx",
        "task": "text-matching",
        "task_type": "text-matching",
        "prompt_name": None,
        "vector_space": "jina-v5-text-matching-mlx",
        "fallback_used": False,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise TextSelectionContractError(
                f"Jina text-selection contract mismatch for {key}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )


def select_from_qwen_pool(
    qwen_candidates: Sequence[Mapping[str, Any]],
    jina_scores: Mapping[str, float],
    *,
    top_k: int | None = None,
    jina_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Reorder a Qwen pool using scores for the same IDs only.

    The returned records retain the complete Qwen result and add Jina-native
    selection fields. No model is loaded here and no new corpus candidate can
    enter the result.
    """

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if jina_metadata is not None:
        validate_jina_contract(jina_metadata)

    pool = [deepcopy(dict(item)) for item in qwen_candidates]
    pool_ids = [_candidate_id(item) for item in pool]
    if len(pool_ids) != len(set(pool_ids)):
        raise TextSelectionContractError("Qwen candidate pool contains duplicate IDs")

    unknown_ids = set(jina_scores) - set(pool_ids)
    if unknown_ids:
        raise TextSelectionContractError(
            "Jina scores contain IDs outside the Qwen candidate pool: "
            + ", ".join(sorted(unknown_ids)[:3])
        )
    missing_ids = set(pool_ids) - set(jina_scores)
    if missing_ids:
        raise TextSelectionContractError(
            "Jina scores are incomplete for the Qwen candidate pool: "
            + ", ".join(sorted(missing_ids)[:3])
        )

    original_order = {document_id: index for index, document_id in enumerate(pool_ids)}
    ranked = sorted(
        pool,
        key=lambda item: (
            -float(jina_scores[_candidate_id(item)]),
            original_order[_candidate_id(item)],
        ),
    )
    if top_k is not None:
        ranked = ranked[:top_k]

    selected: list[dict[str, Any]] = []
    for selection_rank, item in enumerate(ranked, start=1):
        document_id = _candidate_id(item)
        item["qwen_candidate_rank"] = original_order[document_id] + 1
        item["jina_selection_score"] = float(jina_scores[document_id])
        item["jina_selection_rank"] = selection_rank
        item["selection_source"] = "qwen_candidate_pool"
        item["selection_backend"] = "jina-v5-mlx"
        item["selection_task"] = "text-matching"
        selected.append(item)
    return selected
