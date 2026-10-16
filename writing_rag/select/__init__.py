"""Jina text-matching and selection over frozen Qwen pools only."""

from __future__ import annotations

from .jina_text_matching_mlx import (
    JinaTextMatchingContractError,
    JinaTextMatchingMLX,
    metal_probe,
    resolve_snapshot,
)
from .jina_text_selection import (
    TextSelectionContractError,
    select_from_qwen_pool,
    validate_jina_contract,
)

__all__ = [
    "JinaTextMatchingContractError",
    "JinaTextMatchingMLX",
    "TextSelectionContractError",
    "metal_probe",
    "resolve_snapshot",
    "select_from_qwen_pool",
    "validate_jina_contract",
]
