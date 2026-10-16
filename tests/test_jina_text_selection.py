import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jina_text_selection import (  # noqa: E402
    TextSelectionContractError,
    select_from_qwen_pool,
)


CONTRACT = {
    "embedding_backend": "jina-v5-mlx",
    "task": "text-matching",
    "task_type": "text-matching",
    "prompt_name": None,
    "vector_space": "jina-v5-text-matching-mlx",
    "fallback_used": False,
}


def pool():
    return [
        {"rank": 1, "score": 0.80, "document": {"document_id": "q1"}},
        {"rank": 2, "score": 0.70, "document": {"document_id": "q2"}},
        {"rank": 3, "score": 0.60, "document": {"document_id": "q3"}},
    ]


def test_selection_can_only_reorder_qwen_candidates():
    selected = select_from_qwen_pool(
        pool(), {"q1": 0.1, "q2": 0.9, "q3": 0.2}, top_k=2, jina_metadata=CONTRACT
    )

    assert [item["document"]["document_id"] for item in selected] == ["q2", "q3"]
    assert selected[0]["qwen_candidate_rank"] == 2
    assert selected[0]["selection_source"] == "qwen_candidate_pool"
    assert selected[0]["selection_task"] == "text-matching"


@pytest.mark.parametrize(
    "scores",
    [
        {"q1": 0.1, "q2": 0.9, "new": 0.8},
        {"q1": 0.1, "q2": 0.9},
    ],
)
def test_selection_rejects_unknown_or_missing_candidates(scores):
    with pytest.raises(TextSelectionContractError):
        select_from_qwen_pool(pool(), scores, jina_metadata=CONTRACT)


def test_selection_rejects_retrieval_task_metadata():
    bad_contract = {**CONTRACT, "task": "retrieval", "task_type": "retrieval"}
    with pytest.raises(TextSelectionContractError):
        select_from_qwen_pool(
            pool(), {"q1": 0.1, "q2": 0.9, "q3": 0.2}, jina_metadata=bad_contract
        )


def test_selection_rejects_prompt_role_or_fallback():
    for key, value in (("prompt_name", "query"), ("fallback_used", True)):
        bad_contract = {**CONTRACT, key: value}
        with pytest.raises(TextSelectionContractError):
            select_from_qwen_pool(
                pool(), {"q1": 0.1, "q2": 0.9, "q3": 0.2}, jina_metadata=bad_contract
            )
