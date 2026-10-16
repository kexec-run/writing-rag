"""Pure validation tests for the isolated zembed retrieval experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/active/zembed-retrieval-validity-v1/run_retrieval_experiment.py"
spec = importlib.util.spec_from_file_location("zembed_runner", RUNNER)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_card_example_requires_expected_order_and_error_bounds() -> None:
    class FakeBackend:
        def encode_queries(self, texts, *, batch_size):
            return [[1.0, 0.0, 0.0]]

        def encode_documents(self, texts, *, batch_size):
            return [[0.7525, 0.6586, 0.0], [0.567, 0.8237, 0.0], [0.6835, 0.7299, 0.0]]

    result = runner.model_card_check(FakeBackend())
    assert result["passed"] is True


def test_card_check_rejects_wrong_order() -> None:
    class FakeBackend:
        def encode_queries(self, texts, *, batch_size):
            return [[1.0, 0.0]]

        def encode_documents(self, texts, *, batch_size):
            return [[0.5, 0.8660254], [0.9, 0.4358899], [0.4, 0.9165151]]

    assert runner.model_card_check(FakeBackend())["passed"] is False


def test_comparison_is_rank_and_label_based() -> None:
    aggregate = {"10": {"mrr": 0.8, "ndcg_at_10": 0.8, "recall_at_10": 0.8, "precision_at_10": 0.1}, "50": {}, "100": {}}
    q0_row = {"id": "q", "evidence_absent": False, "metrics": aggregate, "hard_negative_ranks": {}}
    z_row = {"id": "q", "evidence_absent": False, "metrics": aggregate, "hard_negative_ranks": {}}
    q0 = {"modes": {mode: {"queries": [q0_row], "aggregate": aggregate} for mode in runner.MODES}}
    zembed = {"modes": {mode: {"queries": [z_row], "aggregate": aggregate} for mode in runner.MODES}}
    comparison = runner.compare_to_q0(zembed, q0)
    assert comparison["dense_only"]["deltas_at_10"]["mrr"] == 0
    assert "cosine_delta" not in str(comparison)


def test_document_resume_reuses_validated_batches_and_rejects_identity_change(tmp_path: Path) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def contract_metadata(self):
            return {
                "backend": "fake",
                "model_name": "fake-model",
                "resolved_model": "/fake/model",
                "dimension": 2,
                "snapshot_files": {"model.safetensors": "fake"},
            }

        def encode_documents(self, texts, *, batch_size):
            self.calls += 1
            return [[float(index), 1.0] for index, _ in enumerate(texts, 1)]

    documents = [{"document_id": "a", "text": "alpha"}, {"document_id": "b", "text": "beta"}]
    backend = FakeBackend()
    first, metadata = runner.encode_documents_resumable(backend, documents, batch_size=1, resume_dir=tmp_path)
    assert len(first) == 2
    assert metadata["batch_count"] == 2
    assert backend.calls == 2

    second, resumed = runner.encode_documents_resumable(backend, documents, batch_size=1, resume_dir=tmp_path)
    assert second == first
    assert resumed["resumed"] is True
    assert backend.calls == 2

    changed = [{"document_id": "a", "text": "changed"}, {"document_id": "b", "text": "beta"}]
    try:
        runner.encode_documents_resumable(backend, changed, batch_size=1, resume_dir=tmp_path)
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("changed document substrate must fail closed")
