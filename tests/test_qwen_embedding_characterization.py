"""Tests for the isolated Q0/Q1 embedding-characterization harness."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("numpy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qwen_embedding_characterization import (  # noqa: E402
    CATEGORIES,
    CaseValidationError,
    candidate_registry,
    category_summary,
    combine_case_suites,
    analyze_vector,
    role_metrics,
    run_characterization,
    validate_case_suite,
    validate_source_spans,
    write_artifacts,
)


def synthetic_case(
    *,
    case_id: str = "direct-1",
    base_case_id: str = "direct",
    category: str = "direct_evidence",
    construction: str = "controlled_synthetic",
) -> dict:
    candidates = [
        {"candidate_id": f"{case_id}-positive", "text": "The server exposed the index to the public internet.", "role": "positive"},
        {"candidate_id": f"{case_id}-hard", "text": "The server restarted after a scheduled update.", "role": "hard_negative"},
        {"candidate_id": f"{case_id}-easy", "text": "The garden needed more water.", "role": "easy_negative"},
    ]
    if category == "evidence_absent":
        candidates[0]["role"] = "hard_negative"
    if construction != "controlled_synthetic":
        for candidate in candidates:
            candidate["source_spans"] = [{"document_id": "fixture-doc", "text": candidate["text"]}]
    return {
        "case_id": case_id,
        "base_case_id": base_case_id,
        "category": category,
        "construction": construction,
        "query": "Why was the index available publicly?",
        "review_notes": "Synthetic fixture used only for unit tests.",
        "candidates": candidates,
    }


def suite(name: str, cases: list[dict]) -> dict:
    return {
        "version": "qwen-embedding-characterization-v1",
        "suite": name,
        "freeze_metadata": {"frozen_at": "2026-08-04T00:00:00Z", "reviewer": "test"},
        "cases": cases,
    }


class FakeBackend:
    model_name = "fake-qwen"
    resolved_model = "/tmp/fake-snapshot"
    pooling_config_sha256 = "pooling"
    query_prompt = "Instruct: retrieve evidence\nQuery:"
    device = "cpu"
    last_output_dtype = "float32"

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls: list[str] = []

    def encode_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        self.document_calls += 1
        return [[1.0, 0.0] if "public internet" in text else [0.0, 1.0] if "scheduled" in text else [-1.0, 0.0] for text in texts]

    def encode_queries(self, texts: list[str], *, batch_size: int, policy: str) -> list[list[float]]:
        self.query_calls.append(policy)
        return [[1.0, 0.0] if policy == "plain" else [0.6, 0.8] for _text in texts]


class QwenEmbeddingCharacterizationTests(unittest.TestCase):
    def test_corpus_case_requires_exact_source_spans(self) -> None:
        payload = suite("diagnostic", [synthetic_case(construction="corpus_exact")])
        payload["cases"][0]["candidates"][0].pop("source_spans")
        with self.assertRaisesRegex(CaseValidationError, "source_spans"):
            validate_case_suite(payload)

    def test_evidence_absent_cannot_contain_positive(self) -> None:
        payload = suite("diagnostic", [synthetic_case(category="evidence_absent")])
        payload["cases"][0]["candidates"][0]["role"] = "positive"
        with self.assertRaisesRegex(CaseValidationError, "cannot have a positive"):
            validate_case_suite(payload)

    def test_required_category_gate_is_explicit(self) -> None:
        payload = suite("diagnostic", [synthetic_case()])
        with self.assertRaisesRegex(CaseValidationError, "missing required categories"):
            validate_case_suite(payload, require_all_categories=True)
        all_cases = [synthetic_case(case_id=f"case-{index}", base_case_id=f"base-{index}", category=category) for index, category in enumerate(CATEGORIES)]
        validate_case_suite(suite("diagnostic", all_cases), require_all_categories=True)

    def test_duplicate_candidate_id_with_changed_provenance_fails(self) -> None:
        first = synthetic_case(case_id="one")
        second = synthetic_case(case_id="two")
        second["candidates"][0]["candidate_id"] = first["candidates"][0]["candidate_id"]
        second["candidates"][0]["text"] = "Different text"
        cases = combine_case_suites([validate_case_suite(suite("diagnostic", [first])), validate_case_suite(suite("confirmation", [second]))])
        with self.assertRaisesRegex(CaseValidationError, "inconsistent text or provenance"):
            candidate_registry(cases)

    def test_source_span_audit_checks_document_and_candidate_text(self) -> None:
        payload = validate_case_suite(suite("diagnostic", [synthetic_case(construction="corpus_trimmed")]))
        documents = [{"document_id": "fixture-doc", "text": "The server exposed the index to the public internet."}]
        with self.assertRaisesRegex(CaseValidationError, "source span is not present"):
            validate_source_spans([payload], documents)
        documents[0]["text"] = "The server exposed the index to the public internet. The server restarted after a scheduled update. The garden needed more water."
        validate_source_spans([payload], documents)

    def test_role_metrics_keep_best_and_worst_margin_distinct(self) -> None:
        case = synthetic_case()
        case["candidates"].append({"candidate_id": "second-positive", "text": "Another answer", "role": "positive"})
        ranking = [
            {"candidate_id": f"{case['case_id']}-positive", "rank": 1, "cosine": 0.9},
            {"candidate_id": f"{case['case_id']}-hard", "rank": 2, "cosine": 0.8},
            {"candidate_id": "second-positive", "rank": 3, "cosine": 0.7},
            {"candidate_id": f"{case['case_id']}-easy", "rank": 4, "cosine": 0.1},
        ]
        metrics = role_metrics(case, ranking)
        self.assertAlmostEqual(metrics["best_positive_margin"], 0.1)
        self.assertAlmostEqual(metrics["worst_positive_margin"], -0.1)
        self.assertEqual(metrics["hard_negatives_above_every_positive"], 0)
        self.assertEqual(metrics["hard_negatives_between_positive_extremes"], 1)

    def test_runner_encodes_passages_once_and_never_calls_retrieval(self) -> None:
        diagnostic = validate_case_suite(suite("diagnostic", [synthetic_case()]))
        confirmation = validate_case_suite(suite("confirmation", [synthetic_case(case_id="confirm-1", base_case_id="confirm")]))
        backend = FakeBackend()
        result = run_characterization([diagnostic, confirmation], backend=backend, batch_size=2)
        self.assertEqual(backend.document_calls, 1)
        self.assertEqual(backend.query_calls, ["plain", "qwen_card_query"])
        self.assertTrue(result["run_metadata"]["passages_encoded_once"])
        comparison = result["comparisons"][0]
        self.assertTrue(comparison["flags"]["hard_negative_promoted"])
        self.assertIn("global_rank", result["local_results"]["Q0_plain"]["policy_results"][0]["ranking"][0])
        candidate_flags = comparison["candidate_comparisons"][1]["hard_negative_flags"]
        self.assertIsNone(candidate_flags["hard_negative_entered_global_top_10"])
        self.assertEqual(result["run_metadata"]["raw_model_output_dtype"]["passages"], "float32")

    def test_float32_contract_rejects_zero_and_dimension_mismatch(self) -> None:
        record = analyze_vector([3.0, 4.0])
        self.assertEqual(record["analysis_vector_dtype"], "float32")
        with self.assertRaises(ValueError):
            analyze_vector([0.0, 0.0])

    def test_identical_text_merges_global_identity_and_keeps_aliases(self) -> None:
        first = synthetic_case(case_id="one")
        second = synthetic_case(case_id="two")
        second["candidates"][0]["text"] = first["candidates"][0]["text"]
        registry = candidate_registry(combine_case_suites([suite("diagnostic", [first]), suite("confirmation", [second])]))
        matching = [item for item in registry if item["text"] == first["candidates"][0]["text"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(set(matching[0]["candidate_ids"]), {"one-positive", "two-positive"})
        self.assertEqual(len(matching[0]["provenance_aliases"]), 2)

    def test_category_summary_requires_independent_base_cases(self) -> None:
        rows = [
            {
                "suite": "diagnostic",
                "category": "direct_evidence",
                "base_case_id": "same-base",
                "query_vector_cosine_q0_q1": 0.9,
                "delta": {"best_positive_margin": 0.01},
                "flags": {"positive_rank_improved": False, "positive_rank_regressed": False, "best_margin_improved": True, "best_margin_regressed": False, "hard_negative_promoted": False},
            }
            for _ in range(4)
        ]
        summary = category_summary(rows)[0]
        self.assertEqual(summary["independent_base_case_count"], 1)
        self.assertEqual(summary["conclusion_status"], "exploratory_insufficient_independent_cases")

    def test_write_artifacts_has_separate_policy_and_trace_outputs(self) -> None:
        diagnostic = validate_case_suite(suite("diagnostic", [synthetic_case()]))
        confirmation = validate_case_suite(suite("confirmation", [synthetic_case(case_id="confirm-1", base_case_id="confirm")]))
        result = run_characterization([diagnostic, confirmation], backend=FakeBackend(), batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            write_artifacts(result, [diagnostic, confirmation], output)
            self.assertTrue((output / "Q0_plain" / "local-results.json").is_file())
            self.assertTrue((output / "Q1_card" / "global-pool-results.json").is_file())
            self.assertTrue((output / "comparison" / "report.md").is_file())
            with self.assertRaises(FileExistsError):
                write_artifacts(result, [diagnostic, confirmation], output)


if __name__ == "__main__":
    unittest.main()
