"""Tests for corrected-pilot validation and repeatability contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys
import pytest

pytest.importorskip("numpy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_qwen_characterization_runs import compare_values  # noqa: E402
from qwen_embedding_characterization import analyze_vector  # noqa: E402
from validate_qwen_embedding_characterization import (  # noqa: E402
    ensure_fresh_output,
    phase_check,
    validate_run_metadata,
    validate_vector_records,
)


class QwenCharacterizationValidationTests(unittest.TestCase):
    def test_authorization_is_computed_from_positive_and_inverse(self) -> None:
        self.assertTrue(phase_check("pass", "works", True, True)["authorized_to_proceed"])
        self.assertFalse(phase_check("fail", "inverse missing", True, False)["authorized_to_proceed"])

    def test_vector_inverse_rejects_zero_norm(self) -> None:
        valid = {
            "vector": analyze_vector([3.0, 4.0])["vector"],
            "analysis_vector_dtype": "float32",
            "raw_vector_dtype": "float32",
            "raw_vector_norm": 5.0,
            "normalized_vector_norm": 1.0,
        }
        ok, errors = validate_vector_records([valid])
        self.assertTrue(ok, errors)
        invalid = {**valid, "vector": [0.0, 0.0]}
        ok, errors = validate_vector_records([invalid])
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_metadata_requires_source_hash(self) -> None:
        ok, missing = validate_run_metadata({"source_document_hash": "hash"})
        self.assertFalse(ok)
        self.assertIn("model_name", missing)

    def test_repeatability_comparison_uses_numeric_tolerance(self) -> None:
        self.assertEqual(compare_values({"score": 0.5}, {"score": 0.500000001}), [])
        self.assertTrue(compare_values({"score": 0.5}, {"score": 0.51}))

    def test_existing_output_is_not_redirected_or_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / "artifact.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_fresh_output(output)


if __name__ == "__main__":
    unittest.main()
