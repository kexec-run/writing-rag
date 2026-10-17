from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/ranking_replay"
RERANKER_DIR = ROOT / "experiments/active/qwen-top50-reranker-v2"
REPEATABILITY_SCRIPT = ROOT / "experiments/active/j1-qwen-jina-confirmation-v1/verify_repeatability.py"
sys.path.insert(0, str(RERANKER_DIR))

import freeze_qwen_top50
import replay_frozen_rankings
import rerank_frozen_pools


def _load_builder():
    spec = importlib.util.spec_from_file_location("ranking_replay_fixture_builder", FIXTURES / "build_fixture.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load synthetic fixture builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


class FrozenRankingReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.inputs = cls.root / "inputs"
        BUILDER.build(FIXTURES / "fixture_spec.json", cls.inputs)
        cls.substrate_path = cls.inputs / "frozen-top50.json"
        cls.qwen_path = cls.inputs / "qwen-rankings.json"
        cls.jina_path = cls.inputs / "jina-rankings.json"
        cls.substrate = json.loads(cls.substrate_path.read_text(encoding="utf-8"))
        cls.qwen_input = json.loads(cls.qwen_path.read_text(encoding="utf-8"))
        cls.jina_input = json.loads(cls.jina_path.read_text(encoding="utf-8"))
        cls.expected = json.loads((FIXTURES / "expected_metrics.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _replay(self, ranking_input: dict, *, substrate: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            substrate_path = base / "substrate.json"
            rankings_path = base / "rankings.json"
            if substrate is None:
                substrate_path.write_bytes(self.substrate_path.read_bytes())
            else:
                substrate_path.write_text(json.dumps(substrate), encoding="utf-8")
            rankings_path.write_text(json.dumps(ranking_input), encoding="utf-8")
            return replay_frozen_rankings.replay(substrate_path, rankings_path)

    def _assert_metrics(self, condition_result: dict, expected: dict) -> None:
        native = condition_result["native_metrics"]
        reranked = condition_result["reranked_metrics"]
        for key in ("recall_at_5", "recall_at_10", "recall_at_50", "mrr_at_50", "ndcg_at_10"):
            self.assertAlmostEqual(native[key], self.expected["native"][key], places=6)
            self.assertAlmostEqual(reranked[key], expected[key], places=6)
        for band, metrics in expected["groups"].items():
            actual = condition_result["subgroups"]["band"][band]["reranked_metrics"]
            for key, value in metrics.items():
                self.assertAlmostEqual(actual[key], value, places=6)
        self.assertEqual(condition_result["transitions"]["10"]["rescues"], expected["rescues_at_10"])
        self.assertEqual(condition_result["transitions"]["10"]["damages"], expected["damages_at_10"])

    def test_six_query_metrics_and_subgroups_match_independent_expectations(self) -> None:
        qwen = self._replay(self.qwen_input)
        jina = self._replay(self.jina_input)
        self.assertEqual(qwen["execution_mode"], "synthetic_replay")
        self.assertIs(qwen["model_executed"], False)
        for condition in freeze_qwen_top50.CONDITIONS:
            self._assert_metrics(qwen["conditions"][condition], self.expected["qwen_order"])
            self._assert_metrics(jina["conditions"][condition], self.expected["jina_order"])

    def test_top50_recall_is_unchanged_and_missing_gold_cannot_be_rescued(self) -> None:
        for supplied, expected in ((self.qwen_input, "qwen_order"), (self.jina_input, "jina_order")):
            result = self._replay(supplied)
            for condition in result["conditions"].values():
                self.assertAlmostEqual(condition["native_metrics"]["recall_at_50"], 5 / 6, places=6)
                self.assertAlmostEqual(condition["reranked_metrics"]["recall_at_50"], 5 / 6, places=6)
                missing = next(row for row in condition["trace_rows"] if row["query_id"] == "synthetic-q06")
                self.assertEqual(missing["reranked_rank"], 51)
                self.assertFalse(missing["reranked_top10"])

    def test_duplicate_query_ids_in_frozen_input_fail(self) -> None:
        substrate = copy.deepcopy(self.substrate)
        rows = substrate["conditions"]["baseline"]["queries"]
        rows[-1]["query_id"] = rows[0]["query_id"]
        with self.assertRaisesRegex(ValueError, "duplicate query ID"):
            freeze_qwen_top50.validate_frozen(substrate)

    def test_declared_query_count_and_condition_sets_are_enforced(self) -> None:
        substrate = copy.deepcopy(self.substrate)
        substrate["conditions"]["baseline"]["query_count"] = 5
        with self.assertRaisesRegex(ValueError, "invalid condition"):
            freeze_qwen_top50.validate_frozen(substrate)
        substrate = copy.deepcopy(self.substrate)
        del substrate["conditions"]["c1_full"]
        with self.assertRaisesRegex(ValueError, "conditions are missing"):
            freeze_qwen_top50.validate_frozen(substrate)

    def test_altered_query_and_candidate_text_hashes_fail(self) -> None:
        for field in ("query_text", "candidate_text"):
            with self.subTest(field=field):
                substrate = copy.deepcopy(self.substrate)
                row = substrate["conditions"]["baseline"]["queries"][0]
                target = row if field == "query_text" else row["candidates"][0]
                target[field] += " altered"
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    freeze_qwen_top50.validate_frozen(substrate)

    def test_wrong_native_order_and_gold_membership_fail(self) -> None:
        substrate = copy.deepcopy(self.substrate)
        substrate["conditions"]["baseline"]["queries"][0]["candidates"][0]["native_rank"] = 2
        with self.assertRaisesRegex(ValueError, "native order mismatch"):
            freeze_qwen_top50.validate_frozen(substrate)
        substrate = copy.deepcopy(self.substrate)
        substrate["conditions"]["baseline"]["queries"][0]["gold_native_rank"] = 51
        with self.assertRaisesRegex(ValueError, "gold pool membership mismatch"):
            freeze_qwen_top50.validate_frozen(substrate)

    def test_ranking_hash_and_labels_fail_closed(self) -> None:
        for field, value in (("substrate_sha256", ""), ("substrate_sha256", "0" * 64), ("label", ""), ("execution_mode", "model_execution")):
            with self.subTest(field=field, value=value):
                rankings = copy.deepcopy(self.qwen_input)
                rankings[field] = value
                with self.assertRaises(ValueError):
                    self._replay(rankings)

    def test_ranking_query_and_condition_sets_fail(self) -> None:
        mutations = []
        duplicate = copy.deepcopy(self.qwen_input)
        duplicate["conditions"]["baseline"]["queries"][-1]["query_id"] = duplicate["conditions"]["baseline"]["queries"][0]["query_id"]
        mutations.append(duplicate)
        missing = copy.deepcopy(self.qwen_input)
        missing["conditions"]["baseline"]["queries"].pop()
        mutations.append(missing)
        unexpected = copy.deepcopy(self.qwen_input)
        unexpected["conditions"]["baseline"]["queries"][0]["query_id"] = "unexpected"
        mutations.append(unexpected)
        missing_condition = copy.deepcopy(self.qwen_input)
        del missing_condition["conditions"]["jina_selected"]
        mutations.append(missing_condition)
        wrong_total_count = copy.deepcopy(self.qwen_input)
        wrong_total_count["query_count"] = 5
        mutations.append(wrong_total_count)
        wrong_condition_count = copy.deepcopy(self.qwen_input)
        wrong_condition_count["conditions"]["baseline"]["query_count"] = 5
        mutations.append(wrong_condition_count)
        for rankings in mutations:
            with self.assertRaises(ValueError):
                self._replay(rankings)

    def test_missing_ranking_field_and_input_file_fail_clearly(self) -> None:
        rankings = copy.deepcopy(self.qwen_input)
        del rankings["conditions"]["baseline"]["queries"][0]["reranked_ids"]
        with self.assertRaisesRegex(ValueError, "exactly 50 IDs"):
            self._replay(rankings)
        with tempfile.TemporaryDirectory() as directory:
            missing = subprocess.run(
                [sys.executable, "-S", str(RERANKER_DIR / "replay_frozen_rankings.py"), "--substrate", str(Path(directory) / "missing.json"), "--rankings", str(self.qwen_path), "--output", str(Path(directory) / "out.json")],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("No such file", missing.stderr)

    def test_duplicate_missing_and_foreign_candidate_ids_fail(self) -> None:
        mutations = []
        duplicate = copy.deepcopy(self.qwen_input)
        duplicate["conditions"]["baseline"]["queries"][0]["reranked_ids"][0] = duplicate["conditions"]["baseline"]["queries"][0]["reranked_ids"][1]
        mutations.append(duplicate)
        missing = copy.deepcopy(self.qwen_input)
        missing["conditions"]["baseline"]["queries"][0]["reranked_ids"].pop()
        mutations.append(missing)
        foreign = copy.deepcopy(self.qwen_input)
        foreign["conditions"]["baseline"]["queries"][0]["reranked_ids"][0] = "foreign-id"
        mutations.append(foreign)
        for rankings in mutations:
            with self.assertRaises(ValueError):
                self._replay(rankings)

    def test_saved_metrics_must_match_rankings(self) -> None:
        result = self._replay(self.qwen_input)
        result["conditions"]["baseline"]["reranked_metrics"]["recall_at_10"] = 0.99
        with self.assertRaisesRegex(ValueError, "metrics or trace disagree"):
            replay_frozen_rankings.validate_result(
                result,
                self.substrate,
                replay_frozen_rankings._sha256(self.substrate_path),
            )

    def test_same_metrics_with_changed_full_order_fails_repeatability(self) -> None:
        left = self._replay(self.qwen_input)
        right = copy.deepcopy(left)
        trace = right["conditions"]["baseline"]["trace_rows"][0]
        trace["reranked_ids"][20], trace["reranked_ids"][21] = trace["reranked_ids"][21], trace["reranked_ids"][20]
        # This changes only irrelevant-item order; the gold rank and aggregate metrics stay identical.
        rows = self.substrate["conditions"]["baseline"]["queries"]
        rankings = {row["query_id"]: next(item for item in right["conditions"]["baseline"]["trace_rows"] if item["query_id"] == row["query_id"])["reranked_ids"] for row in rows}
        right["conditions"]["baseline"] = rerank_frozen_pools._evaluate(rows, rankings)
        with tempfile.TemporaryDirectory() as directory:
            left_path = Path(directory) / "left.json"
            right_path = Path(directory) / "right.json"
            left_path.write_text(json.dumps(left), encoding="utf-8")
            right_path.write_text(json.dumps(right), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("repeatability", REPEATABILITY_SCRIPT)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)
            result = module.verify(left_path, right_path, substrate_path=self.substrate_path)
        self.assertFalse(result["passed"])
        self.assertTrue(any("complete ranking differs" in error for error in result["errors"]))

    def test_synthetic_vs_model_mode_comparison_is_rejected(self) -> None:
        qwen = self._replay(self.qwen_input)
        model_result = copy.deepcopy(qwen)
        model_result["execution_mode"] = "model_execution"
        model_result["model_executed"] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qwen_path, model_path = root / "qwen.json", root / "model.json"
            qwen_path.write_text(json.dumps(qwen), encoding="utf-8")
            model_path.write_text(json.dumps(model_result), encoding="utf-8")
            command = [
                sys.executable, "-S", str(RERANKER_DIR / "compare_rerankers.py"),
                "--qwen", str(qwen_path), "--jina", str(model_path),
                "--substrate", str(self.substrate_path), "--output", str(root / "comparison.json"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("cannot compare synthetic replay with model-execution results", completed.stderr)

    def test_cli_replay_comparison_and_repeatability_use_stdlib_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = {}
            for name, ranking_path in (("qwen", self.qwen_path), ("jina", self.jina_path)):
                for run in (1, 2):
                    output = root / f"{name}-{run}.json"
                    command = [sys.executable, "-S", str(RERANKER_DIR / "replay_frozen_rankings.py"), "--substrate", str(self.substrate_path), "--rankings", str(ranking_path), "--output", str(output)]
                    completed = subprocess.run(command, capture_output=True, text=True, check=False)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    generated[f"{name}-{run}"] = output
            compare = subprocess.run(
                [sys.executable, "-S", str(RERANKER_DIR / "compare_rerankers.py"), "--qwen", str(generated["qwen-1"]), "--jina", str(generated["jina-1"]), "--substrate", str(self.substrate_path), "--output", str(root / "comparison.json")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(compare.returncode, 0, compare.stderr)
            report = (root / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("Synthetic ranking replay; no model inference was executed.", report)
            self.assertIn("cannot change a model or production decision", report)
            for name in ("qwen", "jina"):
                repeat = subprocess.run(
                    [sys.executable, "-S", str(REPEATABILITY_SCRIPT), "--left", str(generated[f"{name}-1"]), "--right", str(generated[f"{name}-2"]), "--substrate", str(self.substrate_path)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(repeat.returncode, 0, repeat.stderr)
                self.assertIn('"passed": true', repeat.stdout)
            for source in (RERANKER_DIR / "replay_frozen_rankings.py", REPEATABILITY_SCRIPT):
                help_result = subprocess.run([sys.executable, "-S", str(source), "--help"], capture_output=True, text=True, check=False)
                self.assertEqual(help_result.returncode, 0, help_result.stderr)


if __name__ == "__main__":
    unittest.main()
