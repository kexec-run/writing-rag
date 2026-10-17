from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/active/shape-beat-axis0-v1/run_public_validation.py"
MINIMAL = ROOT / "experiments/active/shape-beat-axis0-v1/fixtures/synthetic-minimal"
MULTI_BEAT = ROOT / "tests/fixtures/shape_beat_validation/multi-beat"
SPEC = importlib.util.spec_from_file_location("shape_beat_public_validation", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize(
    ("case_name", "expected_units"),
    [("minimal", {"S": 2, "SB": 2}), ("multi-beat", {"S": 2, "SB": 4})],
)
def test_public_case_compiles_and_executes_all_negative_checks(
    case_name: str, expected_units: dict[str, int]
) -> None:
    artifacts = runner.build_validation(case_name)
    report = artifacts["validation-report.json"]
    compiled = artifacts["compiled-units.json"]

    assert report["status"] == "passed"
    assert report["model_executed"] is False
    assert report["counts"] == {
        "required_checks": 22,
        "passed_checks": 22,
        "failed_checks": 0,
        "negative_checks_executed": 15,
        "negative_checks_passed": 15,
    }
    assert {arm: len(compiled[arm]["units"]) for arm in ("S", "SB")} == expected_units
    assert len(artifacts) == 5


def test_pool_mutations_are_rejected_against_full_baseline() -> None:
    pool = json.loads((MULTI_BEAT / "g0-pool.json").read_text(encoding="utf-8"))
    baseline = json.loads((MULTI_BEAT / "g0-baseline.json").read_text(encoding="utf-8"))
    mutations = (
        ("query", "changed query"),
        ("evidence_budget", 99),
        ("qwen_metadata_sha256", "0" * 64),
        ("candidate_document_ids", list(reversed(pool["candidate_document_ids"]))),
        ("allowed_witness_ids", pool["allowed_witness_ids"][:-1]),
    )
    for key, value in mutations:
        changed = copy.deepcopy(pool)
        changed[key] = value
        assert runner._validate_pool(changed, baseline), key

    changed_span = copy.deepcopy(pool)
    changed_span["candidates"][0]["span_id"] = "span-not-in-baseline"
    assert runner._validate_pool(changed_span, baseline)

    changed_hash = copy.deepcopy(pool)
    changed_hash["candidates"][0]["text_sha256"] = "0" * 64
    assert runner._validate_pool(changed_hash, baseline)


def test_obligation_bindings_and_case_inputs_are_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = json.loads((MULTI_BEAT / "composition-plan.json").read_text(encoding="utf-8"))
    invalid = copy.deepcopy(plan)
    invalid["evidence_obligations"][0]["binds"]["beat_id"] = "B99-99"
    assert runner.validate_public_obligation_contract(
        invalid, runner.EXPECTED_OBLIGATION_IDS["multi-beat"]
    )

    case_copy = tmp_path / "multi-beat"
    case_copy.mkdir()
    for name in runner.INPUT_NAMES:
        (case_copy / name).write_bytes((MULTI_BEAT / name).read_bytes())
    changed_pool = json.loads((case_copy / "g0-pool.json").read_text(encoding="utf-8"))
    changed_baseline = json.loads(
        (case_copy / "g0-baseline.json").read_text(encoding="utf-8")
    )
    changed_pool["extra_contract_field"] = "mutated"
    changed_baseline["extra_contract_field"] = "mutated"
    (case_copy / "g0-pool.json").write_text(
        json.dumps(changed_pool), encoding="utf-8"
    )
    (case_copy / "g0-baseline.json").write_text(
        json.dumps(changed_baseline), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "CASE_DIRECTORIES", {"multi-beat": case_copy})
    with pytest.raises(ValueError, match="fixed public case manifest"):
        runner.build_validation("multi-beat")


def test_required_fixture_failure_happens_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_copy = tmp_path / "case"
    fixture_copy.mkdir()
    for name in runner.INPUT_NAMES:
        source = MULTI_BEAT / name
        if name != "g0-baseline.json":
            (fixture_copy / name).write_bytes(source.read_bytes())
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "CASE_DIRECTORIES", {"broken": fixture_copy})
    output = tmp_path / "result"

    with pytest.raises(FileNotFoundError, match="g0-baseline.json"):
        runner.build_validation("broken")
    assert not output.exists()


def test_cli_rejects_unknown_case_extra_private_path_and_existing_output(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    output = tmp_path / "existing"
    output.mkdir()

    unknown = subprocess.run(
        [sys.executable, "-S", str(RUNNER), "--case", "oracle", "--output", str(tmp_path / "bad")],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert unknown.returncode == 2
    assert not (tmp_path / "bad").exists()

    private_path = subprocess.run(
        [sys.executable, "-S", str(RUNNER), "--case", "minimal", "--output", str(tmp_path / "bad"), "--input-dir", "/private/corpus"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert private_path.returncode == 2
    assert not (tmp_path / "bad").exists()

    existing = subprocess.run(
        [sys.executable, "-S", str(RUNNER), "--case", "minimal", "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert existing.returncode == 1
    assert "refusing to overwrite" in existing.stderr


@pytest.mark.parametrize("case_name", ["minimal", "multi-beat"])
def test_cli_outputs_are_repeatable_without_site_packages(
    case_name: str, tmp_path: Path
) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    output_dirs = [tmp_path / "first", tmp_path / "second"]
    for output in output_dirs:
        result = subprocess.run(
            [sys.executable, "-S", str(RUNNER), "--case", case_name, "--output", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
        assert '"status": "passed"' in result.stdout

    first_files = sorted(path.name for path in output_dirs[0].iterdir())
    second_files = sorted(path.name for path in output_dirs[1].iterdir())
    assert first_files == second_files == [
        "compiled-units.json",
        "obligation-set.json",
        "oflat.txt",
        "schedules.json",
        "validation-report.json",
    ]
    for filename in first_files:
        assert (output_dirs[0] / filename).read_bytes() == (
            output_dirs[1] / filename
        ).read_bytes()


def test_symlinked_fixture_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_dir = tmp_path / "fixtures" / "case"
    fixture_dir.mkdir(parents=True)
    source = MULTI_BEAT / "composition-plan.json"
    (fixture_dir / "composition-plan.json").symlink_to(source)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "CASE_DIRECTORIES", {"linked": fixture_dir})

    with pytest.raises(FileNotFoundError, match="composition-plan.json"):
        runner._read_case_json(fixture_dir, "composition-plan.json")
