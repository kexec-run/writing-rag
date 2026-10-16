"""Keep optional characterization tests outside the default core suite.

The archived tests are retained because they document prior model and runtime
experiments, but they require optional packages and are not part of the
canonical Qwen retrieval contract.  They can be collected explicitly with
``pytest --run-experiment-tests``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


OPTIONAL_EXPERIMENT_TESTS = {
    "test_qwen_characterization_validation.py",
    "test_qwen_embedding_characterization.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-experiment-tests",
        action="store_true",
        default=False,
        help="collect optional model/runtime characterization tests",
    )
    parser.addoption(
        "--run-private-data-tests",
        action="store_true",
        default=False,
        help="run tests that require private corpus or evaluation artifacts",
    )


def pytest_ignore_collect(collection_path: object, config: pytest.Config) -> bool:
    name = getattr(collection_path, "name", "")
    path_text = str(collection_path)
    if "experiments/archived/" in path_text:
        return True
    if config.getoption("--run-experiment-tests"):
        return False
    return name in OPTIONAL_EXPERIMENT_TESTS


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-private-data-tests"):
        return
    skip_private = pytest.mark.skip(reason="requires private corpus/evaluation artifacts; use --run-private-data-tests")
    for item in items:
        if "private_data" in item.keywords:
            item.add_marker(skip_private)
