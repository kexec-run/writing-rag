"""Load the approved Jina v5 Small MLX text-matching adapter.

This adapter is intentionally not a retriever. It accepts texts supplied by a
caller, encodes them symmetrically under the text-matching task, and exposes a
contract that can be passed to ``jina_text_selection``.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from writing_rag.select.jina_text_selection import validate_jina_contract
from writing_rag.runtime.model_paths import configure_model_cache


MODEL_NAME = "jinaai/jina-embeddings-v5-text-small-mlx"
MODEL_REVISION = "fe69cad2caa9a4adc37eaecc9d12c7be304caa36"
TASK = "text-matching"
DIMENSION = 1024
VECTOR_SPACE = "jina-v5-text-matching-mlx"


class JinaTextMatchingContractError(RuntimeError):
    """Raised when the local Jina snapshot or runtime violates the contract."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_hashes(snapshot: Path) -> dict[str, str]:
    names = ["config.json", "model.py", "model.safetensors", "tokenizer.json", "utils.py"]
    for task in ("retrieval", "text-matching", "clustering", "classification"):
        names.extend(
            [
                f"adapters/{task}/adapter_config.json",
                f"adapters/{task}/adapter_model.safetensors",
            ]
        )
    return {
        name: _sha256_file(snapshot / name)
        for name in names
        if (snapshot / name).is_file()
    }


def resolve_snapshot(model_cache: str | Path) -> Path:
    cache_home = configure_model_cache(model_cache)
    repo_dir = cache_home / "hub" / f"models--{MODEL_NAME.replace('/', '--')}"
    revision_file = repo_dir / "refs" / "main"
    revision = revision_file.read_text(encoding="utf-8").strip() if revision_file.is_file() else ""
    snapshot = repo_dir / "snapshots" / revision if revision else Path()
    if revision != MODEL_REVISION or not snapshot.is_dir():
        raise JinaTextMatchingContractError(
            f"expected local {MODEL_NAME}@{MODEL_REVISION}, found {revision or 'no refs/main'}"
        )
    required = ("config.json", "model.py", "model.safetensors", "tokenizer.json", "utils.py")
    if any(not (snapshot / name).is_file() for name in required):
        raise JinaTextMatchingContractError(f"Jina snapshot is incomplete: {snapshot}")
    for task in ("retrieval", "text-matching", "clustering", "classification"):
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (snapshot / "adapters" / task / name).is_file():
                raise JinaTextMatchingContractError(
                    f"Jina snapshot is missing {task} adapter file {name}"
                )
    return snapshot


def metal_probe() -> dict[str, Any]:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - host-dependent
        raise JinaTextMatchingContractError("MLX is unavailable") from exc
    value = mx.array([1.0]) + 1
    mx.eval(value)
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise JinaTextMatchingContractError("Jina MLX text matching did not select Metal")
    return {
        "status": "passed",
        "operation": "mx.array([1.0]) + 1; mx.eval(value)",
        "device": str(mx.default_device()),
        "result": value.tolist(),
    }


def _load_utils(snapshot: Path) -> Any:
    module_name = f"writing_rag_jina_text_matching_utils_{MODEL_REVISION[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, snapshot / "utils.py")
    if spec is None or spec.loader is None:
        raise JinaTextMatchingContractError(f"cannot load Jina utilities from {snapshot}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JinaTextMatchingMLX:
    """Encode supplied texts with Jina's symmetric text-matching adapter."""

    def __init__(self, *, model_cache: str | Path, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise JinaTextMatchingContractError(
                "text-matching requires batch_size=1 to preserve order symmetry"
            )
        import mlx.core as mx

        self.mx = mx
        self.metal = metal_probe()
        self.model_cache = configure_model_cache(model_cache)
        self.snapshot = resolve_snapshot(self.model_cache)
        utils = _load_utils(self.snapshot)
        self.model = utils.load_model(self.snapshot)
        self.model.switch_task(TASK)
        self.raw_output_dtype = "unknown"

    def encode(self, texts: list[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for text in texts:
            values = self.model.encode([text], task_type=TASK)
            self.mx.eval(values)
            self.raw_output_dtype = str(values.dtype)
            batch = np.asarray(values.tolist(), dtype=np.float32)
            if batch.shape != (1, DIMENSION):
                raise JinaTextMatchingContractError(
                    f"unexpected Jina text-matching shape: {batch.shape}"
                )
            rows.append(batch[0].tolist())
        return rows

    def contract(self) -> dict[str, Any]:
        contract = {
            "embedding_backend": "jina-v5-mlx",
            "backend": "jina-v5-mlx",
            "model_name": MODEL_NAME,
            "resolved_model": str(self.snapshot),
            "model_snapshot_revision": self.snapshot.name,
            "model_snapshot_file_hashes": _snapshot_hashes(self.snapshot),
            "task": TASK,
            "task_type": TASK,
            "prompt_name": None,
            "dimension": DIMENSION,
            "normalized": True,
            "fallback_used": False,
            "vector_space": VECTOR_SPACE,
            "backend_device": "metal",
            "mlx_device": str(self.mx.default_device()),
            "raw_output_dtype": self.raw_output_dtype,
            "analysis_dtype": "float32",
            "batch_size": 1,
        }
        validate_jina_contract(contract)
        return contract
