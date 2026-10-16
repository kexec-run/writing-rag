"""Strict MLX adapter for lexrivera/zembed-1-embedding-mlx-6Bit."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

MODEL_NAME = "lexrivera/zembed-1-embedding-mlx-6Bit"
MODEL_CARD_REFERENCE = "https://huggingface.co/zeroentropy/zembed-1-embedding"
DIMENSION = 2560
MAX_LENGTH = 32768
QUERY_PREFIX = "<|im_start|>system\nquery<|im_end|>\n<|im_start|>user\n"
DOCUMENT_PREFIX = "<|im_start|>system\ndocument<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n"


class ZembedContractError(RuntimeError):
    """Raised when the MLX checkpoint cannot satisfy the embedding contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_snapshot(model: str, model_cache: str | Path) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    cache_root = Path(model_cache).expanduser().resolve()
    direct = cache_root / model
    if direct.is_dir():
        return direct.resolve()
    encoded_name = model.replace("/", "--")
    matches = sorted(
        path
        for path in cache_root.glob(f"**/models--{encoded_name}/snapshots/*")
        if (path / "model.safetensors").is_file()
    )
    if matches:
        return matches[-1].resolve()
    raise ZembedContractError(f"could not resolve {model} from repo cache {cache_root}")


def _normalise_rows(rows: Any, mx: Any) -> Any:
    norms = mx.sqrt(mx.sum(rows * rows, axis=-1, keepdims=True))
    mx.eval(norms)
    if bool(mx.any(norms <= 0)):
        raise ZembedContractError("zero embedding produced")
    return rows / norms


class ZembedMLXBackend:
    """Batch encoder with explicit zembed query/document roles."""

    name = "zembed-mlx-6bit"

    def __init__(self, model: str = MODEL_NAME, *, model_cache: str | Path, batch_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - host dependency
            raise ZembedContractError("MLX and mlx-lm are required") from exc
        self.mx = mx
        self.model_name = model
        self.model_cache = str(Path(model_cache).expanduser().resolve())
        self.resolved_model = str(_resolve_snapshot(model, model_cache))
        self.batch_size = batch_size
        self.model, self.tokenizer = load(self.resolved_model)
        self.last_output_dtype = "unknown"
        self.last_role: str | None = None
        self._validate_snapshot()

    def _validate_snapshot(self) -> None:
        root = Path(self.resolved_model)
        required = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ZembedContractError(f"checkpoint missing required files: {missing}")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("hidden_size") != DIMENSION:
            raise ZembedContractError(f"expected hidden_size={DIMENSION}, got {config.get('hidden_size')}")
        quant = config.get("quantization", config.get("quantization_config", {}))
        if quant.get("bits") != 6:
            raise ZembedContractError(f"expected 6-bit checkpoint, got {quant}")
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.configured_padding_side = tokenizer_config.get("padding_side")
        if self.configured_padding_side not in {"left", "right"}:
            raise ZembedContractError(
                f"unsupported tokenizer padding_side={self.configured_padding_side!r}"
            )
        # mlx-lm's causal Qwen3 forward path creates only a causal mask and
        # does not consume an attention mask. Right-padding keeps pad tokens
        # after each real sequence, so they cannot affect its hidden states.
        self.padding_side = "right"

    def _tokenize(self, texts: list[str]) -> tuple[Any, Any]:
        tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = self.padding_side
        try:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="np",
                add_special_tokens=False,
            )
        except TypeError:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="np",
            )
        input_ids = self.mx.array(encoded["input_ids"])
        attention = encoded.get("attention_mask")
        if attention is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                raise ZembedContractError("tokenizer has no attention mask or pad token")
            attention = input_ids != pad_id
        return input_ids, self.mx.array(attention)

    def _hidden_states(self, input_ids: Any) -> Any:
        core = getattr(self.model, "model", None)
        if core is None:
            raise ZembedContractError("MLX model exposes no transformer hidden-state module")
        result = core(input_ids)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, dict):
            if result.get("last_hidden_state") is not None:
                result = result["last_hidden_state"]
            elif result.get("hidden_states"):
                result = result["hidden_states"][-1]
            else:
                result = None
        shape = getattr(result, "shape", None)
        if result is None or shape is None or len(shape) != 3 or shape[-1] != DIMENSION:
            raise ZembedContractError(f"expected hidden states [batch, sequence, {DIMENSION}], got {shape}")
        return result

    def encode(self, texts: list[str], *, role: str, batch_size: int | None = None) -> list[list[float]]:
        if role not in {"query", "document"}:
            raise ValueError("role must be query or document")
        if not texts:
            return []
        prefix = QUERY_PREFIX if role == "query" else DOCUMENT_PREFIX
        formatted = [prefix + str(text) + SUFFIX for text in texts]
        size = batch_size or self.batch_size
        output: list[list[float]] = []
        self.last_role = role
        started = time.perf_counter()
        for start in range(0, len(formatted), size):
            input_ids, attention = self._tokenize(formatted[start : start + size])
            hidden = self._hidden_states(input_ids)
            if self.padding_side == "left":
                last_positions_valid = self.mx.all(attention[:, -1] == 1)
                self.mx.eval(last_positions_valid)
                if not bool(last_positions_valid):
                    raise ZembedContractError("left-padded batch has an invalid final position")
                rows = hidden[:, -1, :]
            else:
                first_positions_valid = self.mx.all(attention[:, 0] == 1)
                self.mx.eval(first_positions_valid)
                if not bool(first_positions_valid):
                    raise ZembedContractError("right-padded batch has an invalid first position")
                lengths = self.mx.sum(attention, axis=1) - 1
                rows = hidden[self.mx.arange(hidden.shape[0]), lengths]
            rows = rows.astype(self.mx.float32)
            rows = _normalise_rows(rows, self.mx)
            self.mx.eval(rows)
            self.last_output_dtype = str(rows.dtype)
            output.extend(rows.tolist())
        self.last_elapsed_seconds = time.perf_counter() - started
        return output

    def encode_queries(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
        return self.encode(texts, role="query", batch_size=batch_size)

    def encode_documents(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
        return self.encode(texts, role="document", batch_size=batch_size)

    def contract_metadata(self) -> dict[str, Any]:
        root = Path(self.resolved_model)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        return {
            "backend": self.name,
            "model_name": self.model_name,
            "resolved_model": self.resolved_model,
            "model_card_reference": MODEL_CARD_REFERENCE,
            "dimension": DIMENSION,
            "max_length": MAX_LENGTH,
            "quantization": config.get("quantization", config.get("quantization_config")),
            "query_prefix": QUERY_PREFIX,
            "document_prefix": DOCUMENT_PREFIX,
            "suffix": SUFFIX,
            "pooling": "last_non_padding_token_hidden_state",
            "padding_side": self.padding_side,
            "configured_padding_side": self.configured_padding_side,
            "padding_policy": "right_override_for_mlx_causal_attention_without_padding_mask",
            "normalized": True,
            "fallback_used": False,
            "snapshot_files": {name: sha256_file(root / name) for name in required_snapshot_files(root)},
        }


def required_snapshot_files(root: Path) -> tuple[str, ...]:
    return tuple(name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors") if (root / name).is_file())
