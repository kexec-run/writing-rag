"""Load project-local model tokenizers without permitting a network fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_paths import configure_model_cache, resolve_cached_model_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATOR_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"


def load_local_tokenizer(
    model_name: str,
    model_cache: str | Path | None = None,
) -> tuple[Any, Path]:
    """Load a tokenizer from the repository-local Hugging Face snapshot."""

    cache_root = configure_model_cache(model_cache)
    resolved = resolve_cached_model_path(model_name, cache_root)
    resolved_path = Path(resolved)
    if not resolved_path.is_dir():
        raise FileNotFoundError(
            f"Model tokenizer is not available in the local cache: {model_name}. "
            f"Expected a snapshot under {cache_root / 'hub'}."
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime install
        raise RuntimeError(
            "Model-aware token counting requires transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    return tokenizer, resolved_path


def count_with_tokenizer(tokenizer: Any, text: str) -> int:
    """Count text tokens without adding BOS/EOS or chat-template tokens."""

    return len(tokenizer.encode(text, add_special_tokens=False))


def tokenizer_metadata(tokenizer: Any, snapshot: Path, model_name: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "snapshot": str(snapshot),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens_added": False,
    }
