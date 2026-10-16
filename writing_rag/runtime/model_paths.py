"""Project-local paths for downloaded model weights and Hugging Face caches."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
DEFAULT_HF_HOME = MODEL_WEIGHTS_DIR / "huggingface"
MODEL_CACHE_ENV = "WRITING_RAG_MODEL_CACHE"


def configure_model_cache(cache_dir: str | Path | None = None) -> Path:
    """Configure and return the Hugging Face home used by this project.

    An explicit function argument wins first, followed by the project-specific
    environment variable. With no override, model files stay under
    ``models/weights/huggingface``.
    """

    requested = cache_dir or os.environ.get(MODEL_CACHE_ENV) or DEFAULT_HF_HOME
    hf_home = Path(requested).expanduser()
    if not hf_home.is_absolute():
        hf_home = PROJECT_ROOT / hf_home
    hub_cache = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    # Set both current and compatibility names because the installed local
    # libraries do not all consult the same Hugging Face environment variable.
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    return hf_home


def resolve_cached_model_path(
    model_name: str | Path,
    cache_dir: str | Path | None = None,
) -> str | Path:
    """Return a complete local snapshot when one exists for a model ID.

    Passing a model ID to a loader can trigger a network lookup even when most
    weights are already cached. Resolving the ``refs/main`` snapshot ourselves
    keeps offline runs deterministic and leaves a model ID as the fallback when
    the cache is genuinely incomplete.
    """

    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate

    model_text = str(model_name)
    if "/" not in model_text:
        return model_name

    hf_home = configure_model_cache(cache_dir)
    repo_dir = hf_home / "hub" / f"models--{model_text.replace('/', '--')}"
    revision_file = repo_dir / "refs" / "main"
    if not revision_file.is_file():
        return model_name
    revision = revision_file.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    required = (snapshot / "config.json", snapshot / "tokenizer.json")
    if snapshot.is_dir() and all(path.exists() for path in required):
        return snapshot
    return model_name
