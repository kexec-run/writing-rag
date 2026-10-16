"""Shared helpers for the local writing-memory pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    tiktoken = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FallbackEncoding:
    """Small deterministic approximation used when the optional tokenizer is absent."""

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _load_encoding() -> Any:
    if tiktoken is None:
        return _FallbackEncoding()
    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except Exception:  # pragma: no cover - depends on unavailable tokenizer cache/network
        return _FallbackEncoding()


ENCODING = _load_encoding()


def project_path(path: str | Path) -> Path:
    """Resolve a project-relative path without changing the caller's cwd."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_json(path: str | Path) -> Any:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def normalized_sender(sender: Any) -> str | None:
    value = clean_text(sender)
    if not value:
        return None
    return re.sub(r"\s+", " ", value).casefold()


def stable_digest(*parts: Any, length: int = 16) -> str:
    payload = "\x1f".join(clean_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_message_id(source: str, raw_message_id: str, *identity_parts: Any) -> str:
    return f"{source}_msg_{stable_digest(raw_message_id, *identity_parts)}"


def relative_source_path(path: str | Path) -> str:
    candidate = project_path(path)
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def message_sort_key(message: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(message.get("timestamp_ms", 0)),
        str(message.get("source", "")),
        str(message.get("source_thread_id", "")),
        str(message.get("raw_message_id", "")),
        str(message.get("message_id", "")),
    )
