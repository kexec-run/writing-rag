"""Failure-isolated boundaries for local generation calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationCallResult:
    """The outcome of one model/tool call, including recoverable failures."""

    stage: str
    ok: bool
    text: str
    attempts: int
    error: str | None = None
    fallback_used: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _error_text(error: Exception) -> str:
    message = str(error).strip() or "no error message"
    return f"{type(error).__name__}: {message}"[:500]


def _retryable(error: Exception) -> bool:
    """Avoid repeating failures that cannot be fixed by another model call."""

    message = str(error).casefold()
    fatal_markers = (
        "requires mlx",
        "no module named",
        "unsupported generator backend",
        "compatible apple silicon runtime",
    )
    return not any(marker in message for marker in fatal_markers)


def run_generation_call(
    generate: Callable[[str, int, float], str],
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    stage: str,
    retries: int = 1,
    fallback: str = "",
    retry_temperature: float | None = None,
) -> GenerationCallResult:
    """Run a model call without allowing one stage to abort the whole run.

    A retry is attempted for ordinary runtime failures and empty responses. Known
    setup failures, such as a missing MLX package, fail immediately. A fallback is
    returned as text but the result remains ``ok=False`` so callers cannot mistake
    source text for model output.
    """

    errors: list[str] = []
    max_attempts = max(0, retries) + 1
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        attempt_temperature = (
            temperature
            if attempt == 1 or retry_temperature is None
            else retry_temperature
        )
        try:
            raw_text = generate(prompt, max_tokens, attempt_temperature)
        except Exception as error:  # model/runtime errors are stage-local
            errors.append(_error_text(error))
            if not _retryable(error):
                break
            continue

        text = str(raw_text or "").strip()
        if text:
            return GenerationCallResult(
                stage=stage,
                ok=True,
                text=text,
                attempts=attempt,
            )
        errors.append("empty model output")

    error = "; ".join(dict.fromkeys(errors)) or "generation failed"
    fallback_text = fallback.strip()
    return GenerationCallResult(
        stage=stage,
        ok=False,
        text=fallback_text,
        attempts=attempts,
        error=error,
        fallback_used=bool(fallback_text),
    )
