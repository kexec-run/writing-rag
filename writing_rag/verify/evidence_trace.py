"""Trace retrieval decisions before any paragraph is accepted or rejected.

This module deliberately does not decide whether an article is acceptable. It
records what the retriever returned, what the selector kept, and what still has
not been interpreted. The trace is therefore useful for separating missing
corpus coverage, ranking, source-span selection, and verifier-protocol errors.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


TRACE_VERSION = "1"
INTERPRETATION_PENDING = "pending"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")


def _source_reference(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    keys = (
        "source",
        "source_path",
        "path",
        "heading",
        "title",
        "thread_id",
        "source_id",
        "speaker",
    )
    return {
        key: metadata[key]
        for key in keys
        if key in metadata and metadata[key] not in (None, "", [])
    }


def _document_text(document: Any) -> str:
    return str(document.get("text") or "") if isinstance(document, dict) else ""


def _document_status(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "empty"
    words = WORD_RE.findall(stripped)
    if not words:
        return "gibberish"
    # This is intentionally conservative. A short fragment is not called
    # gibberish merely because it is short; only a repeated single token is
    # marked as suspicious here.
    if len(words) >= 4 and len(set(word.casefold() for word in words)) == 1:
        return "gibberish"
    return None


def _candidate_status(
    candidate: dict[str, Any],
    *,
    document_id: str,
    source_span: str,
    selected_ids: set[str],
    excluded_ids: set[str],
    seen_ids: set[str],
    minimum_score: float,
) -> tuple[str, str]:
    document = candidate.get("document")
    if not isinstance(document, dict):
        return "malformed_document", "retrieval result has no document object"
    text = _document_text(document)
    text_status = _document_status(text)
    if text_status:
        return text_status, f"document text classified as {text_status}"
    if document_id in seen_ids:
        return "duplicate", "document appeared more than once in this result set"
    if document_id in excluded_ids:
        return "excluded_existing", "document was already selected before this retrieval round"
    if document_id in selected_ids:
        if source_span:
            return "selected", "accepted by the evidence selector"
        return "selected_missing_source_span", "selected by rank but no source span was extracted"
    evidence_score = float(
        candidate.get("evidence_score", candidate.get("score", 0.0))
    )
    if evidence_score < minimum_score:
        return "below_threshold", "score is below the configured minimum"
    if not source_span:
        return "missing_source_span", "candidate scored but span selection returned no text"
    if (
        float(candidate.get("lexical_score", 0.0)) == 0.0
        and evidence_score < minimum_score + 0.05
    ):
        return "low_signal", "candidate has no lexical overlap and only a marginal score"
    return "available_adjacent", "candidate has usable text but was not selected"


def build_candidate_trace(
    intent: dict[str, Any],
    retrieval_query: str,
    candidates: list[dict[str, Any]],
    *,
    span_selector: Callable[[str, str], str],
    minimum_score: float,
    selected_ids: set[str] | None = None,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return an inspectable trace for every candidate in one retrieval call."""

    selected = selected_ids or set()
    excluded = excluded_ids or set()
    seen_ids: set[str] = set()
    trace: list[dict[str, Any]] = []
    for fallback_rank, candidate in enumerate(candidates, start=1):
        document = candidate.get("document")
        document_id = (
            str(document.get("document_id"))
            if isinstance(document, dict) and document.get("document_id") is not None
            else ""
        )
        text = _document_text(document)
        source_span = ""
        span_error = ""
        if text:
            try:
                source_span = str(span_selector(text, retrieval_query) or "").strip()
            except Exception as exc:  # pragma: no cover - defensive trace path
                span_error = f"{type(exc).__name__}: {exc}"
        status, reason = _candidate_status(
            candidate,
            document_id=document_id,
            source_span=source_span,
            selected_ids=selected,
            excluded_ids=excluded,
            seen_ids=seen_ids,
            minimum_score=minimum_score,
        )
        seen_ids.add(document_id)
        trace.append(
            {
                "rank": candidate.get("rank", fallback_rank),
                "document_id": document_id,
                "score": candidate.get("score"),
                "evidence_score": candidate.get(
                    "evidence_score", candidate.get("score")
                ),
                "base_score": candidate.get("base_score"),
                "rrf_score": candidate.get("rrf_score"),
                "cosine_score": candidate.get("cosine_score"),
                "lexical_score": candidate.get("lexical_score"),
                "retrieval_strategy": candidate.get(
                    "retrieval_strategy", "baseline"
                ),
                "facet_trace": candidate.get("facet_trace", {}),
                "source_reference": _source_reference(document)
                if isinstance(document, dict)
                else {},
                "source_span": source_span,
                "source_span_error": span_error,
                "document_text_length": len(text),
                "retrieval_status": status,
                "selection_reason": reason,
                # These fields are intentionally empty until a real
                # evidence-to-question interpretation stage runs.
                "interpretation_status": INTERPRETATION_PENDING,
                "supports_claim": None,
                "answers_question": None,
                "implied_question": None,
                "proposition": None,
                "role_candidates": [],
                "interpretation_confidence": None,
                "missing_nuance": None,
            }
        )
    return trace


def retrieval_trace_event(
    *,
    intent: dict[str, Any],
    query: str,
    candidates: list[dict[str, Any]],
    candidate_trace: list[dict[str, Any]],
    round_number: int,
    top_k: int,
    minimum_score: float,
) -> dict[str, Any]:
    """Package one retrieval round without claiming interpretation occurred."""

    return {
        "round": round_number,
        "query": query,
        "candidate_count": len(candidates),
        "requested_evidence_count": top_k,
        "minimum_score": minimum_score,
        "target_question": intent.get("question", ""),
        "target_claim": intent.get("intended_claim", ""),
        "target_role": intent.get("role", ""),
        "interpretation_status": INTERPRETATION_PENDING,
        "interpretation_note": (
            "The retriever has not inferred an implied question or proposition "
            "from these candidates."
        ),
        "candidates": candidate_trace,
    }


def render_debug_trace(ledger: list[dict[str, Any]]) -> str:
    """Render the JSON trace as explicit start/end-labelled debug text."""

    lines = ["[START: evidence_trace]", f"trace_version: {TRACE_VERSION}"]
    for intent in ledger:
        lines.extend(
            [
                "[START: intent]",
                f"intent_id: {intent.get('intent_id', '')}",
                f"order: {intent.get('order', '')}",
                f"role: {intent.get('role', '')}",
                f"question: {intent.get('question', '')}",
                f"intended_claim: {intent.get('intended_claim', '')}",
            ]
        )
        for event in intent.get("retrieval_trace", []):
            lines.extend(
                [
                    "[START: retrieval]",
                    f"round: {event.get('round', '')}",
                    f"query: {event.get('query', '')}",
                    f"target_claim: {event.get('target_claim', '')}",
                    f"target_question: {event.get('target_question', '')}",
                    f"target_role: {event.get('target_role', '')}",
                    f"interpretation_status: {event.get('interpretation_status', '')}",
                    "[END: retrieval]",
                ]
            )
            for candidate in event.get("candidates", []):
                lines.extend(
                    [
                        "[START: candidate]",
                        f"rank: {candidate.get('rank', '')}",
                        f"document_id: {candidate.get('document_id', '')}",
                        f"score: {candidate.get('score', '')}",
                        f"cosine_score: {candidate.get('cosine_score', '')}",
                        f"lexical_score: {candidate.get('lexical_score', '')}",
                        f"source_reference: {candidate.get('source_reference', {})}",
                        f"retrieval_status: {candidate.get('retrieval_status', '')}",
                        f"selection_reason: {candidate.get('selection_reason', '')}",
                        "[START: source_span]",
                        candidate.get("source_span", "") or "<none>",
                        "[END: source_span]",
                        "[START: interpretation]",
                        f"status: {candidate.get('interpretation_status', '')}",
                        "supports_claim: <not assessed>",
                        "answers_question: <not assessed>",
                        "implied_question: <not inferred>",
                        "proposition: <not inferred>",
                        "role_candidates: <not inferred>",
                        "missing_nuance: <not inferred>",
                        "[END: interpretation]",
                        "[END: candidate]",
                    ]
                )
        lines.append("[END: intent]")
    lines.append("[END: evidence_trace]")
    return "\n".join(lines) + "\n"
