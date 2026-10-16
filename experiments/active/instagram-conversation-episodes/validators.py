"""Positive and negative validation for experimental Instagram episodes."""

from __future__ import annotations

from typing import Any, Callable


def validate_episode(episode: dict[str, Any], messages_by_id: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    message_ids = [str(value) for value in episode.get("message_ids", [])]
    if not message_ids:
        errors.append("empty_message_ids")
    if len(message_ids) != len(set(message_ids)):
        errors.append("duplicate_message_id")
    missing = [message_id for message_id in message_ids if message_id not in messages_by_id]
    if missing:
        errors.append("missing_provenance")
    messages = [messages_by_id[message_id] for message_id in message_ids if message_id in messages_by_id]
    timestamps = [int(message.get("timestamp_ms", 0)) for message in messages]
    if timestamps != sorted(timestamps):
        errors.append("non_chronological")
    self_ids = {str(value) for value in episode.get("self_message_ids", [])}
    context_ids = {str(value) for value in episode.get("context_message_ids", [])}
    if self_ids & context_ids:
        errors.append("self_context_overlap")
    if not self_ids.issubset(set(message_ids)) or not context_ids.issubset(set(message_ids)):
        errors.append("projection_outside_episode")
    for message_id in self_ids:
        if messages_by_id.get(message_id, {}).get("speaker") != "self":
            errors.append("style_contamination")
    for role in episode.get("context_roles", []):
        if not set(role.get("source_message_ids", [])).issubset(context_ids):
            errors.append("context_summary_without_source")
    for edge in episode.get("self_edges", []):
        if edge.get("from_message_id") not in self_ids or edge.get("to_message_id") not in self_ids:
            errors.append("self_edge_outside_style_projection")
    return sorted(set(errors))


def run_fixture_checks() -> dict[str, bool]:
    """Check validator behavior with positive and deliberately invalid records."""

    messages = {
        "self-1": {"message_id": "self-1", "speaker": "self", "timestamp_ms": 1},
        "other-1": {"message_id": "other-1", "speaker": "other", "timestamp_ms": 2},
    }
    valid = {
        "message_ids": ["self-1", "other-1"],
        "self_message_ids": ["self-1"],
        "context_message_ids": ["other-1"],
        "context_roles": [{"source_message_ids": ["other-1"]}],
        "self_edges": [],
    }
    invalid = dict(valid)
    invalid["context_message_ids"] = ["self-1"]
    invalid["self_message_ids"] = ["other-1"]
    positive = validate_episode(valid, messages) == []
    negative = bool(validate_episode(invalid, messages))
    return {
        "positive_valid_episode": positive,
        "negative_contamination_rejected": negative,
        "passed": positive and negative,
    }


def run_perturbation_checks(builder: Callable[..., dict[str, Any]]) -> dict[str, bool]:
    """Exercise positive continuity and negative boundary perturbations."""

    def message(message_id: str, timestamp: int, speaker: str, content: str) -> dict[str, Any]:
        return {
            "message_id": message_id,
            "source": "instagram",
            "source_thread_id": "instagram_default",
            "source_path": "fixture",
            "raw_message_id": message_id,
            "timestamp_ms": timestamp,
            "speaker": speaker,
            "content": content,
        }

    base = [
        message("self-premise", 1, "self", "The first premise matters."),
        message("other-question", 2, "other", "Why?"),
        message("self-answer", 3, "self", "Because the next point follows from it."),
    ]
    positive = builder(base, 2, widths=(2, 4))["primary"]
    positive_continuity = {
        "other-question",
        "self-answer",
    }.issubset(set(positive["message_ids"]))

    unrelated = message("other-unrelated", 0, "other", "The weather is unrelated.")
    negative_noise_messages = [unrelated, *base]
    negative_noise = builder(negative_noise_messages, 3, widths=(3, 5))["primary"]
    unrelated_rejected = "other-unrelated" not in set(negative_noise["message_ids"])

    closure_messages = [
        *base,
        message("other-ack", 4, "other", "Okay."),
        message("self-unrelated", 5, "self", "A completely separate subject starts here."),
    ]
    closure = builder(closure_messages, 2, widths=(3, 5))["primary"]
    closure_respected = "self-unrelated" not in set(closure["message_ids"])
    passed = positive_continuity and unrelated_rejected and closure_respected
    return {
        "positive_question_response_preserved": positive_continuity,
        "negative_unrelated_insertion_rejected": unrelated_rejected,
        "negative_closure_boundary_respected": closure_respected,
        "passed": passed,
    }
