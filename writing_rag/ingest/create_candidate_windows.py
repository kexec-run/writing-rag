"""Create structure-aware conversation chunks from canonical messages.

Documents use headings and paragraphs; conversations use turns, paragraph
boundaries, time gaps, topic shifts, and token limits. Headings are never
invented here. Topic terms are metadata only.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from writing_rag.runtime.pipeline_common import count_tokens, message_sort_key, read_json, write_json
from writing_rag.retrieve.embed_chunks import tokenize


def combined_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(message["content"] for message in messages)


def window_token_count(
    messages: list[dict[str, Any]],
    token_counter: Callable[[str], int] = count_tokens,
) -> int:
    return token_counter(combined_text(messages))


def paragraph_count(messages: list[dict[str, Any]]) -> int:
    return sum(max(1, len([part for part in message["content"].split("\n\n") if part.strip()]))
               for message in messages)


def turn_count(messages: list[dict[str, Any]]) -> int:
    if not messages:
        return 0
    return 1 + sum(
        messages[index]["speaker"] != messages[index - 1]["speaker"]
        for index in range(1, len(messages))
    )


def topic_terms(text: str) -> set[str]:
    return {term for term in tokenize(text) if len(term) > 2}


def topic_similarity(left: str, right: str) -> float:
    left_terms = topic_terms(left)
    right_terms = topic_terms(right)
    if len(left_terms) < 2 or len(right_terms) < 2:
        return 1.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def build_window(
    number: int,
    messages: list[dict[str, Any]],
    *,
    max_gap_minutes: int,
    max_tokens: int,
    overlap_messages: int,
    overlap_count: int,
    start_reason: str,
    topic_shift_threshold: float,
    graph_path_id: str | None = None,
    token_counter: Callable[[str], int] = count_tokens,
) -> dict[str, Any]:
    self_ids = [message["message_id"] for message in messages if message["speaker"] == "self"]
    other_ids = [message["message_id"] for message in messages if message["speaker"] == "other"]
    system_ids = [message["message_id"] for message in messages if message["speaker"] == "system"]
    terms = sorted({term for message in messages for term in topic_terms(message["content"])})[:64]
    return {
        "window_id": f"window_{number:06d}",
        "chunk_type": "conversation",
        "heading": None,
        "source": messages[0]["source"],
        "source_thread_id": graph_path_id or messages[0]["source_thread_id"],
        "graph_path_id": graph_path_id,
        "start_timestamp_ms": messages[0]["timestamp_ms"],
        "end_timestamp_ms": messages[-1]["timestamp_ms"],
        "message_ids": [message["message_id"] for message in messages],
        "self_message_ids": self_ids,
        "other_message_ids": other_ids,
        "system_message_ids": system_ids,
        "message_count": len(messages),
        "self_message_count": len(self_ids),
        "other_message_count": len(other_ids),
        "system_message_count": len(system_ids),
        "paragraph_count": paragraph_count(messages),
        "turn_count": turn_count(messages),
        "topic_terms": terms,
        "token_count": window_token_count(messages, token_counter),
        "grouping": {
            "method": (
                "graph_path_conversation_window"
                if graph_path_id
                else "conversation_structure_window"
            ),
            "boundaries": [
                "paragraph",
                "speaker_turn",
                "time_gap",
                "topic_shift",
                "max_tokens",
            ],
            "max_gap_minutes": max_gap_minutes,
            "max_tokens": max_tokens,
            "overlap_messages": overlap_messages,
            "topic_shift_threshold": topic_shift_threshold,
            "start_reason": start_reason,
        },
        "overlap_message_count": overlap_count,
        "contains_overlap": overlap_count > 0,
    }


def _create_linear_windows(
    messages: list[dict[str, Any]],
    *,
    max_gap_minutes: int = 30,
    max_tokens: int = 1200,
    overlap_messages: int = 0,
    topic_shift_threshold: float = 0.08,
    min_topic_tokens: int = 120,
    graph_path_id: str | None = None,
    token_counter: Callable[[str], int] = count_tokens,
) -> list[dict[str, Any]]:
    if max_tokens <= 0 or max_gap_minutes < 0 or overlap_messages < 0:
        raise ValueError("Window limits must be non-negative, with max_tokens greater than zero")
    if not 0 <= topic_shift_threshold <= 1 or min_topic_tokens < 0:
        raise ValueError("Topic-shift threshold must be between zero and one")

    ordered = sorted(messages, key=message_sort_key)
    windows: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    current_overlap_count = 0
    current_start_reason = "source_start"

    for message in ordered:
        if not current:
            current = [message]
            current_tokens = token_counter(message["content"])
            current_overlap_count = 0
            continue

        previous = current[-1]
        same_group = (
            message["source"] == previous["source"]
            and (
                graph_path_id is not None
                or message["source_thread_id"] == previous["source_thread_id"]
            )
        )
        gap_ms = message["timestamp_ms"] - previous["timestamp_ms"]
        candidate_tokens = window_token_count(current + [message], token_counter)
        breaks_group = not same_group
        breaks_time = gap_ms > max_gap_minutes * 60 * 1000
        breaks_topic = (
            current_tokens >= min_topic_tokens
            and topic_similarity(previous["content"], message["content"]) < topic_shift_threshold
        )
        breaks_tokens = candidate_tokens > max_tokens and bool(current)

        if breaks_group or breaks_time or breaks_topic or breaks_tokens:
            windows.append(
                build_window(
                    len(windows) + 1,
                    current,
                    max_gap_minutes=max_gap_minutes,
                    max_tokens=max_tokens,
                    overlap_messages=overlap_messages,
                    overlap_count=current_overlap_count,
                    start_reason=current_start_reason,
                    topic_shift_threshold=topic_shift_threshold,
                    graph_path_id=graph_path_id,
                    token_counter=token_counter,
                )
            )

            if breaks_group or breaks_time or breaks_topic:
                overlap = []
            else:
                overlap = current[-overlap_messages:] if overlap_messages else []
            current = overlap + [message]
            current_overlap_count = len(overlap)
            current_start_reason = (
                "source_or_thread_boundary"
                if breaks_group
                else "time_gap"
                if breaks_time
                else "topic_shift"
                if breaks_topic
                else "max_tokens"
            )
            current_tokens = window_token_count(current, token_counter)
            while current_tokens > max_tokens and len(current) > 1:
                current.pop(0)
                current_overlap_count = max(0, current_overlap_count - 1)
                current_tokens = window_token_count(current, token_counter)
            continue

        current.append(message)
        current_tokens = candidate_tokens

    if current:
        windows.append(
            build_window(
                len(windows) + 1,
                current,
                max_gap_minutes=max_gap_minutes,
                max_tokens=max_tokens,
                overlap_messages=overlap_messages,
                overlap_count=current_overlap_count,
                start_reason=current_start_reason,
                topic_shift_threshold=topic_shift_threshold,
                graph_path_id=graph_path_id,
                token_counter=token_counter,
            )
        )
    return windows


def create_candidate_windows(
    messages: list[dict[str, Any]],
    *,
    max_gap_minutes: int = 30,
    max_tokens: int = 1200,
    overlap_messages: int = 0,
    topic_shift_threshold: float = 0.08,
    min_topic_tokens: int = 120,
    token_counter: Callable[[str], int] = count_tokens,
) -> list[dict[str, Any]]:
    """Use explicit graph paths when present and structural windows otherwise."""
    graph_messages = [message for message in messages if message.get("graph_path_ids")]
    ordinary_messages = [message for message in messages if not message.get("graph_path_ids")]
    windows = _create_linear_windows(
        ordinary_messages,
        max_gap_minutes=max_gap_minutes,
        max_tokens=max_tokens,
        overlap_messages=overlap_messages,
        topic_shift_threshold=topic_shift_threshold,
        min_topic_tokens=min_topic_tokens,
        token_counter=token_counter,
    )

    path_ids = sorted(
        {
            path_id
            for message in graph_messages
            for path_id in message.get("graph_path_ids", [])
        }
    )
    for path_id in path_ids:
        path_messages = [
            message
            for message in graph_messages
            if path_id in message.get("graph_path_ids", [])
        ]
        path_messages.sort(
            key=lambda message: (
                message.get("graph_path_positions", {}).get(path_id, 0),
                message["message_id"],
            )
        )
        windows.extend(
            _create_linear_windows(
                path_messages,
                max_gap_minutes=max_gap_minutes,
                max_tokens=max_tokens,
                overlap_messages=overlap_messages,
                topic_shift_threshold=topic_shift_threshold,
                min_topic_tokens=min_topic_tokens,
                graph_path_id=path_id,
                token_counter=token_counter,
            )
        )

    for number, window in enumerate(windows, start=1):
        window["window_id"] = f"window_{number:06d}"
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/normalized/messages.json")
    parser.add_argument("--output", default="data/windows/candidate_windows.json")
    parser.add_argument("--max-gap-minutes", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--overlap-messages", type=int, default=0)
    parser.add_argument("--topic-shift-threshold", type=float, default=0.08)
    parser.add_argument("--min-topic-tokens", type=int, default=120)
    args = parser.parse_args()

    messages = read_json(args.input)
    windows = create_candidate_windows(
        messages,
        max_gap_minutes=args.max_gap_minutes,
        max_tokens=args.max_tokens,
        overlap_messages=args.overlap_messages,
        topic_shift_threshold=args.topic_shift_threshold,
        min_topic_tokens=args.min_topic_tokens,
    )
    write_json(args.output, windows)
    print(f"Conversation chunks: {len(windows)}")
    print(f"Overlapping chunks: {sum(window['contains_overlap'] for window in windows)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
