"""Compare source-agnostic Instagram windows with chain-aware episode windows.

This is an isolated experiment. It does not change the production window
builder or retrieval artifacts. The chain-aware builder keeps short messages
together when nearby turns show question/answer or causal continuity, while
retaining hard thread/time boundaries and a bounded token size.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESSAGES = PROJECT_ROOT / "data/normalized/messages.json"
DEFAULT_WINDOWS = PROJECT_ROOT / "data/windows/candidate_windows.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/instagram-chain-chunk-experiment-v1.json"

WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")
QUESTION_RE = re.compile(
    r"\?\s*$|^\s*(what|why|how|does|is|are|can|could|would|"
    r"kyu|kyunki|kya|kaise|kaise)\b",
    re.I,
)
CAUSAL_RE = re.compile(
    r"\b(because|therefore|that's why|that is why|which means|"
    r"what do you mean|implies|imply|alternative strategy)\b",
    re.I,
)
CLOSURE_RE = re.compile(
    r"^\s*(got it|achcha|okay|ok|theek hai|haan|yes|no|yeah|true)\b", re.I
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def token_count(text: str, token_counter: Callable[[str], int] | None = None) -> int:
    if token_counter is not None:
        return token_counter(text)
    return len(WORD_RE.findall(text))


def terms(text: str) -> set[str]:
    return {token.casefold() for token in WORD_RE.findall(text) if len(token) > 2}


def jaccard(left: str, right: str) -> float:
    left_terms = terms(left)
    right_terms = terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def is_question(text: str) -> bool:
    return bool(QUESTION_RE.search(text))


def causal_signal(left: str, right: str) -> bool:
    return bool(CAUSAL_RE.search(right) or (is_question(left) and right.strip()))


def edge_decision(
    left: dict[str, Any],
    right: dict[str, Any],
    current_tokens: int,
    *,
    max_gap_minutes: int,
    short_message_tokens: int,
    max_tokens: int = 700,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[bool, list[str]]:
    """Return whether adjacent turns belong to the same episode."""

    reasons: list[str] = []
    if left["source"] != right["source"]:
        return False, ["source_boundary"]
    if left["source_thread_id"] != right["source_thread_id"]:
        return False, ["thread_boundary"]

    gap_minutes = (right["timestamp_ms"] - left["timestamp_ms"]) / 60_000
    if gap_minutes > max_gap_minutes:
        return False, ["time_gap"]

    if gap_minutes <= 5:
        reasons.append("nearby_turns")
    elif gap_minutes <= 15:
        reasons.append("active_episode")

    left_tokens = token_count(left["content"], token_counter)
    right_tokens = token_count(right["content"], token_counter)
    if left_tokens <= short_message_tokens or right_tokens <= short_message_tokens:
        reasons.append("short_message_continuity")

    overlap = jaccard(left["content"], right["content"])
    if overlap >= 0.04:
        reasons.append("lexical_continuity")

    if is_question(left["content"]) or is_question(right["content"]):
        reasons.append("question_response_candidate")
    if causal_signal(left["content"], right["content"]):
        reasons.append("causal_language")

    if CLOSURE_RE.search(left["content"]) and not (
        jaccard(left["content"], right["content"]) >= 0.08
        or is_question(right["content"])
        or CAUSAL_RE.search(right["content"])
        or CLOSURE_RE.search(right["content"])
    ):
        return False, ["acknowledgement_boundary"]

    # Strong local adjacency is enough for tiny chat turns. Longer windows
    # need an explicit continuity signal so unrelated topics do not merge.
    strong_signal = any(
        reason
        in {
            "lexical_continuity",
            "question_response_candidate",
            "causal_language",
        }
        for reason in reasons
    )
    nearby_short_turns = (
        current_tokens < max_tokens
        and gap_minutes <= 5
        and (left_tokens <= short_message_tokens or right_tokens <= short_message_tokens)
    )
    keep = strong_signal or nearby_short_turns

    if keep and current_tokens >= max_tokens and not strong_signal:
        return False, ["token_budget"]
    return keep, reasons or ["weak_continuity"]


def build_chain_windows(
    messages: list[dict[str, Any]],
    *,
    max_gap_minutes: int = 30,
    max_tokens: int = 700,
    short_message_tokens: int = 24,
    token_counter: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(messages, key=lambda item: item["timestamp_ms"])
    windows: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    boundary_reasons: list[str] = []

    def flush() -> None:
        nonlocal current, current_tokens, boundary_reasons
        if not current:
            return
        windows.append(
            {
                "window_id": f"ig_chain_{len(windows) + 1:03d}",
                "message_ids": [item["message_id"] for item in current],
                "start_timestamp_ms": current[0]["timestamp_ms"],
                "end_timestamp_ms": current[-1]["timestamp_ms"],
                "message_count": len(current),
                "self_message_count": sum(item["speaker"] == "self" for item in current),
                "other_message_count": sum(item["speaker"] == "other" for item in current),
                "token_count": current_tokens,
                "turn_count": sum(
                    index == 0 or current[index]["speaker"] != current[index - 1]["speaker"]
                    for index in range(len(current))
                ),
                "boundary_before": boundary_reasons,
                "text": "\n".join(
                    f"{item['speaker'].title()}: {item['content']}" for item in current
                ),
            }
        )
        current = []
        current_tokens = 0
        boundary_reasons = []

    for message in ordered:
        if not current:
            current = [message]
            current_tokens = token_count(message["content"], token_counter)
            continue

        keep, reasons = edge_decision(
            current[-1],
            message,
            current_tokens,
            max_gap_minutes=max_gap_minutes,
            short_message_tokens=short_message_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        candidate_tokens = current_tokens + token_count(message["content"], token_counter)
        if candidate_tokens > max_tokens and current:
            keep = False
            reasons = ["token_budget"]
        if not keep:
            flush()
            boundary_reasons = reasons
        current.append(message)
        current_tokens += token_count(message["content"], token_counter)

    flush()
    return windows


def summarize_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [window["token_count"] for window in windows]
    return {
        "count": len(windows),
        "total_tokens": sum(counts),
        "median_tokens": sorted(counts)[(len(counts) - 1) // 2] if counts else 0,
        "max_tokens": max(counts, default=0),
        "message_count": sum(window["message_count"] for window in windows),
        "self_message_count": sum(window["self_message_count"] for window in windows),
    }


def current_overlapping_windows(
    windows: list[dict[str, Any]], start_ms: int, end_ms: int
) -> list[dict[str, Any]]:
    return [
        window
        for window in windows
        if window["source"] == "instagram"
        and window["end_timestamp_ms"] >= start_ms
        and window["start_timestamp_ms"] <= end_ms
    ]


def project_baseline_windows(
    windows: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Restrict baseline windows to the selected messages for fair comparison."""

    selected_by_id = {message["message_id"]: message for message in selected}
    projected: list[dict[str, Any]] = []
    for window in windows:
        message_ids = [
            message_id
            for message_id in window["message_ids"]
            if message_id in selected_by_id
        ]
        if not message_ids:
            continue
        messages = [selected_by_id[message_id] for message_id in message_ids]
        projected.append(
            {
                "window_id": window["window_id"],
                "message_ids": message_ids,
                "start_timestamp_ms": messages[0]["timestamp_ms"],
                "end_timestamp_ms": messages[-1]["timestamp_ms"],
                "message_count": len(messages),
                "self_message_count": sum(
                    message["speaker"] == "self" for message in messages
                ),
                "other_message_count": sum(
                    message["speaker"] == "other" for message in messages
                ),
                "token_count": sum(token_count(message["content"]) for message in messages),
                "turn_count": sum(
                    index == 0 or messages[index]["speaker"] != messages[index - 1]["speaker"]
                    for index in range(len(messages))
                ),
            }
        )
    return projected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--max-gap-minutes", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--short-message-tokens", type=int, default=24)
    args = parser.parse_args()

    all_messages = read_json(Path(args.messages))
    selected = [
        message
        for message in all_messages
        if message["source"] == "instagram"
        and args.start_ms <= message["timestamp_ms"] <= args.end_ms
    ]
    if not selected:
        raise SystemExit("No Instagram messages matched the requested time range")

    old_windows = project_baseline_windows(
        current_overlapping_windows(
            read_json(Path(args.windows)), args.start_ms, args.end_ms
        ),
        selected,
    )
    new_windows = build_chain_windows(
        selected,
        max_gap_minutes=args.max_gap_minutes,
        max_tokens=args.max_tokens,
        short_message_tokens=args.short_message_tokens,
    )
    result = {
        "experiment": "instagram_chain_chunk_experiment_v1",
        "input": {
            "start_ms": args.start_ms,
            "end_ms": args.end_ms,
            "message_count": len(selected),
            "self_message_count": sum(item["speaker"] == "self" for item in selected),
            "other_message_count": sum(item["speaker"] == "other" for item in selected),
        },
        "baseline": {
            "window_count": len(old_windows),
            "summary": summarize_windows(old_windows),
            "windows": old_windows,
        },
        "chain_aware": {
            "parameters": {
                "max_gap_minutes": args.max_gap_minutes,
                "max_tokens": args.max_tokens,
                "short_message_tokens": args.short_message_tokens,
            },
            "summary": summarize_windows(new_windows),
            "windows": new_windows,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("experiment", "input", "baseline", "chain_aware") if key != "baseline" and key != "chain_aware"}, indent=2))
    print(f"Baseline windows: {len(old_windows)}")
    print(f"Chain-aware windows: {len(new_windows)}")
    print(f"Saved experiment to {output}")


if __name__ == "__main__":
    main()
