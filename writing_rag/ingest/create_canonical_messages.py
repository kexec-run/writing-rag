"""Build the lossless, source-agnostic canonical message layer."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from writing_rag.runtime.pipeline_common import (
    PROJECT_ROOT,
    clean_text,
    message_sort_key,
    normalized_sender,
    relative_source_path,
    stable_message_id,
    write_json,
)


def _configured_self_senders() -> dict[str, set[str]]:
    """Read source-specific self identities from local environment configuration."""
    raw = os.environ.get("WRITING_RAG_SELF_SENDERS", "{}")
    try:
        configured = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("WRITING_RAG_SELF_SENDERS must be a JSON object of source to sender-list mappings") from exc
    if not isinstance(configured, dict):
        raise TypeError("WRITING_RAG_SELF_SENDERS must be a JSON object")
    supported_sources = {"instagram", "whatsapp", "reddit"}
    unknown_sources = set(configured) - supported_sources
    if unknown_sources:
        raise ValueError(f"unsupported self-identity source(s): {', '.join(sorted(unknown_sources))}")
    result: dict[str, set[str]] = {}
    for source, senders in configured.items():
        if not isinstance(senders, list) or any(not isinstance(sender, str) for sender in senders):
            raise ValueError(f"self identities for {source} must be a list of strings")
        result[source] = {
            normalized
            for sender in senders
            if (normalized := normalized_sender(sender))
        }
    return result


SELF_SENDERS = _configured_self_senders()
# Whisper transcripts are source-native self narration, not an account identity.
SELF_SENDERS["whisper"] = {"self"}
WHATSAPP_TIMEZONE_NAME = os.environ.get("WRITING_RAG_WHATSAPP_TIMEZONE", "UTC").strip() or "UTC"
try:
    WHATSAPP_TIMEZONE = ZoneInfo(WHATSAPP_TIMEZONE_NAME)
except ZoneInfoNotFoundError as exc:
    raise ValueError(
        "WRITING_RAG_WHATSAPP_TIMEZONE must name a valid IANA timezone"
    ) from exc
WHATSAPP_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}), "
    r"(?P<time>\d{1,2}:\d{2}) - (?P<body>.*)$"
)


def whatsapp_timestamp(date_text: str, time_text: str) -> int:
    year_format = "%d/%m/%y" if len(date_text.rsplit("/", 1)[-1]) == 2 else "%d/%m/%Y"
    parsed = datetime.strptime(f"{date_text} {time_text}", f"{year_format} %H:%M")  # noqa: DTZ007
    return int(parsed.replace(tzinfo=WHATSAPP_TIMEZONE).timestamp() * 1000)


def is_reddit_deleted(event: dict[str, Any]) -> bool:
    content = event.get("content") or {}
    body = clean_text(content.get("body")).casefold()
    reason = clean_text(content.get("reason")).casefold()
    return (
        "scheduled deletion" in reason
        or "deleted" in reason
        or body in {"[deleted]", "[removed]"}
    )


def record(
    *,
    source: str,
    raw_thread_id: str,
    raw_message_id: str,
    timestamp_ms: int,
    sender: Any,
    content: Any,
    mode: str,
    source_path: str | Path,
    system: bool = False,
) -> dict[str, Any] | None:
    text = clean_text(content)
    if not text:
        return None

    sender_value = clean_text(sender) or None
    sender_value_normalized = normalized_sender(sender_value)
    self_senders = SELF_SENDERS.get(source, set())
    if system or sender_value_normalized is None:
        speaker = "system"
    elif sender_value_normalized in self_senders:
        speaker = "self"
    else:
        speaker = "other"

    source_thread_id = clean_text(raw_thread_id) or f"{source}_default"
    message_id = stable_message_id(
        source,
        raw_message_id,
        source_thread_id,
        timestamp_ms,
        sender_value_normalized,
        text,
    )
    return {
        "message_id": message_id,
        "source": source,
        "source_thread_id": source_thread_id,
        "raw_thread_id": raw_thread_id,
        "raw_message_id": raw_message_id,
        "timestamp_ms": int(timestamp_ms),
        "speaker": speaker,
        "sender": sender_value,
        "sender_normalized": sender_value_normalized,
        "content": text,
        "mode": mode,
        "source_path": relative_source_path(source_path),
    }


def load_instagram() -> Iterable[dict[str, Any]]:
    raw_dir = PROJECT_ROOT / "data/raw/instagram"
    for path in sorted(raw_dir.glob("message_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for index, message in enumerate(data.get("messages", [])):
            yield record(
                source="instagram",
                raw_thread_id="instagram_default",
                raw_message_id=f"{path.name}:{index}",
                timestamp_ms=int(message.get("timestamp_ms", 0)),
                sender=message.get("sender_name"),
                content=message.get("content"),
                mode="conversational",
                source_path=path,
                system=normalized_sender(message.get("sender_name")) in {"meta ai", "instagram"},
            )


def parse_whatsapp_file(path: Path) -> Iterable[dict[str, Any]]:
    current: dict[str, Any] | None = None
    current_line = 0

    def flush() -> dict[str, Any] | None:
        nonlocal current
        if current is None:
            return None
        value = current
        current = None
        return value

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            match = WHATSAPP_RE.match(line)
            if match:
                previous = flush()
                if previous is not None:
                    yield previous
                current_line = line_number
                body = match.group("body")
                if ": " in body:
                    sender, content = body.split(": ", 1)
                    current = {
                        "timestamp_ms": whatsapp_timestamp(match.group("date"), match.group("time")),
                        "sender": sender,
                        "content": content,
                        "raw_message_id": f"{path.name}:{current_line}",
                        "system": False,
                    }
                else:
                    current = {
                        "timestamp_ms": whatsapp_timestamp(match.group("date"), match.group("time")),
                        "sender": None,
                        "content": body,
                        "raw_message_id": f"{path.name}:{current_line}",
                        "system": True,
                    }
                continue
            if current is not None:
                current["content"] += f"\n{line}"

    previous = flush()
    if previous is not None:
        yield previous


def load_whatsapp() -> Iterable[dict[str, Any]]:
    raw_dir = PROJECT_ROOT / "data/raw/whatsapp"
    for path in sorted(raw_dir.glob("*.txt")):
        for message in parse_whatsapp_file(path):
            content = clean_text(message["content"])
            if content in {"<Media omitted>", "‎<Media omitted>"}:
                continue
            yield record(
                source="whatsapp",
                raw_thread_id=path.stem,
                raw_message_id=message["raw_message_id"],
                timestamp_ms=message["timestamp_ms"],
                sender=message["sender"],
                content=content,
                mode="long_form_reasoning",
                source_path=path,
                system=message["system"],
            )


def load_reddit() -> Iterable[dict[str, Any]]:
    path = PROJECT_ROOT / "data/raw/reddit/reddit_chat.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    events = data if isinstance(data, list) else next(
        (data[key] for key in ("events", "messages", "chunk") if isinstance(data.get(key), list)),
        [],
    )
    for index, event in enumerate(events):
        if event.get("type") != "m.room.message" or is_reddit_deleted(event):
            continue
        content = event.get("content") or {}
        yield record(
            source="reddit",
            raw_thread_id=event.get("room_id") or "reddit_default",
            raw_message_id=event.get("event_id") or f"event_{index}",
            timestamp_ms=int(event.get("origin_server_ts", 0)),
            sender=event.get("sender"),
            content=content.get("body"),
            mode="conversational",
            source_path=path,
        )


def load_whisper() -> Iterable[dict[str, Any]]:
    path = PROJECT_ROOT / "data/raw/whisper/transcript.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for index, segment in enumerate(data.get("transcription", [])):
        offsets = segment.get("offsets") or {}
        yield record(
            source="whisper",
            raw_thread_id="whisper_transcript",
            raw_message_id=f"segment_{index:06d}",
            timestamp_ms=int(offsets.get("from", 0)),
            sender="self",
            content=segment.get("text"),
            mode="long_form_reasoning",
            source_path=path,
        )


def load_chatgpt() -> Iterable[dict[str, Any]]:
    """Project normalized ChatGPT branches into the canonical graph-aware layer."""
    messages_path = PROJECT_ROOT / "data/chatgpt/normalized/messages.json"
    branches_path = PROJECT_ROOT / "data/chatgpt/normalized/branches.json"
    if not messages_path.exists() or not branches_path.exists():
        return
    with messages_path.open("r", encoding="utf-8") as handle:
        messages = json.load(handle)
    with branches_path.open("r", encoding="utf-8") as handle:
        branches = json.load(handle)

    branch_by_id = {branch["branch_id"]: branch for branch in branches}
    positions = {
        (branch["branch_id"], message_id): index
        for branch in branches
        for index, message_id in enumerate(branch.get("message_ids", []))
    }
    paths_by_message: dict[str, list[str]] = {}
    for branch in branches:
        for message_id in branch.get("message_ids", []):
            paths_by_message.setdefault(message_id, []).append(branch["branch_id"])
    for message in messages:
        branch_id = message["branch_id"]
        branch = branch_by_id.get(branch_id, {})
        raw_id = str(message["message_id"])
        canonical_id = f"chatgpt_{raw_id}"
        parent_id = message.get("parent_message_id")
        path_ids = sorted(set(paths_by_message.get(raw_id, [branch_id])))
        yield {
            "message_id": canonical_id,
            "source": "chatgpt",
            "source_thread_id": branch_id,
            "raw_thread_id": message.get("conversation_id") or branch_id,
            "raw_message_id": raw_id,
            "timestamp_ms": int(message.get("timestamp_ms") or 0),
            "speaker": message.get("speaker", "other"),
            "sender": message.get("role"),
            "sender_normalized": clean_text(message.get("role")).casefold() or None,
            "content": clean_text(message.get("content")),
            "mode": "branch_conversation",
            "source_path": relative_source_path(branch.get("source_file", messages_path)),
            "graph_kind": "chatgpt_branch",
            "graph_path_ids": path_ids,
            "graph_path_positions": {
                path_id: positions.get((path_id, raw_id), 0) for path_id in path_ids
            },
            "graph_parent_message_ids": [f"chatgpt_{parent_id}"] if parent_id else [],
            "graph_child_message_ids": [
                f"chatgpt_{child_id}" for child_id in message.get("child_message_ids", [])
            ],
            "parent_branch_id": branch.get("parent_branch_id"),
            "root_branch_id": branch.get("root_branch_id", branch_id),
        }


def load_twitter() -> Iterable[dict[str, Any]]:
    """Project self-containing Twitter graph paths into canonical messages."""
    tweets_path = PROJECT_ROOT / "data/twitter/normalized/tweets.json"
    paths_path = PROJECT_ROOT / "data/twitter/normalized/conversation_paths.json"
    if not tweets_path.exists() or not paths_path.exists():
        return
    with tweets_path.open("r", encoding="utf-8") as handle:
        tweets = json.load(handle)
    with paths_path.open("r", encoding="utf-8") as handle:
        paths = json.load(handle)

    memberships: dict[str, list[str]] = {}
    positions: dict[str, dict[str, int]] = {}
    for path in paths:
        if not path.get("contains_self"):
            continue
        path_id = path["path_id"]
        for index, message_id in enumerate(path.get("message_ids", [])):
            memberships.setdefault(message_id, []).append(path_id)
            positions.setdefault(message_id, {})[path_id] = index

    for tweet in tweets:
        message_id = tweet["message_id"]
        path_ids = sorted(set(memberships.get(message_id, [])))
        if not path_ids:
            continue
        parents = [
            parent
            for parent in (
                tweet.get("in_reply_to_message_id"),
                tweet.get("quoted_message_id"),
            )
            if parent in memberships
        ]
        yield {
            "message_id": message_id,
            "source": "twitter",
            "source_thread_id": str(tweet.get("conversation_id") or message_id),
            "raw_thread_id": str(tweet.get("conversation_id") or message_id),
            "raw_message_id": str(tweet.get("tweet_id") or message_id),
            "timestamp_ms": int(tweet.get("timestamp_ms") or 0),
            "speaker": tweet.get("speaker", "other"),
            "sender": tweet.get("sender"),
            "sender_normalized": clean_text(tweet.get("sender")).casefold() or None,
            "content": clean_text(tweet.get("content")),
            "mode": "graph_conversation",
            "source_path": "data/raw/twitter/twitter_no_metadata.json",
            "graph_kind": "twitter_reply_quote_path",
            "graph_path_ids": path_ids,
            "graph_path_positions": positions[message_id],
            "graph_parent_message_ids": parents,
            "graph_child_message_ids": [],
        }


def build_canonical_messages() -> list[dict[str, Any]]:
    loaders = (
        load_instagram,
        load_whatsapp,
        load_reddit,
        load_whisper,
        load_chatgpt,
        load_twitter,
    )
    messages = [message for loader in loaders for message in loader() if message is not None]
    return sorted(messages, key=message_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/normalized/messages.json",
        help="Canonical message output path relative to the project root.",
    )
    args = parser.parse_args()
    messages = build_canonical_messages()
    write_json(args.output, messages)
    print(f"Canonical messages: {len(messages)}")
    print(f"Self messages: {sum(message['speaker'] == 'self' for message in messages)}")
    print(f"Context messages: {sum(message['speaker'] == 'other' for message in messages)}")
    print(f"System messages: {sum(message['speaker'] == 'system' for message in messages)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
