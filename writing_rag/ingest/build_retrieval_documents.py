"""Project canonical conversation chunks into two retrieval collections.

This is the core path. Reasoning chains can enrich these documents later, but
they are not required for retrieval. Content documents contain claims and
context; style documents contain self-authored text only.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

from writing_rag.runtime.pipeline_common import read_json, write_json


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def content_text(chunk: dict[str, Any], messages_by_id: dict[str, dict[str, Any]]) -> str:
    sections: list[str] = []
    context = [
        messages_by_id[message_id]["content"]
        for message_id in chunk["other_message_ids"] + chunk["system_message_ids"]
    ]
    self_text = [
        messages_by_id[message_id]["content"]
        for message_id in chunk["self_message_ids"]
    ]
    if context:
        sections.append("Context:\n" + "\n".join(context))
    if self_text:
        sections.append("Self:\n" + "\n".join(self_text))
    return "\n\n".join(sections)


def style_text(chunk: dict[str, Any], messages_by_id: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        messages_by_id[message_id]["content"]
        for message_id in chunk["self_message_ids"]
    )


def base_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["window_id"],
        "source": chunk["source"],
        "source_thread_id": chunk["source_thread_id"],
        "graph_path_id": chunk.get("graph_path_id"),
        "grouping_method": chunk.get("grouping", {}).get("method"),
        "message_ids": ",".join(chunk["message_ids"]),
        "self_message_ids": ",".join(chunk["self_message_ids"]),
        "other_message_ids": ",".join(chunk["other_message_ids"]),
        "system_message_ids": ",".join(chunk["system_message_ids"]),
        "start_timestamp_ms": chunk["start_timestamp_ms"],
        "end_timestamp_ms": chunk["end_timestamp_ms"],
        "paragraph_count": chunk["paragraph_count"],
        "turn_count": chunk["turn_count"],
        "topic_terms": ",".join(chunk.get("topic_terms", [])),
        "heading": None,
        "reasoning_status": "not_generated",
    }


def build_documents(
    chunks: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    messages_by_id = {message["message_id"]: message for message in messages}
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        metadata = base_metadata(chunk)
        content = content_text(chunk, messages_by_id)
        if content.strip():
            dedup_key = ("content_memory", normalize_for_dedup(content))
            if dedup_key not in seen:
                seen.add(dedup_key)
                content_metadata = {
                    **metadata,
                    "view": "content",
                    "collection": "content_memory",
                    "source_scope": "content",
                    "speaker_filter": "self_and_context",
                    "style_owner": "none",
                }
                documents.append(
                    {
                        "document_id": f"{chunk['window_id']}_content",
                        "chunk_id": chunk["window_id"],
                        "view": "content",
                        "collection": "content_memory",
                        "text": content,
                        "metadata": content_metadata,
                    }
                )

        style = style_text(chunk, messages_by_id)
        if style.strip() and chunk["self_message_ids"]:
            if any(
                messages_by_id[message_id]["speaker"] != "self"
                for message_id in chunk["self_message_ids"]
            ):
                raise ValueError(f"Style chunk {chunk['window_id']} contains non-self messages")
            dedup_key = ("style_memory", normalize_for_dedup(style))
            if dedup_key not in seen:
                seen.add(dedup_key)
                style_metadata = {
                    **metadata,
                    "view": "style",
                    "collection": "style_memory",
                    "source_scope": "style",
                    "speaker_filter": "self_only",
                    "style_owner": "self",
                }
                documents.append(
                    {
                        "document_id": f"{chunk['window_id']}_style",
                        "chunk_id": chunk["window_id"],
                        "view": "style",
                        "collection": "style_memory",
                        "text": style,
                        "metadata": style_metadata,
                    }
                )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default="data/normalized/messages.json")
    parser.add_argument("--chunks", default="data/windows/candidate_windows.json")
    parser.add_argument("--output", default="data/chroma_docs/chroma_documents.json")
    args = parser.parse_args()

    documents = build_documents(read_json(args.chunks), read_json(args.messages))
    write_json(args.output, documents)
    counts = {collection: sum(document["collection"] == collection for document in documents)
              for collection in ("content_memory", "style_memory")}
    print(f"Retrieval documents: {len(documents)}")
    print(f"Content documents: {counts['content_memory']}")
    print(f"Style documents: {counts['style_memory']}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
