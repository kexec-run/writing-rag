"""Hydrate retrieval results into bounded, hash-checked source packets.

Retrieval documents are locator records.  This module resolves those records
to the smallest source representation that can be used by a narrative plan:
message-backed windows retain canonical message spans, while document-backed
sources such as MDX retain one exact document span.  It never performs a new
retrieval and it fails closed when a message-backed record cannot be resolved.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


MESSAGE_BACKED_SOURCES = {"instagram", "whatsapp", "reddit", "twitter"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_from_result(item: Mapping[str, Any]) -> Mapping[str, Any]:
    document = item.get("document")
    return document if isinstance(document, Mapping) else item


def document_id_from_result(item: Mapping[str, Any]) -> str:
    document = document_from_result(item)
    return str(document.get("document_id") or item.get("candidate_id") or "")


def metadata_message_ids(item: Mapping[str, Any]) -> list[str]:
    document = document_from_result(item)
    metadata = document.get("metadata", {})
    value = metadata.get("message_ids", "") if isinstance(metadata, Mapping) else ""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part) for part in value if str(part)]
    return []


def hydrate_message_ids(
    item: Mapping[str, Any],
    canonical_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Resolve direct IDs plus the recorded source-thread/time window."""

    document = document_from_result(item)
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    requested = metadata_message_ids(item)
    source_thread_id = str(metadata.get("source_thread_id", ""))
    start = int(metadata.get("start_timestamp_ms", 0) or 0)
    end = int(metadata.get("end_timestamp_ms", 0) or 0)
    if source_thread_id and end >= start:
        requested.extend(
            message_id
            for message_id, message in canonical_by_id.items()
            if str(message.get("source_thread_id", "")) == source_thread_id
            and start <= int(message.get("timestamp_ms", 0) or 0) <= end
        )
    requested = list(dict.fromkeys(str(value) for value in requested if str(value)))
    unresolved = [message_id for message_id in requested if message_id not in canonical_by_id]
    return [message_id for message_id in requested if message_id in canonical_by_id], unresolved


def _message_span(document_id: str, message: Mapping[str, Any]) -> dict[str, Any]:
    message_id = str(message.get("message_id", ""))
    text = str(message.get("content", ""))
    span = {
        "span_id": f"{document_id}:message:{message_id}",
        "message_id": message_id,
        "text": text,
        "sha256": sha256_text(text),
        "source": str(message.get("source", "")),
        "source_path": str(message.get("source_path", "")),
        "raw_message_id": str(message.get("raw_message_id", "")),
        "timestamp_ms": int(message.get("timestamp_ms", 0) or 0),
        "speaker": str(message.get("speaker", "unknown")),
    }
    # These are canonical relations, not relations inferred from the hydrated
    # window.  Keep the fields absent when the canonical record is absent so a
    # downstream selector cannot mistake a default for an observed relation.
    for key in ("source_thread_id", "reply_to_message_id"):
        if key in message:
            span[key] = message[key]
    return span


def _document_span(document_id: str, document: Mapping[str, Any]) -> dict[str, Any]:
    text = str(document.get("text", ""))
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "span_id": f"{document_id}:document:0001",
        "document_id": document_id,
        "text": text,
        "sha256": sha256_text(text),
        "source": str(metadata.get("source", "")),
        "source_file": str(metadata.get("source_file", "")),
        "heading": metadata.get("heading"),
    }


def hydrate_retrieval_documents(
    results: list[Mapping[str, Any]],
    canonical_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return source packets and deterministic hydration diagnostics.

    A message-backed document with no resolvable canonical IDs is an error.
    A non-message document with a source scope is retained as a document-backed
    packet, which is the path used for MDX evidence.
    """

    canonical_by_id = canonical_by_id or {}
    packets: list[dict[str, Any]] = []
    message_ids_by_document: dict[str, list[str]] = {}
    errors: list[str] = []
    seen_document_ids: set[str] = set()

    for item in results:
        document = document_from_result(item)
        document_id = document_id_from_result(item)
        if not document_id:
            errors.append("retrieval result has no document_id")
            continue
        if document_id in seen_document_ids:
            errors.append(f"duplicate retrieval document_id: {document_id}")
            continue
        seen_document_ids.add(document_id)
        metadata = document.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        source = str(metadata.get("source", ""))
        source_scope = str(metadata.get("source_scope", ""))
        collection = str(
            document.get("collection")
            or metadata.get("collection", "")
        )
        if (
            collection == "style_memory"
            or source == "style"
            or source_scope == "style_memory"
            or source_scope.startswith("style_")
        ):
            errors.append(f"{document_id}: style source cannot enter article plan")
            continue
        message_ids, unresolved = hydrate_message_ids(item, canonical_by_id)
        direct_message_ids = metadata_message_ids(item)
        is_message_backed = bool(direct_message_ids or source in MESSAGE_BACKED_SOURCES)
        if unresolved:
            errors.extend(
                f"{document_id}: unresolved message ID {message_id}"
                for message_id in unresolved
            )
        if is_message_backed:
            message_ids_by_document[document_id] = message_ids
            if not message_ids:
                errors.append(f"{document_id}: message-backed source has no canonical messages")
                continue
            ordered_messages = sorted(
                (canonical_by_id[message_id] for message_id in message_ids),
                key=lambda message: (
                    int(message.get("timestamp_ms", 0) or 0),
                    str(message.get("message_id", "")),
                ),
            )
            packets.append(
                {
                    "packet_id": f"packet:{document_id}",
                    "document_id": document_id,
                    "source_document_id": document_id,
                    "source_family": "message",
                    "source_scope": source_scope or source,
                    "text": str(document.get("text", "")),
                    "text_sha256": sha256_text(str(document.get("text", ""))),
                    "message_ids": [
                        str(message.get("message_id", "")) for message in ordered_messages
                    ],
                    "source_spans": [
                        _message_span(document_id, message) for message in ordered_messages
                    ],
                    "metadata": dict(metadata),
                }
            )
            continue

        if not source_scope and source != "mdx":
            errors.append(f"{document_id}: document has no resolvable source scope")
            continue
        span = _document_span(document_id, document)
        packets.append(
            {
                "packet_id": f"packet:{document_id}",
                "document_id": document_id,
                "source_document_id": document_id,
                "source_family": "document",
                "source_scope": source_scope or source,
                "text": str(document.get("text", "")),
                "text_sha256": sha256_text(str(document.get("text", ""))),
                "message_ids": [],
                "source_spans": [span],
                "metadata": dict(metadata),
            }
        )

    return {
        "schema_version": "source-hydration-v1",
        "status": "ready" if not errors else "failed",
        "packets": packets,
        "message_ids_by_document": message_ids_by_document,
        "errors": sorted(set(errors)),
    }
