"""Validate structural and source-boundary invariants of pipeline artifacts."""

from __future__ import annotations

import argparse
from typing import Any

from pipeline_common import PROJECT_ROOT, message_sort_key, read_json


BASE_MESSAGE_FIELDS = {
    "message_id",
    "source",
    "source_thread_id",
    "timestamp_ms",
    "speaker",
    "content",
    "mode",
}
SPEAKERS = {"self", "other", "system"}
CONTENT_VIEWS = {"content", "summary", "reasoning", "knowledge"}


def validate_messages(messages: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [message.get("message_id") for message in messages]
    if len(ids) != len(set(ids)):
        errors.append("canonical messages contain duplicate message_id values")
    if messages != sorted(messages, key=message_sort_key):
        errors.append("canonical messages are not in deterministic chronological order")
    for index, message in enumerate(messages):
        missing = BASE_MESSAGE_FIELDS - message.keys()
        if missing:
            errors.append(f"message {index} is missing fields: {sorted(missing)}")
        if message.get("speaker") not in SPEAKERS:
            errors.append(f"message {index} has invalid speaker {message.get('speaker')!r}")
        if not isinstance(message.get("timestamp_ms"), int):
            errors.append(f"message {index} has a non-integer timestamp_ms")
        if not str(message.get("content", "")).strip():
            errors.append(f"message {index} has empty content")
    return errors


def validate_windows(windows: list[dict[str, Any]], message_ids: set[str]) -> list[str]:
    errors: list[str] = []
    seen_window_ids: set[str] = set()
    for index, window in enumerate(windows):
        window_id = window.get("window_id")
        if window_id in seen_window_ids:
            errors.append(f"duplicate window_id {window_id}")
        seen_window_ids.add(window_id)
        ids = window.get("message_ids", [])
        if not ids:
            errors.append(f"window {index} is empty")
        if not set(ids) <= message_ids:
            errors.append(f"window {window_id} references unknown messages")
        if window.get("message_count") != len(ids):
            errors.append(f"window {window_id} has an incorrect message_count")
        if window.get("token_count", 0) <= 0:
            errors.append(f"window {window_id} has no tokens")
        if window.get("chunk_type") != "conversation":
            errors.append(f"window {window_id} is not marked as a conversation chunk")
        if "heading" not in window or window["heading"] is not None:
            errors.append(f"window {window_id} invents a heading")
        grouping = window.get("grouping", {})
        if grouping.get("method") not in {
            "conversation_structure_window",
            "graph_path_conversation_window",
        }:
            errors.append(f"window {window_id} has the wrong grouping method")
        if (
            grouping.get("method") == "graph_path_conversation_window"
            and not window.get("graph_path_id")
        ):
            errors.append(f"window {window_id} is graph-aware but has no graph_path_id")
        if window.get("turn_count", 0) <= 0 or window.get("paragraph_count", 0) <= 0:
            errors.append(f"window {window_id} lacks turn/paragraph metadata")
    return errors


def validate_chains(
    chains: list[dict[str, Any]], window_ids: set[str], message_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    seen_chain_ids: set[str] = set()
    for chain in chains:
        chain_id = chain.get("chain_id")
        if chain_id in seen_chain_ids:
            errors.append(f"duplicate chain_id {chain_id}")
        seen_chain_ids.add(chain_id)
        if chain.get("window_id") not in window_ids:
            errors.append(f"chain {chain_id} references an unknown window")
        if not set(chain.get("message_ids", [])) <= message_ids:
            errors.append(f"chain {chain_id} references unknown messages")
        scores = chain.get("scores", {})
        for name in (
            "reasoning_density",
            "continuity",
            "context_dependency",
            "article_usefulness",
            "retrieval_value",
        ):
            value = scores.get(name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(f"chain {chain_id} has invalid score {name}")
        for point in chain.get("reasoning_points", []):
            if not point.get("raw_refs"):
                errors.append(f"chain {chain_id} contains a reasoning point without raw_refs")
            if not set(ref.get("message_id") for ref in point.get("raw_refs", [])) <= message_ids:
                errors.append(f"chain {chain_id} contains an unknown raw reference")
    return errors


def validate_documents(
    documents: list[dict[str, Any]],
    messages_by_id: dict[str, dict[str, Any]],
    chain_ids: set[str] | None = None,
    chunk_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    chain_ids = chain_ids or set()
    chunk_ids = chunk_ids or set()
    for document in documents:
        document_id = document.get("document_id")
        if document_id in seen:
            errors.append(f"duplicate document_id {document_id}")
        seen.add(document_id)
        if not document.get("text", "").strip():
            errors.append(f"document {document_id} has empty text")

        collection = document.get("collection")
        view = document.get("view")
        metadata = document.get("metadata", {})
        if collection not in {"content_memory", "style_memory"}:
            errors.append(f"document {document_id} has an invalid collection")
        if view not in CONTENT_VIEWS | {"style"}:
            errors.append(f"document {document_id} has an invalid view")
        if collection == "style_memory" and view != "style":
            errors.append(f"document {document_id} routes non-style data to style_memory")
        if collection == "content_memory" and view == "style":
            errors.append(f"document {document_id} routes style data to content_memory")
        if metadata.get("collection") != collection:
            errors.append(f"document {document_id} metadata collection mismatch")

        if "chunk_id" in document:
            if document["chunk_id"] not in chunk_ids:
                errors.append(f"document {document_id} references an unknown chunk")
        elif "chain_id" in document and document["chain_id"] not in chain_ids:
            errors.append(f"document {document_id} references an unknown chain")
        else:
            errors.append(f"document {document_id} has neither chunk_id nor chain_id")

        if collection == "style_memory":
            if metadata.get("source_scope") != "style":
                errors.append(f"style document {document_id} lacks source_scope=style")
            if metadata.get("speaker_filter") != "self_only":
                errors.append(f"style document {document_id} lacks speaker_filter=self_only")
            if metadata.get("style_owner") != "self":
                errors.append(f"style document {document_id} lacks style_owner=self")
            referenced = [item for item in metadata.get("self_message_ids", "").split(",") if item]
            if not referenced or not set(referenced) <= set(messages_by_id):
                errors.append(f"style document {document_id} has invalid self message references")
            elif any(messages_by_id[item].get("speaker") != "self" for item in referenced):
                errors.append(f"style document {document_id} references non-self text")
        else:
            if metadata.get("source_scope") != "content":
                errors.append(f"content document {document_id} lacks source_scope=content")
            if metadata.get("style_owner") != "none":
                errors.append(f"content document {document_id} can substitute for style evidence")
    return errors


def validate_all(
    messages: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    chains: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[str]:
    message_ids = {message.get("message_id") for message in messages}
    messages_by_id = {message["message_id"]: message for message in messages}
    window_ids = {window.get("window_id") for window in windows}
    chain_ids = {chain.get("chain_id") for chain in chains}
    return (
        validate_messages(messages)
        + validate_windows(windows, message_ids)
        + validate_chains(chains, window_ids, message_ids)
        + validate_documents(documents, messages_by_id, chain_ids, window_ids)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default="data/normalized/messages.json")
    parser.add_argument("--windows", default="data/windows/candidate_windows.json")
    parser.add_argument("--chains", default="data/chains/reasoning_chains.json")
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    args = parser.parse_args()

    chains_path = PROJECT_ROOT / args.chains
    chains = read_json(args.chains) if chains_path.exists() else []
    errors = validate_all(
        read_json(args.messages),
        read_json(args.windows),
        chains,
        read_json(args.documents),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Pipeline artifacts passed structural validation.")


if __name__ == "__main__":
    main()
