from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from source_hydration import (  # noqa: E402
    hydrate_message_ids,
    hydrate_retrieval_documents,
    metadata_message_ids,
    sha256_text,
)


def _messages() -> dict[str, dict[str, object]]:
    return {
        "self-1": {
            "message_id": "self-1",
            "timestamp_ms": 1000,
            "speaker": "self",
            "content": "I saw the output change.",
            "source": "instagram",
            "source_thread_id": "thread-1",
        },
        "other-1": {
            "message_id": "other-1",
            "timestamp_ms": 1100,
            "speaker": "other",
            "content": "What changed?",
            "source": "instagram",
            "source_thread_id": "thread-1",
        },
        "self-2": {
            "message_id": "self-2",
            "timestamp_ms": 1200,
            "speaker": "self",
            "content": "Because the control changed, I measured it.",
            "source": "instagram",
            "source_thread_id": "thread-1",
        },
    }


def _message_result() -> dict[str, object]:
    return {
        "document": {
            "document_id": "window_001",
            "text": "context\nself text",
            "metadata": {
                "source": "instagram",
                "source_scope": "instagram_content",
                "message_ids": "self-1,other-1,self-2",
                "source_thread_id": "thread-1",
                "start_timestamp_ms": 1000,
                "end_timestamp_ms": 1200,
            },
        }
    }


def _mdx_result() -> dict[str, object]:
    return {
        "document": {
            "document_id": "mdx_demo_001_content",
            "text": "Article: demo\n\nThe control image stayed unchanged.",
            "metadata": {
                "source": "mdx",
                "source_scope": "mdx_content",
                "source_file": "data/generated/demo.mdx",
                "heading": "Result",
            },
        }
    }


def test_positive_metadata_ids_accepts_string_and_list() -> None:
    assert metadata_message_ids(_message_result()) == ["self-1", "other-1", "self-2"]
    item = _message_result()
    item["document"]["metadata"]["message_ids"] = ["self-1", "other-1"]
    assert metadata_message_ids(item) == ["self-1", "other-1"]


def test_positive_time_window_adds_canonical_messages() -> None:
    item = _message_result()
    item["document"]["metadata"]["message_ids"] = ""
    ids, unresolved = hydrate_message_ids(item, _messages())
    assert ids == ["self-1", "other-1", "self-2"]
    assert unresolved == []


def test_positive_message_result_becomes_hash_checked_packet() -> None:
    result = hydrate_retrieval_documents([_message_result()], _messages())
    assert result["status"] == "ready"
    packet = result["packets"][0]
    assert packet["source_family"] == "message"
    assert len(packet["source_spans"]) == 3
    assert packet["source_spans"][0]["sha256"] == sha256_text("I saw the output change.")


def test_positive_mdx_result_becomes_document_packet() -> None:
    result = hydrate_retrieval_documents([_mdx_result()])
    assert result["status"] == "ready"
    packet = result["packets"][0]
    assert packet["source_family"] == "document"
    assert packet["source_spans"][0]["span_id"].endswith(":document:0001")


def test_positive_mixed_sources_preserve_both_families() -> None:
    result = hydrate_retrieval_documents([_message_result(), _mdx_result()], _messages())
    assert result["status"] == "ready"
    assert [packet["source_family"] for packet in result["packets"]] == ["message", "document"]


def test_negative_unresolved_message_id_fails_closed() -> None:
    item = _message_result()
    item["document"]["metadata"]["message_ids"] = "self-1,missing"
    result = hydrate_retrieval_documents([item], _messages())
    assert result["status"] == "failed"
    assert any("missing" in error for error in result["errors"])


def test_negative_message_source_without_ids_fails_closed() -> None:
    item = _message_result()
    item["document"]["metadata"].pop("message_ids")
    item["document"]["metadata"].pop("source_thread_id")
    result = hydrate_retrieval_documents([item], {})
    assert result["status"] == "failed"
    assert any("no canonical messages" in error for error in result["errors"])


def test_negative_unknown_non_mdx_document_scope_fails_closed() -> None:
    item = {
        "document": {
            "document_id": "unknown-1",
            "text": "unscoped",
            "metadata": {"source": "fixture"},
        }
    }
    result = hydrate_retrieval_documents([item])
    assert result["status"] == "failed"
    assert "unknown-1: document has no resolvable source scope" in result["errors"]


def test_negative_duplicate_document_ids_fail_closed() -> None:
    result = hydrate_retrieval_documents([_mdx_result(), _mdx_result()])
    assert result["status"] == "failed"
    assert any("duplicate retrieval document_id" in error for error in result["errors"])


def test_negative_style_source_cannot_be_hydrated_into_article_plan() -> None:
    item = _mdx_result()
    item["document"]["collection"] = "style_memory"
    result = hydrate_retrieval_documents([item])
    assert result["status"] == "failed"
    assert "mdx_demo_001_content: style source cannot enter article plan" in result["errors"]
