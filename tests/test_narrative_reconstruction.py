from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "writing_rag/plan/narrative_reconstruction.py"
SHIM_PATH = ROOT / "scripts/narrative_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("narrative_reconstruction", MODULE_PATH)
assert SPEC and SPEC.loader
narrative = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(narrative)

sys.path.insert(0, str(ROOT / "scripts"))
from generate import build_generation_context  # noqa: E402


def _messages() -> dict[str, dict[str, object]]:
    return {
        "self-1": {
            "message_id": "self-1",
            "source": "instagram",
            "source_path": "fixture",
            "raw_message_id": "raw:self-1",
            "timestamp_ms": 1,
            "speaker": "self",
            "content": "I noticed the image was different.",
        },
        "other-1": {
            "message_id": "other-1",
            "source": "instagram",
            "source_path": "fixture",
            "raw_message_id": "raw:other-1",
            "timestamp_ms": 2,
            "speaker": "other",
            "content": "Why did you compare those two?",
        },
        "self-2": {
            "message_id": "self-2",
            "source": "instagram",
            "source_path": "fixture",
            "raw_message_id": "raw:self-2",
            "timestamp_ms": 3,
            "speaker": "self",
            "content": "Because the output had been resized.",
        },
    }


def _episode() -> dict[str, object]:
    return {
        "episode_id": "fixture:w2",
        "message_ids": ["self-1", "other-1", "self-2"],
        "context_roles": [
            {
                "message_id": "other-1",
                "roles": ["question"],
                "confidence": "high",
            }
        ],
        "context_links": [
            {
                "context_message_id": "other-1",
                "self_message_id": "self-2",
                "features": {"continuity_score": 0.8, "reasons": ["question_response_candidate"]},
            }
        ],
    }


def test_positive_map_keeps_context_style_and_explicit_causality_separate():
    messages = _messages()
    result = narrative.build_narrative_map(messages, episode=_episode())
    errors = narrative.validate_narrative_map(result, messages)

    assert errors == []
    assert result["self_message_ids"] == ["self-1", "self-2"]
    assert result["context_message_ids"] == ["other-1"]
    assert any(edge["edge_type"] == "explicit_causal" for edge in result["edges"])
    assert any(edge["edge_type"] == "interaction_context" and not edge["explicit"] for edge in result["edges"])
    assert narrative.narrative_metrics(result, errors)["source_span_coverage"] == 1.0


def test_positive_slot_packet_uses_only_explicit_assignments():
    messages = _messages()
    result = narrative.build_narrative_map(messages, episode=_episode())
    causal_id = next(edge["edge_id"] for edge in result["edges"] if edge["edge_type"] == "explicit_causal")
    packet = narrative.build_slot_packets(
        [{"slot_id": "slot-1", "role": "finding"}],
        result,
        {"slot-1": {"node_ids": ["message:self-1", "message:self-2"], "edge_ids": [causal_id]}},
    )[0]
    assert packet["support_status"] == "supported"
    assert packet["source_allowlist"] == ["self-1", "self-2"]


def test_negative_source_mutation_fails_hash_validation():
    messages = _messages()
    result = narrative.build_narrative_map(messages, episode=_episode())
    messages["self-2"]["content"] = "Because the output was unchanged."
    errors = narrative.validate_narrative_map(result, messages)
    assert "source_text_mismatch:self-2" in errors
    assert "source_hash_mismatch:self-2" in errors


def test_negative_inferred_context_cannot_be_marked_explicit():
    messages = _messages()
    result = narrative.build_narrative_map(messages, episode=_episode())
    context_edge = next(edge for edge in result["edges"] if edge["edge_type"] == "interaction_context")
    context_edge["explicit"] = True
    assert any("inferred_context_marked_explicit" in error for error in narrative.validate_narrative_map(result, messages))


def test_negative_slot_assignment_cannot_escape_map():
    messages = _messages()
    result = narrative.build_narrative_map(messages, episode=_episode())
    try:
        narrative.build_slot_packets(
            [{"slot_id": "slot-1", "role": "finding"}],
            result,
            {"slot-1": {"node_ids": ["message:not-in-episode"], "edge_ids": []}},
        )
    except ValueError as exc:
        assert "escaped narrative map" in str(exc)
    else:
        raise AssertionError("escaped slot assignment was accepted")


def test_generation_context_receives_narrative_packets_as_separate_evidence():
    context = build_generation_context(
        "fixture query",
        [
            {
                    "document": {
                        "document_id": "doc-1",
                        "text": "content",
                        "view": "content",
                        "collection": "content_memory",
                        "metadata": {"source_scope": "content", "style_owner": "none"},
                    },
                    "score": 1.0,
                }
        ],
        "write from evidence",
        narrative_packets=[
            {
                "slot_id": "slot-1",
                "role": "finding",
                "support_status": "supported",
                "source_allowlist": ["self-1"],
                "nodes": [
                    {
                        "node_id": "message:self-1",
                        "speaker": "self",
                        "role": "finding",
                        "text": "The output changed.",
                    }
                ],
                "edges": [
                    {
                        "edge_id": "context:1",
                        "edge_type": "interaction_context",
                        "explicit": False,
                        "from_node_id": "message:other-1",
                        "to_node_id": "message:self-1",
                    }
                ],
            }
        ],
    )
    assert "## Source-backed narrative packets" in context
    assert "status=inferred" in context
    assert "The output changed." in context
