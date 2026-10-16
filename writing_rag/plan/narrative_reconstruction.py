"""Build and validate a provenance-bearing narrative map.

This module is deliberately source-first.  It does not ask a model to invent
an episode, a causal edge, or a scaffold beat.  Time, reply/context links,
lexical overlap, and dialogue-role heuristics may identify candidates, but
only source text containing an explicit causal marker can produce an explicit
causal edge.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping


CAUSAL_RE = re.compile(
    r"\b(because|therefore|that's why|that is why|which means|implies|since)\b",
    re.IGNORECASE,
)

SELF_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("question", re.compile(r"\?|^\s*(why|what|how|does|can|could)\b", re.I)),
    ("reframe", re.compile(r"\b(actually|rather|instead|i realized|i now think)\b", re.I)),
    ("experiment", re.compile(r"\b(test|tested|compare|compared|measured|checked|ran)\b", re.I)),
    ("finding", re.compile(r"\b(found|showed|result|difference|residual|observed)\b", re.I)),
    ("decision", re.compile(r"\b(so i|i chose|i decided|the practical choice)\b", re.I)),
    ("conclusion", re.compile(r"\b(therefore|in the end|finally|conclusion)\b", re.I)),
)


def _message_id(message: Mapping[str, Any]) -> str:
    return str(message.get("message_id", ""))


def _source_span(message: Mapping[str, Any]) -> dict[str, Any]:
    text = str(message.get("content", ""))
    return {
        "message_id": _message_id(message),
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source": str(message.get("source", "")),
        "source_path": str(message.get("source_path", "")),
        "raw_message_id": str(message.get("raw_message_id", "")),
    }


def _self_role(text: str) -> tuple[str, str]:
    for role, pattern in SELF_ROLE_PATTERNS:
        if pattern.search(text):
            return role, "medium"
    return "observation", "low"


def _context_role(message_id: str, context_roles: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    record = context_roles.get(message_id, {})
    roles = [str(value) for value in record.get("roles", []) if value]
    if roles:
        return roles[0], str(record.get("confidence", "low"))
    return "context", "low"


def _edge(
    *,
    edge_id: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    edge_type: str,
    explicit: bool,
    confidence: float,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    left_span = _source_span(left)
    right_span = _source_span(right)
    return {
        "edge_id": edge_id,
        "from_node_id": f"message:{_message_id(left)}",
        "to_node_id": f"message:{_message_id(right)}",
        "edge_type": edge_type,
        "explicit": explicit,
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "reasons": list(reasons or []),
        "source_spans": [left_span, right_span],
    }


def build_narrative_map(
    messages_by_id: Mapping[str, Mapping[str, Any]],
    *,
    message_ids: list[str] | None = None,
    episode: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic map from an already selected source episode.

    ``episode`` may contain the existing context summaries and context links.
    Those links are retained as inferred relations.  The function never
    expands the episode beyond ``message_ids`` or the episode's own IDs.
    """

    selected = [str(value) for value in (message_ids or (episode or {}).get("message_ids", []))]
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate message IDs cannot form a narrative map")
    missing = [value for value in selected if value not in messages_by_id]
    if missing:
        raise ValueError(f"narrative source messages missing: {missing[:5]}")

    ordered = sorted(
        (messages_by_id[value] for value in selected),
        key=lambda item: (int(item.get("timestamp_ms", 0)), _message_id(item)),
    )
    context_roles = {
        str(record.get("message_id")): record
        for record in (episode or {}).get("context_roles", [])
        if record.get("message_id")
    }
    nodes: list[dict[str, Any]] = []
    self_ids: list[str] = []
    context_ids: list[str] = []
    for message in ordered:
        message_id = _message_id(message)
        speaker = str(message.get("speaker", "unknown"))
        if speaker == "self":
            role, confidence = _self_role(str(message.get("content", "")))
            self_ids.append(message_id)
        else:
            role, confidence = _context_role(message_id, context_roles)
            context_ids.append(message_id)
        nodes.append(
            {
                "node_id": f"message:{message_id}",
                "message_id": message_id,
                "speaker": speaker,
                "timestamp_ms": int(message.get("timestamp_ms", 0)),
                "role": role,
                "role_confidence": confidence,
                "text": str(message.get("content", "")),
                "source_spans": [_source_span(message)],
            }
        )

    by_id = {_message_id(message): message for message in ordered}
    edges: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(ordered, ordered[1:]), 1):
        edges.append(
            _edge(
                edge_id=f"temporal:{index:04d}",
                left=left,
                right=right,
                edge_type="temporal",
                explicit=True,
                confidence=1.0,
                reasons=["canonical_timestamp_order"],
            )
        )
        combined = f"{left.get('content', '')}\n{right.get('content', '')}"

    self_messages = [message for message in ordered if message.get("speaker") == "self"]
    for index, (left, right) in enumerate(zip(self_messages, self_messages[1:]), 1):
        combined = f"{left.get('content', '')}\n{right.get('content', '')}"
        if CAUSAL_RE.search(combined):
            edges.append(
                _edge(
                    edge_id=f"causal:{index:04d}",
                    left=left,
                    right=right,
                    edge_type="explicit_causal",
                    explicit=True,
                    confidence=1.0,
                    reasons=["explicit_causal_marker", "consecutive_self_messages"],
                )
            )

    for index, link in enumerate((episode or {}).get("context_links", []), 1):
        context_id = str(link.get("context_message_id", ""))
        self_id = str(link.get("self_message_id", ""))
        if context_id not in by_id or self_id not in by_id:
            continue
        features = link.get("features", {})
        reasons = [str(value) for value in features.get("reasons", [])]
        confidence = float(features.get("continuity_score", 0.0))
        edges.append(
            _edge(
                edge_id=f"context:{index:04d}",
                left=by_id[context_id],
                right=by_id[self_id],
                edge_type="interaction_context",
                explicit=False,
                confidence=confidence,
                reasons=reasons or ["heuristic_context_link"],
            )
        )

    return {
        "schema_version": "narrative-map-v1",
        "episode_id": str((episode or {}).get("episode_id", "")),
        "message_ids": [_message_id(message) for message in ordered],
        "self_message_ids": self_ids,
        "context_message_ids": context_ids,
        "nodes": nodes,
        "edges": edges,
        "style_policy": "self_messages_only",
        "causal_policy": "explicit_source_marker_or_qualified_inference",
    }


def validate_narrative_map(
    narrative_map: Mapping[str, Any],
    messages_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return invariant failures for a narrative map."""

    errors: list[str] = []
    message_ids = [str(value) for value in narrative_map.get("message_ids", [])]
    if len(message_ids) != len(set(message_ids)):
        errors.append("duplicate_message_ids")
    if any(value not in messages_by_id for value in message_ids):
        errors.append("missing_source_message")
    self_ids = {str(value) for value in narrative_map.get("self_message_ids", [])}
    context_ids = {str(value) for value in narrative_map.get("context_message_ids", [])}
    if self_ids & context_ids:
        errors.append("self_context_overlap")
    if not self_ids | context_ids <= set(message_ids):
        errors.append("projection_outside_episode")

    nodes = narrative_map.get("nodes", [])
    node_ids = [str(node.get("node_id", "")) for node in nodes]
    if set(node_ids) != {f"message:{value}" for value in message_ids}:
        errors.append("node_projection_mismatch")
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    timestamps = [int(node.get("timestamp_ms", 0)) for node in nodes]
    if timestamps != sorted(timestamps):
        errors.append("non_chronological_nodes")

    for node in nodes:
        message_id = str(node.get("message_id", ""))
        source = messages_by_id.get(message_id)
        if source is None:
            continue
        spans = node.get("source_spans", [])
        if len(spans) != 1 or spans[0].get("text") != str(source.get("content", "")):
            errors.append(f"source_text_mismatch:{message_id}")
        expected_hash = hashlib.sha256(str(source.get("content", "")).encode("utf-8")).hexdigest()
        if not spans or spans[0].get("sha256") != expected_hash:
            errors.append(f"source_hash_mismatch:{message_id}")
        if message_id in context_ids and message_id in set(narrative_map.get("self_message_ids", [])):
            errors.append(f"style_contamination:{message_id}")

    for edge in narrative_map.get("edges", []):
        from_id = str(edge.get("from_node_id", ""))
        to_id = str(edge.get("to_node_id", ""))
        if from_id not in node_by_id or to_id not in node_by_id:
            errors.append(f"edge_endpoint_missing:{edge.get('edge_id', '')}")
        confidence = edge.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"edge_confidence_invalid:{edge.get('edge_id', '')}")
        if edge.get("edge_type") == "explicit_causal":
            if edge.get("explicit") is not True:
                errors.append(f"causal_edge_not_explicit:{edge.get('edge_id', '')}")
            source_text = "\n".join(str(span.get("text", "")) for span in edge.get("source_spans", []))
            if not CAUSAL_RE.search(source_text):
                errors.append(f"causal_edge_without_marker:{edge.get('edge_id', '')}")
        if edge.get("edge_type") == "interaction_context" and edge.get("explicit") is True:
            errors.append(f"inferred_context_marked_explicit:{edge.get('edge_id', '')}")
        if not edge.get("source_spans"):
            errors.append(f"edge_without_source_spans:{edge.get('edge_id', '')}")

    return sorted(set(errors))


def narrative_metrics(narrative_map: Mapping[str, Any], errors: list[str] | None = None) -> dict[str, Any]:
    """Return compact metrics for the architectural experiment."""

    nodes = list(narrative_map.get("nodes", []))
    edges = list(narrative_map.get("edges", []))
    explicit_causal = [edge for edge in edges if edge.get("edge_type") == "explicit_causal"]
    inferred = [edge for edge in edges if edge.get("explicit") is False]
    return {
        "node_count": len(nodes),
        "self_node_count": sum(node.get("speaker") == "self" for node in nodes),
        "context_node_count": sum(node.get("speaker") != "self" for node in nodes),
        "edge_count": len(edges),
        "explicit_causal_edge_count": len(explicit_causal),
        "inferred_edge_count": len(inferred),
        "source_span_coverage": sum(bool(node.get("source_spans")) for node in nodes) / max(1, len(nodes)),
        "validation_passed": not errors if errors is not None else None,
        "validation_errors": list(errors or []),
    }


def build_slot_packets(
    slots: list[Mapping[str, Any]],
    narrative_map: Mapping[str, Any],
    assignments: Mapping[str, Mapping[str, list[str]]],
) -> list[dict[str, Any]]:
    """Build bounded slot packets from explicit node/edge assignments.

    Assignments are intentionally caller-provided.  This prevents a model or
    lexical heuristic from silently creating a causal relationship while
    converting a scaffold into article context.
    """

    nodes = {str(node["node_id"]): node for node in narrative_map.get("nodes", [])}
    edges = {str(edge["edge_id"]): edge for edge in narrative_map.get("edges", [])}
    packets: list[dict[str, Any]] = []
    for slot in slots:
        slot_id = str(slot.get("slot_id", ""))
        assignment = assignments.get(slot_id, {})
        node_ids = [str(value) for value in assignment.get("node_ids", [])]
        edge_ids = [str(value) for value in assignment.get("edge_ids", [])]
        unknown_nodes = [value for value in node_ids if value not in nodes]
        unknown_edges = [value for value in edge_ids if value not in edges]
        if unknown_nodes or unknown_edges:
            raise ValueError(f"slot {slot_id} escaped narrative map: {unknown_nodes + unknown_edges}")
        source_ids = sorted(
            {
                str(span.get("message_id"))
                for node_id in node_ids
                for span in nodes[node_id].get("source_spans", [])
            }
            | {
                str(span.get("message_id"))
                for edge_id in edge_ids
                for span in edges[edge_id].get("source_spans", [])
            }
        )
        packets.append(
            {
                "slot_id": slot_id,
                "role": str(slot.get("role", slot.get("label", ""))),
                "node_ids": node_ids,
                "edge_ids": edge_ids,
                "source_allowlist": source_ids,
                "nodes": [nodes[value] for value in node_ids],
                "edges": [edges[value] for value in edge_ids],
                "support_status": "supported" if node_ids or edge_ids else "missing",
            }
        )
    return packets


def format_narrative_packets(
    packets: list[Mapping[str, Any]], *, max_chars: int = 12_000
) -> str:
    """Format bounded narrative packets as generation evidence."""

    lines: list[str] = []
    for packet in packets:
        lines.append(f"### Slot {packet.get('slot_id', '')}: {packet.get('role', '')}")
        lines.append(f"Support status: {packet.get('support_status', 'missing')}")
        lines.append(f"Source allowlist: {', '.join(packet.get('source_allowlist', [])) or 'none'}")
        for node in packet.get("nodes", []):
            lines.append(
                f"- NODE {node.get('node_id')}; speaker={node.get('speaker')}; "
                f"role={node.get('role')}; text={node.get('text', '')}"
            )
        for edge in packet.get("edges", []):
            relation = "explicit" if edge.get("explicit") else "inferred"
            lines.append(
                f"- EDGE {edge.get('edge_id')}; type={edge.get('edge_type')}; "
                f"status={relation}; from={edge.get('from_node_id')}; to={edge.get('to_node_id')}"
            )
    formatted = "\n".join(lines) if lines else "No source-backed narrative packets were available."
    if len(formatted) <= max_chars:
        return formatted
    return formatted[: max_chars - 80].rstrip() + "\n[ narrative packet truncated by budget ]"
