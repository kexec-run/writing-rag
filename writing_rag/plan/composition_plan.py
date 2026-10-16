"""Shape–Beat composition plan schema and validation (composition-plan-v0)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from writing_rag.plan.article_plan import AUTHORITIES

SCHEMA_VERSION = "composition-plan-v0"

RELATION_VOCABULARY = frozenset(
    {
        "motivates",
        "sets_up",
        "requires",
        "contrasts_with",
        "eliminates",
        "reframes",
        "supports",
        "explains",
        "causes",
        "demonstrates",
        "justifies",
        "resolves",
    }
)

OBLIGATION_KINDS = frozenset({"establish", "preserve", "forbid", "witness"})

_BEAT_ID_RE = re.compile(r"^B\d{2}-\d{2}$")
_SHAPE_ID_RE = re.compile(r"^S\d{2}$")


def composition_plan_sha256(plan: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _normalize_depends_on(values: list[str] | None) -> list[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def apply_beat_contract_defaults(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize beat depends_on within each shape when omitted."""

    beats_by_shape: dict[str, list[dict[str, Any]]] = {}
    for beat in plan.get("beats", []):
        shape_id = str(beat.get("shape_id", ""))
        beats_by_shape.setdefault(shape_id, []).append(beat)
    for shape_id, shape_beats in beats_by_shape.items():
        ordered = sorted(
            shape_beats,
            key=lambda item: (int(item.get("order", 0)), str(item.get("id", ""))),
        )
        for index, beat in enumerate(ordered):
            beat["depends_on"] = _normalize_depends_on(beat.get("depends_on"))
            if index and not beat["depends_on"]:
                beat["depends_on"] = [str(ordered[index - 1]["id"])]
    return plan


def validate_composition_plan(plan: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported composition plan schema")
    if plan.get("plan_sha256") != composition_plan_sha256(plan):
        errors.append("composition plan hash mismatch")

    shapes = list(plan.get("shapes", []))
    beats = list(plan.get("beats", []))
    obligations = list(plan.get("evidence_obligations", []))

    shape_ids: list[str] = []
    shape_orders: list[int] = []
    for shape in shapes:
        shape_id = str(shape.get("id", ""))
        if not shape_id or not _SHAPE_ID_RE.fullmatch(shape_id):
            errors.append(f"invalid shape id: {shape_id}")
        if shape_id in shape_ids:
            errors.append(f"duplicate shape id: {shape_id}")
        shape_ids.append(shape_id)
        try:
            shape_orders.append(int(shape.get("order", -1)))
        except (TypeError, ValueError):
            errors.append(f"invalid shape order: {shape_id}")
        for relation in shape.get("relations", []):
            rel_type = str(relation.get("relation", "")).casefold()
            if rel_type not in RELATION_VOCABULARY:
                errors.append(f"invalid shape relation: {shape_id} -> {rel_type}")
            if str(relation.get("from", "")) != shape_id:
                errors.append(f"shape relation from mismatch: {shape_id}")

    if shape_ids and len(shape_orders) != len(set(shape_orders)):
        errors.append("duplicate shape order values")

    beat_ids: list[str] = []
    beat_by_id: dict[str, dict[str, Any]] = {}
    for beat in beats:
        beat_id = str(beat.get("id", ""))
        shape_id = str(beat.get("shape_id", ""))
        if not beat_id or not _BEAT_ID_RE.fullmatch(beat_id):
            errors.append(f"invalid beat id: {beat_id}")
        if beat_id in beat_ids:
            errors.append(f"duplicate beat id: {beat_id}")
        beat_ids.append(beat_id)
        beat_by_id[beat_id] = beat
        if shape_id not in shape_ids:
            errors.append(f"beat references missing shape: {beat_id} -> {shape_id}")

    for beat in beats:
        beat_id = str(beat.get("id", ""))
        for dependency in _normalize_depends_on(beat.get("depends_on")):
            if dependency not in beat_by_id:
                errors.append(
                    f"beat depends on missing unit: {beat_id} -> {dependency}"
                )
            elif beat_ids.index(dependency) >= beat_ids.index(beat_id):
                errors.append(f"beat depends on later unit: {beat_id} -> {dependency}")

    if beat_ids:
        adjacency = {
            beat_id: _normalize_depends_on(beat_by_id[beat_id].get("depends_on"))
            for beat_id in beat_ids
        }
        visited: set[str] = set()
        active: set[str] = set()

        def _visit(node: str) -> bool:
            if node in active:
                errors.append(f"beat dependency cycle involving: {node}")
                return True
            if node in visited:
                return False
            visited.add(node)
            active.add(node)
            for dependency in adjacency.get(node, []):
                if _visit(dependency):
                    return True
            active.remove(node)
            return False

        for beat_id in beat_ids:
            if beat_id not in visited:
                _visit(beat_id)

    obligation_ids: set[str] = set()
    for obligation in obligations:
        obligation_id = str(obligation.get("obligation_id", ""))
        if not obligation_id:
            errors.append("missing obligation_id")
            continue
        if obligation_id in obligation_ids:
            errors.append(f"duplicate obligation id: {obligation_id}")
        obligation_ids.add(obligation_id)
        kind = str(obligation.get("kind", ""))
        if kind not in OBLIGATION_KINDS:
            errors.append(f"invalid obligation kind: {obligation_id}")
        authority = str(obligation.get("authority", ""))
        if authority and authority not in AUTHORITIES:
            errors.append(f"unknown obligation authority: {obligation_id}")
        witness = obligation.get("witness", {})
        if authority == "source_backed":
            if not isinstance(witness, Mapping):
                errors.append(
                    f"source_backed obligation missing witness: {obligation_id}"
                )
            elif not any(
                witness.get(key) for key in ("document_id", "span_id", "edge_id")
            ):
                errors.append(
                    f"source_backed obligation missing witness id: {obligation_id}"
                )
        if authority == "author_asserted" and (
            not isinstance(witness, Mapping) or not witness.get("author_assertion_id")
        ):
            errors.append(
                f"author_asserted obligation missing assertion id: {obligation_id}"
            )

    return sorted(set(errors))


def shape_by_id(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(shape["id"]): dict(shape)
        for shape in plan.get("shapes", [])
        if shape.get("id")
    }


def beats_for_shape(plan: Mapping[str, Any], shape_id: str) -> list[dict[str, Any]]:
    return [
        dict(beat)
        for beat in plan.get("beats", [])
        if str(beat.get("shape_id", "")) == shape_id
    ]
