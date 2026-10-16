"""Compile composition plans into O-flat scaffold text (semantic obligations only)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from writing_rag.plan.obligation_set import build_obligation_set

FORBIDDEN_GATE_PHRASES = (
    re.compile(r"continue until", re.IGNORECASE),
    re.compile(r"keep generating until", re.IGNORECASE),
    re.compile(r"gate closes", re.IGNORECASE),
    re.compile(r"do not stop until", re.IGNORECASE),
    re.compile(r"until this gate", re.IGNORECASE),
)


def validate_oflat_text(text: str) -> list[str]:
    errors: list[str] = []
    for pattern in FORBIDDEN_GATE_PHRASES:
        if pattern.search(text):
            errors.append(
                f"O-flat contains forbidden execution directive: {pattern.pattern}"
            )
    return sorted(set(errors))


def _obligation_line(obligation: Mapping[str, Any]) -> str:
    obligation_id = str(obligation.get("obligation_id", ""))
    kind = str(obligation.get("kind", ""))
    authority = str(obligation.get("authority", ""))
    binds = obligation.get("binds", {})
    target = ""
    if isinstance(binds, Mapping):
        for key in ("shape_id", "beat_id", "concept_id", "relation_id"):
            if binds.get(key):
                target = f"{key}={binds[key]}"
                break
    purpose = str(obligation.get("purpose", obligation.get("description", ""))).strip()
    line = f"- [{obligation_id}] {kind}"
    if authority:
        line += f" ({authority})"
    if target:
        line += f" {target}"
    if purpose:
        line += f": {purpose}"
    return line


def compile_oflat(plan: Mapping[str, Any]) -> dict[str, Any]:
    obligation_set = build_obligation_set(plan)
    obligation_ids = list(obligation_set.get("obligation_ids", []))
    lines = [
        "AUTHOR-SUPPLIED NARRATIVE SCAFFOLD",
        "Scaffold condition: O-flat",
        (
            "The scaffold supplies narrative direction and semantic obligations. "
            "It is not factual evidence."
        ),
        "",
        f"Thesis: {plan.get('thesis', '')}",
        "",
        "Semantic obligations:",
    ]
    for obligation in obligation_set.get("obligations", []):
        lines.append(_obligation_line(obligation))
    lines.extend(
        [
            "",
            "Shape progression:",
        ]
    )
    for shape in sorted(
        plan.get("shapes", []), key=lambda item: int(item.get("order", 0))
    ):
        shape_id = str(shape.get("id", ""))
        purpose = str(shape.get("purpose", ""))
        entry = shape.get("entry", {})
        exit_block = shape.get("exit", {})
        lines.append(f"- {shape_id}: {purpose}")
        if isinstance(entry, Mapping) and entry.get("required"):
            lines.append(f"  entry requires: {', '.join(entry.get('required', []))}")
        if isinstance(exit_block, Mapping) and exit_block.get("required"):
            lines.append(
                f"  exit requires: {', '.join(exit_block.get('required', []))}"
            )
        if isinstance(exit_block, Mapping) and exit_block.get("forbidden"):
            lines.append(
                f"  exit forbids: {', '.join(exit_block.get('forbidden', []))}"
            )
        for relation in shape.get("relations", []):
            lines.append(
                f"  {relation.get('from')} --{relation.get('relation')}--> {relation.get('to')}"
            )
    scaffold_text = "\n".join(lines).strip() + "\n"
    gate_errors = validate_oflat_text(scaffold_text)
    if gate_errors:
        raise ValueError("; ".join(gate_errors))
    manifest = {
        "compiler": "oflat",
        "rendered_obligation_ids": obligation_ids,
        "obligation_count": len(obligation_ids),
    }
    return {
        "scaffold_text": scaffold_text,
        "rendered_obligation_ids": obligation_ids,
        "manifest": manifest,
    }
