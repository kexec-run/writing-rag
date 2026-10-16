"""Compile load-bearing MDX scaffolds into composition-plan-v0 oracle plans.

A load-bearing scaffold is a per-target node DAG
(``{entry, central_claim, nodes: [{id, role, claim, weight, depends_on}]}``)
produced by the earlier load-bearing coherence study. This compiler turns each
node into a Shape plus one Beat, turns every ``depends_on`` edge into a typed
shape relation from ``RELATION_VOCABULARY``, and emits mechanical relation
definitions: relation R from P to N is established iff every obligation bound
to P verifies, N's entry conditions hold, and the declared dependency edge is
respected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from writing_rag.plan.composition_plan import (
    apply_beat_contract_defaults,
    composition_plan_sha256,
    validate_composition_plan,
)

RELATION_DEFINITIONS_SCHEMA = "load-bearing-relations-v1"

ESTABLISHMENT_RULE = (
    "established iff every obligation bound to the predecessor verifies, "
    "the successor entry conditions hold, and the declared dependency edge "
    "is respected"
)

ROLE_RELATIONS = {
    "opening_question": "motivates",
    "opening_definition": "motivates",
    "thought_experiment": "sets_up",
    "concrete_case": "demonstrates",
    "contrast_case": "contrasts_with",
    "conceptual_distinction": "reframes",
    "language_mechanism": "explains",
    "mechanism": "explains",
    "philosophical_model": "supports",
    "consequence": "justifies",
    "decision": "justifies",
    "finding": "demonstrates",
    "evidence": "supports",
    "conclusion": "resolves",
}

DEFAULT_ROLE_RELATION = "sets_up"


def relation_for_role(role: str) -> str:
    return ROLE_RELATIONS.get(str(role).strip().casefold(), DEFAULT_ROLE_RELATION)


def _node_concept(node: Mapping[str, Any]) -> str:
    return f"{str(node['id']).lower()}_{str(node.get('role', 'node')).strip().casefold()}"


def _validate_nodes(nodes: list[Mapping[str, Any]]) -> None:
    ids = [str(node["id"]) for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate load-bearing node id")
    known = set(ids)
    for node in nodes:
        for dependency in node.get("depends_on", []):
            if str(dependency) not in known:
                raise ValueError(f"unknown dependency: {node['id']} -> {dependency}")
    remaining = dict(enumerate(ids))
    placed: set[str] = set()
    while remaining:
        progress = False
        for index, node_id in list(remaining.items()):
            deps = {str(value) for value in nodes[index].get("depends_on", [])}
            if deps <= placed:
                placed.add(node_id)
                del remaining[index]
                progress = True
        if not progress:
            raise ValueError(f"load-bearing dependency cycle involving: {sorted(set(remaining.values()))}")


def _topological_order(nodes: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    order: list[Mapping[str, Any]] = []
    placed: set[str] = set()
    pending = list(nodes)
    while pending:
        ready = [
            node
            for node in pending
            if {str(dep) for dep in node.get("depends_on", [])} <= placed
        ]
        if not ready:
            raise ValueError("load-bearing dependency cycle")
        for node in ready:
            order.append(node)
            placed.add(str(node["id"]))
        pending = [node for node in pending if node not in ready]
    return order


def compile_load_bearing_plan(
    target_id: str,
    scaffold: Mapping[str, Any],
    *,
    plan_id: str | None = None,
) -> dict[str, Any]:
    """Compile one ``load_bearing_scaffold`` mapping into an oracle plan."""

    nodes = list(scaffold.get("nodes", []))
    if not nodes:
        raise ValueError("load-bearing scaffold has no nodes")
    _validate_nodes(list(scaffold.get("nodes", [])))
    ordered = _topological_order(nodes)

    shapes: list[dict[str, Any]] = []
    beats: list[dict[str, Any]] = []
    obligations: list[dict[str, Any]] = []
    author_assertions: list[dict[str, Any]] = []
    relation_definitions: list[dict[str, Any]] = []

    previous_shape_id = ""
    previous_concept = ""
    for index, node in enumerate(ordered, start=1):
        node_id = str(node["id"])
        role = str(node.get("role", ""))
        shape_id = f"S{index:02d}"
        beat_id = f"B{index:02d}-01"
        concept = _node_concept(node)
        obligation_id = f"OBL-{node_id}"

        shapes.append(
            {
                "id": shape_id,
                "order": index,
                "purpose": f"{role}: {node_id}",
                "entry": {"required": [previous_concept] if previous_concept else []},
                "exit": {"forbidden": [], "required": [concept]},
                "relations": [],
            }
        )
        beats.append(
            {
                "id": beat_id,
                "shape_id": shape_id,
                "order": 1,
                "purpose": f"{role}: {node_id}",
                "grounds": [concept],
                "requires": [],
                "depends_on": [],
                "establishes": [obligation_id],
            }
        )
        assertion_id = f"assert-{target_id}-{node_id}"
        author_assertions.append(
            {"author_assertion_id": assertion_id, "text": str(node.get("claim", ""))}
        )
        obligations.append(
            {
                "obligation_id": obligation_id,
                "kind": "establish",
                "authority": "author_asserted",
                "binds": {"beat_id": beat_id, "shape_id": shape_id},
                "purpose": f"Establish {node_id} ({role})",
                "witness": {"author_assertion_id": assertion_id},
            }
        )
        if previous_shape_id:
            relation = relation_for_role(role)
            shapes[index - 2]["relations"].append(
                {"from": previous_shape_id, "relation": relation, "to": shape_id}
            )
            relation_definitions.append(
                {
                    "schema": RELATION_DEFINITIONS_SCHEMA,
                    "from_unit": previous_shape_id,
                    "to_unit": shape_id,
                    "from_node": ordered[index - 2]["id"],
                    "to_node": node_id,
                    "role": role,
                    "relation": relation,
                    "predecessor_obligations": [f"OBL-{ordered[index - 2]['id']}"],
                    "definition": ESTABLISHMENT_RULE,
                }
            )
        previous_shape_id = shape_id
        previous_concept = concept

    central = scaffold.get("central_claim") or {}
    thesis = str(
        central.get("claim") or scaffold.get("entry", {}).get("claim") or target_id
    )

    plan = {
        "schema_version": "composition-plan-v0",
        "plan_id": plan_id or f"oracle-load-bearing-{target_id}",
        "thesis": thesis,
        "shapes": shapes,
        "beats": beats,
        "evidence_obligations": obligations,
        "author_assertions": author_assertions,
        "relation_definitions": relation_definitions,
    }
    plan = apply_beat_contract_defaults(plan)
    plan["plan_sha256"] = composition_plan_sha256(plan)
    errors = validate_composition_plan(plan)
    if errors:
        raise ValueError(f"compiled plan invalid: {errors}")
    return plan


def compile_all(coherence_results: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Compile every target that carries a ``load_bearing_scaffold``."""

    plans: dict[str, dict[str, Any]] = {}
    for target_id, payload in coherence_results.get("targets", {}).items():
        scaffold = payload.get("load_bearing_scaffold")
        if scaffold:
            plans[target_id] = compile_load_bearing_plan(target_id, scaffold)
    return plans
