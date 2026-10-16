"""Article plan, composition harness, source hydration, narrative, and draft packets."""

from __future__ import annotations

from .composition_plan import (
    apply_beat_contract_defaults,
    composition_plan_sha256,
    validate_composition_plan,
)
from .axis0_validation import run_negative_checks
from .composition_projection import project_for_generation, project_for_verification
from .composition_scheduler import reachable_units, schedule_units
from .composition_state import apply_verifier_result, empty_composition_state, empty_evidence_state
from .composition_unit_loop import run_unit_loop
from .g0_freeze import build_g0_artifact, query_sha256, validate_g0_identity
from .obligation_set import build_obligation_set, obligation_set_sha256, validate_obligation_parity
from .oflat_compiler import compile_oflat, validate_oflat_text
from .satisfiability import audit_plan_pool_satisfiability
from .shape_beat_compiler import compile_shape_beat_units
from .source_hydration import (
    document_from_result,
    document_id_from_result,
    hydrate_message_ids,
    hydrate_retrieval_documents,
    metadata_message_ids,
    sha256_text,
)
from .narrative_reconstruction import (
    build_narrative_map,
    build_slot_packets,
    format_narrative_packets,
    narrative_metrics,
    validate_narrative_map,
)
from .article_plan import (
    build_article_plan,
    format_article_plan_context,
    format_scoped_draft_packet,
    format_scoped_draft_packets,
    normalize_author_assertions,
    plan_sha256,
    scope_article_plan,
    validate_article_plan,
    validate_claim_trace,
    validate_scoped_draft_packet,
)

__all__ = [
    "apply_beat_contract_defaults",
    "run_negative_checks",
    "apply_verifier_result",
    "audit_plan_pool_satisfiability",
    "build_g0_artifact",
    "build_obligation_set",
    "build_article_plan",
    "build_narrative_map",
    "build_slot_packets",
    "compile_oflat",
    "compile_shape_beat_units",
    "composition_plan_sha256",
    "document_from_result",
    "document_id_from_result",
    "format_article_plan_context",
    "format_narrative_packets",
    "format_scoped_draft_packet",
    "format_scoped_draft_packets",
    "empty_composition_state",
    "empty_evidence_state",
    "hydrate_message_ids",
    "hydrate_retrieval_documents",
    "metadata_message_ids",
    "narrative_metrics",
    "normalize_author_assertions",
    "obligation_set_sha256",
    "plan_sha256",
    "project_for_generation",
    "project_for_verification",
    "query_sha256",
    "reachable_units",
    "run_unit_loop",
    "schedule_units",
    "scope_article_plan",
    "sha256_text",
    "validate_article_plan",
    "validate_claim_trace",
    "validate_composition_plan",
    "validate_g0_identity",
    "validate_narrative_map",
    "validate_obligation_parity",
    "validate_oflat_text",
    "validate_scoped_draft_packet",
]
