from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from article_plan import (  # noqa: E402
    build_article_plan,
    format_article_plan_context,
    plan_sha256,
    scope_article_plan,
    validate_article_plan,
    validate_claim_trace,
)
from paragraph_generation import (  # noqa: E402
    parse_claim_verification,
    verifier_protocol_diagnostics,
    verification_failures,
)
from test_source_hydration import _mdx_result, _message_result, _messages  # noqa: E402


def _plan() -> dict[str, object]:
    return build_article_plan(
        [_message_result(), _mdx_result()],
        _messages(),
        author_assertions=[
            {"author_assertion_id": "author-1", "text": "I chose this explanation."}
        ],
        plan_id="fixture-plan",
    )


def test_positive_mixed_plan_is_ready_and_validates() -> None:
    plan = _plan()
    assert plan["status"] == "ready"
    assert validate_article_plan(plan) == []


def test_positive_plan_exposes_separate_authority_allowlists() -> None:
    plan = _plan()
    allowlist = plan["allowlists"]["article"]
    assert "author-1" in allowlist["author_assertion_ids"]
    assert allowlist["span_ids"]
    assert allowlist["document_ids"] == ["window_001", "mdx_demo_001_content"]


def test_positive_plan_reports_complete_source_span_hash_coverage() -> None:
    plan = _plan()
    assert plan["coverage"]["source_span_coverage"] == 1.0
    assert plan["coverage"]["source_span_count"] == plan["coverage"]["hashed_source_span_count"]


def test_positive_source_backed_document_claim_passes() -> None:
    plan = _plan()
    assert validate_claim_trace(
        {"authority": "source_backed", "evidence_ids": ["mdx_demo_001_content"]}, plan
    ) == []


def test_positive_source_backed_span_claim_passes() -> None:
    plan = _plan()
    span_id = plan["allowlists"]["article"]["span_ids"][0]
    assert validate_claim_trace(
        {"authority": "source_backed", "evidence_span_ids": [span_id]}, plan
    ) == []


def test_positive_scoped_plan_accepts_only_the_slot_document() -> None:
    plan = _plan()
    scoped = scope_article_plan(plan, "intent-1", {"mdx_demo_001_content"})
    assert validate_claim_trace(
        {"authority": "source_backed", "evidence_ids": ["mdx_demo_001_content"]},
        scoped,
        slot_id="intent-1",
    ) == []


def test_negative_scoped_plan_rejects_cross_slot_document() -> None:
    plan = _plan()
    scoped = scope_article_plan(plan, "intent-1", {"mdx_demo_001_content"})
    errors = validate_claim_trace(
        {"authority": "source_backed", "evidence_ids": ["window_001"]},
        scoped,
        slot_id="intent-1",
    )
    assert "source document escaped allowlist" in errors


def test_positive_author_asserted_claim_passes_with_supplied_id() -> None:
    plan = _plan()
    assert validate_claim_trace(
        {"authority": "author_asserted", "author_assertion_ids": ["author-1"]}, plan
    ) == []


def test_positive_unclassified_interpretation_can_have_no_authority() -> None:
    assert validate_claim_trace(
        {"authority": "none", "relation": "interpretation"}, _plan()
    ) == []


def test_positive_none_authority_with_source_ids_uses_source_provenance() -> None:
    assert validate_claim_trace(
        {
            "authority": "none",
            "relation": "qualified_inference",
            "evidence_ids": ["window_001"],
        },
        _plan(),
    ) == []


def test_positive_explicit_causal_claim_passes_with_edge() -> None:
    plan = build_article_plan([_message_result()], _messages(), plan_id="causal-plan")
    edges = plan["allowlists"]["article"]["causal_edge_ids"]
    assert edges
    assert validate_claim_trace(
        {
            "authority": "source_backed",
            "claim_type": "technical_cause",
            "evidence_ids": ["window_001"],
            "narrative_edge_ids": [edges[0]],
        },
        plan,
    ) == []


def test_positive_context_format_contains_contract() -> None:
    formatted = format_article_plan_context(_plan())
    assert "source_backed" in formatted
    assert "author_asserted" in formatted
    assert "Causal edges:" in formatted


def test_positive_extended_verdict_carries_authority_and_assertion() -> None:
    plan = _plan()
    reports = parse_claim_verification(
        "VERDICT | intent-1 | S1 | interpretation | none | none | author_asserted | none | author-1",
        ["intent-1"],
        {"intent-1": "I chose this explanation."},
        article_plan=plan,
    )
    claim = reports[0]["claims"][0]
    assert claim["authority"] == "author_asserted"
    assert claim["author_assertion_ids"] == ["author-1"]


def test_positive_verifier_wiring_accepts_explicit_causal_edge() -> None:
    plan = build_article_plan([_message_result()], _messages(), plan_id="verifier-plan")
    edge_id = plan["allowlists"]["article"]["causal_edge_ids"][0]
    reports = parse_claim_verification(
        f"VERDICT | intent-1 | S1 | qualified_inference | window_001 | may | source_backed | {edge_id} | none",
        ["intent-1"],
        {"intent-1": "The control may explain the output."},
        article_plan=plan,
    )
    failures = verification_failures(
        reports,
        ["intent-1"],
        article_plan=plan,
        intent_claim_types={"intent-1": {"technical_cause"}},
    )
    assert failures == {}


def test_positive_article_plan_accepts_in_sentence_uncertainty_qualifier() -> None:
    plan = _plan()
    reports = parse_claim_verification(
        "VERDICT | intent-1 | S1 | qualified_inference | window_001 | none | none | none | none",
        ["intent-1"],
        {"intent-1": "The evidence does not converge to certainty."},
        article_plan=plan,
    )
    assert verification_failures(reports, ["intent-1"], article_plan=plan) == {}


def test_negative_relation_in_qualifier_field_is_protocol_error_not_hedge() -> None:
    plan = _plan()
    output = (
        "VERDICT | intent-1 | S1 | qualified_inference | window_001 | "
        "qualified_inference | none | none | none"
    )
    reports = parse_claim_verification(
        output,
        ["intent-1"],
        {"intent-1": "The evidence does not converge to certainty."},
        article_plan=plan,
    )
    claim = reports[0]["claims"][0]
    assert claim["required_qualifier"] is None
    assert claim["protocol_status"] == "malformed_qualifier_field"
    failures = verification_failures(
        reports,
        ["intent-1"],
        article_plan=plan,
    )
    assert any("verifier_protocol_error" in reason for reason in failures["intent-1"])
    assert not any("lacks a recognized hedge" in reason for reason in failures["intent-1"])
    diagnostics = verifier_protocol_diagnostics(
        output,
        ["intent-1"],
        {"intent-1": "The evidence does not converge to certainty."},
        reports=reports,
    )
    assert diagnostics["status"] == "protocol_malformed"
    assert diagnostics["malformed_qualifier_count"] == 1


def test_negative_qualifier_phrase_shift_is_not_parsed_as_evidence() -> None:
    output = (
        "VERDICT | intent-1 | S1 | interpretation | does not converge to certainty | "
        "none | none | none | none"
    )
    paragraphs = {"intent-1": "The evidence does not converge to certainty."}
    reports = parse_claim_verification(output, ["intent-1"], paragraphs)
    claim = reports[0]["claims"][0]
    assert claim["evidence_ids"] == []
    assert claim["required_qualifier"] == "does not converge to certainty"
    assert claim["protocol_status"] == "malformed_extended_field_order"
    failures = verification_failures(reports, ["intent-1"])
    assert failures["intent-1"] == [
        "verifier_protocol_error: qualifier phrase was shifted into the evidence field"
    ]


def test_negative_qualifier_before_evidence_is_normalized_without_scope_escape() -> None:
    output = (
        "VERDICT | intent-1 | S1 | interpretation | important | window_001 | "
        "none | source_backed | none | none"
    )
    paragraphs = {"intent-1": "This is an important interpretation."}
    reports = parse_claim_verification(output, ["intent-1"], paragraphs)
    claim = reports[0]["claims"][0]
    assert claim["evidence_ids"] == ["window_001"]
    assert claim["required_qualifier"] == "important"
    assert claim["scope_violation_ids"] == []
    assert claim["protocol_status"] == "malformed_extended_field_order"
    failures = verification_failures(reports, ["intent-1"])
    assert failures["intent-1"] == [
        "verifier_protocol_error: qualifier phrase was shifted into the evidence field"
    ]


def test_positive_article_plan_accepts_literal_in_sentence_qualifier() -> None:
    plan = _plan()
    reports = parse_claim_verification(
        "VERDICT | intent-1 | S1 | qualified_inference | window_001 | retroactive fitting | none | none | none",
        ["intent-1"],
        {"intent-1": "This is retroactive fitting, not confirmation."},
        article_plan=plan,
    )
    assert verification_failures(reports, ["intent-1"], article_plan=plan) == {}


def test_negative_plan_hash_mutation_fails() -> None:
    plan = copy.deepcopy(_plan())
    plan["source_packets"][0]["text"] = "tampered"
    assert "article plan hash mismatch" in validate_article_plan(plan)


def test_negative_packet_text_hash_mutation_fails() -> None:
    plan = copy.deepcopy(_plan())
    plan["source_packets"][0]["text"] = "tampered"
    plan["plan_sha256"] = plan_sha256(plan)
    assert any("packet text hash mismatch" in error for error in validate_article_plan(plan))


def test_negative_source_span_coverage_mutation_fails() -> None:
    plan = copy.deepcopy(_plan())
    plan["coverage"]["source_span_coverage"] = 0.5
    plan["plan_sha256"] = plan_sha256(plan)
    assert "article plan source span coverage is not 100%" in validate_article_plan(plan)


def test_negative_author_asserted_without_id_fails() -> None:
    errors = validate_claim_trace({"authority": "author_asserted"}, _plan())
    assert "author_asserted claim has no author assertion ID" in errors


def test_negative_unknown_author_assertion_fails() -> None:
    errors = validate_claim_trace(
        {"authority": "author_asserted", "author_assertion_ids": ["invented"]}, _plan()
    )
    assert "author assertion escaped allowlist" in errors


def test_negative_causal_claim_without_edge_fails() -> None:
    errors = validate_claim_trace(
        {
            "authority": "source_backed",
            "claim_type": "technical_cause",
            "evidence_ids": ["window_001"],
        },
        _plan(),
    )
    assert "causal claim has no narrative edge ID" in errors


def test_negative_edge_outside_causal_allowlist_fails() -> None:
    errors = validate_claim_trace(
        {
            "authority": "source_backed",
            "claim_type": "technical_cause",
            "evidence_ids": ["window_001"],
            "narrative_edge_ids": ["context:0001"],
        },
        _plan(),
    )
    assert "causal edge escaped allowlist" in errors


def test_negative_unknown_authority_fails() -> None:
    errors = validate_claim_trace({"authority": "model_inferred"}, _plan())
    assert "unknown authority: model_inferred" in errors


def test_negative_no_authority_on_supported_claim_fails() -> None:
    errors = validate_claim_trace(
        {"authority": "none", "relation": "supported"}, _plan()
    )
    assert "authority=none is invalid for relation: supported" in errors


def test_negative_none_authority_cannot_claim_author_assertion() -> None:
    errors = validate_claim_trace(
        {
            "authority": "none",
            "relation": "interpretation",
            "author_assertion_ids": ["author-1"],
        },
        _plan(),
    )
    assert "authority=none cannot carry author assertion IDs" in errors
