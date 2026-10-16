"""Conceptual stages for evidence-scoped paragraph generation.

This module owns the boundaries that talk to the generator. Planning and the
article coordinator can therefore treat drafting, verification, and repair as
independent stage results instead of catching model errors themselves.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .generation_calls import GenerationCallResult, run_generation_call
from writing_rag.plan.article_plan import scope_article_plan
from .paragraph_generation import (
    article_state,
    build_claim_verification_prompt,
    build_local_repair_prompt,
    build_paragraph_group_prompt,
    evidence_fallback_paragraphs,
    merge_paragraph_repairs,
    parse_claim_verification,
    parse_paragraph_group_output,
    verifier_protocol_diagnostics,
    verification_failures,
)


@dataclass
class ParagraphStageResult:
    """Output contract shared by draft, verification, and repair stages."""

    stage: str
    ok: bool
    paragraphs: dict[str, str] = field(default_factory=dict)
    reports: list[dict[str, Any]] = field(default_factory=list)
    failures: dict[str, list[str]] = field(default_factory=dict)
    verifier_outputs: list[dict[str, Any]] = field(default_factory=list)
    verifier_protocol_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    generation_calls: list[dict[str, object]] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    draft_plan_context_tokens: list[int] = field(default_factory=list)


def _overflow_call(stage: str, measured: int, limit: int) -> GenerationCallResult:
    return GenerationCallResult(
        stage=stage,
        ok=False,
        text="",
        attempts=0,
        error=f"context has {measured} tokens; configured limit is {limit}",
    )


def _failure_for_ids(intent_ids: list[str], message: str) -> dict[str, list[str]]:
    return {intent_id: [message] for intent_id in intent_ids}


def draft_group_stage(
    group: list[dict[str, Any]],
    *,
    query: str,
    style_profile: str,
    narrative_template: str,
    previous_tail: str,
    completed_intents: list[dict[str, Any]],
    target_words_per_intent: int,
    document_type: str = "investigative",
    draft_plan_context: str = "",
    draft_context_policy: str = "legacy",
    draft_plan_context_token_cap: int | None = None,
    generate: Callable[[str, int, float], str],
    count_prompt_tokens: Callable[[str], int],
    context_token_limit: int,
    max_tokens: int,
    temperature: float,
) -> ParagraphStageResult:
    """Draft one independent paragraph group and parse its markers."""

    expected_ids = [intent["intent_id"] for intent in group]
    stage = f"paragraph:{','.join(expected_ids)}"
    prompt = build_paragraph_group_prompt(
        group,
        query=query,
        style_profile=style_profile,
        narrative_template=narrative_template,
        previous_tail=previous_tail,
        article_state=article_state(completed_intents),
        target_words_per_intent=target_words_per_intent,
        document_type=document_type,
        draft_plan_context=draft_plan_context,
        draft_context_policy=draft_context_policy,
    )
    measured = count_prompt_tokens(prompt)
    plan_context_tokens = (
        count_prompt_tokens(draft_plan_context)
        if draft_plan_context.strip()
        else 0
    )
    if (
        draft_plan_context_token_cap is not None
        and plan_context_tokens > draft_plan_context_token_cap
    ):
        plan_context_overflow = True
        call = _overflow_call(
            stage,
            plan_context_tokens,
            draft_plan_context_token_cap,
        )
    else:
        plan_context_overflow = False
        call = (
            _overflow_call(stage, measured, context_token_limit)
            if measured > context_token_limit
            else run_generation_call(
                generate,
                prompt,
                max_tokens,
                temperature,
                stage=stage,
                retries=1,
                retry_temperature=min(temperature, 0.2),
            )
        )
    paragraphs = parse_paragraph_group_output(call.text, expected_ids)
    call_record = call.as_dict()
    call_record["prompt_sha256"] = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    call_record["draft_plan_context_present"] = bool(draft_plan_context.strip())
    call_record["draft_plan_context_token_count"] = plan_context_tokens
    call_record["plan_context_overflow"] = plan_context_overflow
    if not paragraphs and not call.ok and not plan_context_overflow:
        paragraphs = evidence_fallback_paragraphs(group)
        if paragraphs:
            call_record["fallback_used"] = True

    missing = [intent_id for intent_id in expected_ids if intent_id not in paragraphs]
    failures = _failure_for_ids(missing, "generator omitted this paragraph intent")
    if not call.ok:
        message = f"{stage} failed: {call.error}"
        for intent_id in expected_ids:
            failures.setdefault(intent_id, []).append(message)

    return ParagraphStageResult(
        stage=stage,
        ok=call.ok and not missing,
        paragraphs=paragraphs,
        failures=failures,
        generation_calls=[call_record],
        prompt_tokens=[measured],
        draft_plan_context_tokens=[plan_context_tokens],
    )


def verify_group_stage(
    paragraphs: dict[str, str],
    group: list[dict[str, Any]],
    *,
    generate: Callable[[str, int, float], str],
    count_prompt_tokens: Callable[[str], int],
    context_token_limit: int,
    max_tokens: int,
    stage_prefix: str = "verify",
    article_plan: dict[str, Any] | None = None,
) -> ParagraphStageResult:
    """Verify claims for one paragraph group, failing closed on call errors."""

    intent_ids = list(paragraphs)
    stage = f"{stage_prefix}:{','.join(intent_ids)}"
    allowed_evidence_ids = {
        intent["intent_id"]: set(intent.get("evidence_ids", []))
        for intent in group
    }
    intent_article_plans = (
        {
            intent["intent_id"]: scope_article_plan(
                article_plan,
                intent["intent_id"],
                set(intent.get("evidence_ids", [])),
            )
            for intent in group
        }
        if article_plan
        else None
    )
    prompt = build_claim_verification_prompt(
        paragraphs,
        group,
        article_plan=article_plan,
    )
    measured = count_prompt_tokens(prompt)
    call = (
        _overflow_call(stage, measured, context_token_limit)
        if measured > context_token_limit
        else run_generation_call(
            generate,
            prompt,
            min(max_tokens, 1400),
            0.0,
            stage=stage,
            retries=1,
            retry_temperature=0.0,
        )
    )
    call_record = call.as_dict()
    call_record["prompt_sha256"] = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    if not call.ok:
        return ParagraphStageResult(
            stage=stage,
            ok=False,
            failures=_failure_for_ids(intent_ids, f"{stage} failed: {call.error}"),
            generation_calls=[call_record],
            prompt_tokens=[measured],
        )

    reports = parse_claim_verification(
        call.text,
        intent_ids,
        paragraphs,
        allowed_evidence_ids=allowed_evidence_ids,
        article_plan=article_plan,
    )
    protocol = verifier_protocol_diagnostics(
        call.text,
        intent_ids,
        paragraphs,
        reports,
    )
    failures = verification_failures(
        reports,
        intent_ids,
        protocol_diagnostics=protocol,
        article_plan=article_plan,
        intent_claim_types={
            intent["intent_id"]: set(intent.get("claim_types", []))
            for intent in group
        },
        intent_article_plans=intent_article_plans,
    )
    return ParagraphStageResult(
        stage=stage,
        ok=not failures,
        reports=reports,
        failures=failures,
        verifier_outputs=[
            {"stage": stage_prefix, "intent_ids": intent_ids, "output": call.text}
        ],
        verifier_protocol_diagnostics=[
            {
                "stage": stage,
                "intent_ids": intent_ids,
                "raw_output": call.text,
                **protocol,
            }
        ],
        generation_calls=[call_record],
        prompt_tokens=[measured],
    )


def repair_group_stage(
    paragraphs: dict[str, str],
    failures: dict[str, list[str]],
    group: list[dict[str, Any]],
    *,
    generate: Callable[[str, int, float], str],
    count_prompt_tokens: Callable[[str], int],
    context_token_limit: int,
    max_tokens: int,
    temperature: float,
    document_type: str = "investigative",
    article_plan: dict[str, Any] | None = None,
) -> ParagraphStageResult:
    """Repair only failed paragraphs and verify the replacement before commit."""

    repairable = {
        intent_id: reasons
        for intent_id, reasons in failures.items()
        if intent_id in paragraphs
    }
    if not repairable:
        return ParagraphStageResult(stage="repair", ok=True)

    intent_ids = list(repairable)
    stage = f"repair:{','.join(intent_ids)}"
    prompt = build_local_repair_prompt(
        paragraphs,
        repairable,
        group,
        document_type=document_type,
    )
    measured = count_prompt_tokens(prompt)
    call = (
        _overflow_call(stage, measured, context_token_limit)
        if measured > context_token_limit
        else run_generation_call(
            generate,
            prompt,
            min(max_tokens, 1400),
            min(temperature, 0.2),
            stage=stage,
            retries=1,
            retry_temperature=0.1,
        )
    )
    calls = [call.as_dict()]
    if not call.ok:
        return ParagraphStageResult(
            stage=stage,
            ok=False,
            failures=_failure_for_ids(intent_ids, f"{stage} failed: {call.error}"),
            generation_calls=calls,
            prompt_tokens=[measured],
        )

    repairs = parse_paragraph_group_output(call.text, intent_ids)
    candidate = merge_paragraph_repairs(paragraphs, repairs)
    repaired_subset = {
        intent_id: candidate[intent_id]
        for intent_id in intent_ids
        if intent_id in candidate
    }
    if not repaired_subset:
        return ParagraphStageResult(
            stage=stage,
            ok=False,
            failures=_failure_for_ids(
                intent_ids, "repair omitted this paragraph intent"
            ),
            generation_calls=calls,
            prompt_tokens=[measured],
        )

    verification = verify_group_stage(
        repaired_subset,
        group,
        generate=generate,
        count_prompt_tokens=count_prompt_tokens,
        context_token_limit=context_token_limit,
        max_tokens=max_tokens,
        stage_prefix="verify_repair",
        article_plan=article_plan,
    )
    calls.extend(verification.generation_calls)
    prompt_tokens = [measured, *verification.prompt_tokens]
    if not verification.ok:
        return ParagraphStageResult(
            stage=stage,
            ok=False,
            failures=verification.failures,
            generation_calls=calls,
            prompt_tokens=prompt_tokens,
            verifier_outputs=verification.verifier_outputs,
            verifier_protocol_diagnostics=verification.verifier_protocol_diagnostics,
            reports=verification.reports,
        )

    return ParagraphStageResult(
        stage=stage,
        ok=True,
        paragraphs=candidate,
        reports=verification.reports,
        verifier_outputs=verification.verifier_outputs,
        verifier_protocol_diagnostics=verification.verifier_protocol_diagnostics,
        generation_calls=calls,
        prompt_tokens=prompt_tokens,
    )
