#!/usr/bin/env python3
"""Observation-only interpretation audit for paragraph evidence traces."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from generation_calls import run_generation_call
from generate import DEFAULT_GENERATOR_MODEL, GenerationRuntime


AUDIT_SCHEMA_VERSION = "1"
SUPPORT_LEVELS = {"direct", "adjacent", "contradictory", "irrelevant", "unknown"}
QUESTION_LEVELS = {"yes", "partial", "no", "unknown"}
AUDIT_FIELDS = (
    "supports_claim",
    "answers_question",
    "implied_question",
    "proposition",
    "role_candidates",
    "interpretation_confidence",
    "missing_nuance",
)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, float, int]:
    return (
        1 if candidate.get("retrieval_status") == "selected" else 0,
        _as_score(candidate.get("score")),
        -int(candidate.get("rank") or 0),
    )


def _intent_context(intent_trace: dict[str, Any]) -> dict[str, str]:
    """Recover target fields stored on retrieval events in the current trace."""

    events = intent_trace.get("retrieval_trace") or []
    first_event = events[0] if events else {}
    return {
        "intent_id": _as_text(intent_trace.get("intent_id")),
        "role": _as_text(intent_trace.get("role") or first_event.get("target_role")),
        "question": _as_text(
            intent_trace.get("question") or first_event.get("target_question")
        ),
        "intended_claim": _as_text(
            intent_trace.get("intended_claim") or first_event.get("target_claim")
        ),
    }


def candidate_union(
    intent_trace: dict[str, Any], max_candidates: int = 8
) -> list[dict[str, Any]]:
    """Return one inspectable candidate record per document across rounds."""

    observed: dict[str, dict[str, Any]] = {}
    for event in intent_trace.get("retrieval_trace") or []:
        round_number = int(event.get("round") or 0)
        for fallback_rank, source_candidate in enumerate(
            event.get("candidates") or [], start=1
        ):
            candidate_id = _as_text(source_candidate.get("document_id"))
            if not candidate_id:
                continue
            candidate = copy.deepcopy(source_candidate)
            candidate["retrieval_rounds"] = [round_number]
            existing = observed.get(candidate_id)
            if existing is not None:
                rounds = set(existing.get("retrieval_rounds") or [])
                rounds.add(round_number)
                if _candidate_priority(candidate) <= _candidate_priority(existing):
                    existing["retrieval_rounds"] = sorted(rounds)
                    continue
                candidate["retrieval_rounds"] = sorted(rounds)
            candidate.setdefault("rank", fallback_rank)
            observed[candidate_id] = candidate

    ranked = sorted(
        observed.values(),
        key=lambda candidate: (
            -_candidate_priority(candidate)[0],
            -_candidate_priority(candidate)[1],
            candidate.get("rank", 0),
            candidate.get("document_id", ""),
        ),
    )
    return ranked[: max(0, max_candidates)]


def build_interpretation_prompt(
    intent_trace: dict[str, Any], candidates: list[dict[str, Any]]
) -> str:
    """Build a candidate-level prompt; it does not ask for prose or repair."""

    target = _intent_context(intent_trace)
    candidate_blocks: list[str] = []
    for candidate in candidates:
        candidate_blocks.append(
            "\n".join(
                [
                    f"[CANDIDATE {candidate['document_id']}]",
                    f"retrieval_status: {candidate.get('retrieval_status', '')}",
                    f"rank: {candidate.get('rank', '')}",
                    f"score: {candidate.get('score', '')}",
                    "source_span:",
                    _as_text(candidate.get("source_span")) or "<empty>",
                    f"[/CANDIDATE {candidate['document_id']}]",
                ]
            )
        )
    return (
        "You are auditing retrieved evidence for a writing RAG system. This is "
        "an observation-only classification task. Do not write, repair, or reject "
        "the article. Do not use information outside the candidate source spans.\n\n"
        f"Intent: {target['intent_id']}\n"
        f"Role: {target['role']}\n"
        f"Reader question: {target['question']}\n"
        f"Intended claim: {target['intended_claim']}\n\n"
        "For each candidate, determine whether the source span directly supports "
        "the intended claim, is only related to it, contradicts it, is irrelevant, "
        "or cannot be determined. Direct support requires the source span to support "
        "the proposition itself, not merely share topic words. Use contradictory only "
        "when the source explicitly asserts an incompatible proposition under the same "
        "scope; a different question, an alternative framing, or missing support is "
        "not contradictory. Also determine whether the span answers the reader question. "
        "Do not promote an adjacent example into a general mechanism. Preserve missing "
        "conditions, uncertainty, and scope.\n\n"
        "Return exactly one JSON object per candidate in a JSON array. Use these fields:\n"
        "{\n"
        '  "document_id": "candidate id",\n'
        '  "supports_claim": "direct|adjacent|contradictory|irrelevant|unknown",\n'
        '  "answers_question": "yes|partial|no|unknown",\n'
        '  "implied_question": "the question this span appears to answer",\n'
        '  "proposition": "the proposition actually supported by the span",\n'
        '  "role_candidates": ["observation", "mechanism"],\n'
        '  "interpretation_confidence": 0.0,\n'
        '  "missing_nuance": "conditions or meaning absent from the target claim"\n'
        "}\n"
        "Keep implied_question, proposition, and missing_nuance short (12 words or "
        "fewer each). Use at most two role_candidates. Use an empty string or empty "
        "array when a field cannot be inferred. Return one compact JSON array only: "
        "no Markdown, explanation, or repeated source text.\n\n"
        + "\n\n".join(candidate_blocks)
    )


def _payload_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("interpretations", "candidates", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if payload.get("document_id"):
        return [payload]
    return []


def _decode_json_payload(raw_output: str) -> tuple[Any, str | None]:
    cleaned = raw_output.strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as first_error:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1]), None
            except json.JSONDecodeError:
                pass
        return None, f"invalid JSON: {first_error.msg}"


def _normalise_roles(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_as_text(item) for item in value if _as_text(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def parse_interpretation_output(
    raw_output: str, expected_ids: list[str]
) -> dict[str, Any]:
    """Parse audit output while preserving protocol problems as diagnostics."""

    payload, decode_error = _decode_json_payload(raw_output)
    records = _payload_records(payload)
    expected = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    duplicate_ids: list[str] = []

    if decode_error:
        errors.append(decode_error)
    for record in records:
        candidate_id = _as_text(record.get("document_id") or record.get("candidate_id"))
        if not candidate_id:
            errors.append("interpretation record omitted document_id")
            continue
        if candidate_id not in expected:
            errors.append(f"interpretation named unknown candidate: {candidate_id}")
            continue
        if candidate_id in by_id:
            duplicate_ids.append(candidate_id)
        by_id[candidate_id] = {
            "supports_claim": _as_text(
                record.get("supports_claim") or record.get("support_level")
            ).casefold()
            or None,
            "answers_question": _as_text(record.get("answers_question")).casefold()
            or None,
            "implied_question": _as_text(record.get("implied_question")) or None,
            "proposition": _as_text(record.get("proposition")) or None,
            "role_candidates": _normalise_roles(record.get("role_candidates")),
            "interpretation_confidence": record.get("interpretation_confidence"),
            "missing_nuance": _as_text(record.get("missing_nuance")) or None,
        }

    missing_ids = sorted(expected - set(by_id))
    if duplicate_ids:
        errors.append(
            "duplicate candidate records: " + ", ".join(sorted(set(duplicate_ids)))
        )
    if missing_ids:
        errors.append("missing candidate records: " + ", ".join(missing_ids))

    unknown_support = sorted(
        candidate_id
        for candidate_id, record in by_id.items()
        if record.get("supports_claim") not in SUPPORT_LEVELS
    )
    unknown_question = sorted(
        candidate_id
        for candidate_id, record in by_id.items()
        if record.get("answers_question") not in QUESTION_LEVELS
    )
    if unknown_support:
        errors.append(
            "unrecognized supports_claim values: " + ", ".join(unknown_support)
        )
    if unknown_question:
        errors.append(
            "unrecognized answers_question values: " + ", ".join(unknown_question)
        )

    if not records:
        status = "malformed"
    elif errors:
        status = "partial"
    else:
        status = "valid"
    return {
        "status": status,
        "records": by_id,
        "record_count": len(records),
        "expected_count": len(expected_ids),
        "missing_ids": missing_ids,
        "duplicate_ids": sorted(set(duplicate_ids)),
        "errors": list(dict.fromkeys(errors)),
    }


def _candidate_audit_fields(
    interpretation: dict[str, Any] | None,
    *,
    parse_status: str,
) -> dict[str, Any]:
    if interpretation is None:
        return {
            "interpretation_status": "unparsed",
            "supports_claim": None,
            "answers_question": None,
            "implied_question": None,
            "proposition": None,
            "role_candidates": [],
            "interpretation_confidence": None,
            "missing_nuance": None,
        }
    return {
        "interpretation_status": (
            "interpreted" if parse_status == "valid" else "interpreted_partial"
        ),
        **{field: interpretation.get(field) for field in AUDIT_FIELDS},
    }


def _merge_trace_interpretations(
    trace: list[dict[str, Any]], audits_by_intent: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = copy.deepcopy(trace)
    for intent_trace in merged:
        intent_id = _as_text(intent_trace.get("intent_id"))
        audit = audits_by_intent.get(intent_id)
        if audit is None:
            continue
        records = audit.get("parsed", {}).get("records", {})
        for event in intent_trace.get("retrieval_trace") or []:
            event["interpretation_status"] = audit["parse_status"]
            event["interpretation_note"] = (
                "Observation-only candidate interpretation; this does not change "
                "retrieval or acceptance."
            )
            for candidate in event.get("candidates") or []:
                candidate_id = _as_text(candidate.get("document_id"))
                candidate.update(
                    _candidate_audit_fields(
                        records.get(candidate_id), parse_status=audit["parse_status"]
                    )
                )
    return merged


def render_interpretation_debug(
    trace: list[dict[str, Any]], calls_by_intent: dict[str, dict[str, Any]]
) -> str:
    """Render interpreted candidates with explicit boundaries and raw output."""

    lines = [
        "[START: interpretation_audit]",
        f"audit_schema_version: {AUDIT_SCHEMA_VERSION}",
        "mode: observation_only",
    ]
    for intent in trace:
        context = _intent_context(intent)
        lines.extend(
            [
                "[START: intent]",
                f"intent_id: {context['intent_id']}",
                f"role: {context['role']}",
                f"question: {context['question']}",
                f"intended_claim: {context['intended_claim']}",
            ]
        )
        for event in intent.get("retrieval_trace") or []:
            lines.extend(
                [
                    "[START: retrieval]",
                    f"round: {event.get('round', '')}",
                    f"interpretation_status: {event.get('interpretation_status', 'pending')}",
                    "[END: retrieval]",
                ]
            )
            for candidate in event.get("candidates") or []:
                lines.extend(
                    [
                        "[START: candidate]",
                        f"rank: {candidate.get('rank', '')}",
                        f"document_id: {candidate.get('document_id', '')}",
                        f"retrieval_status: {candidate.get('retrieval_status', '')}",
                        f"score: {candidate.get('score', '')}",
                        "[START: source_span]",
                        _as_text(candidate.get("source_span")) or "<none>",
                        "[END: source_span]",
                        "[START: interpretation]",
                        f"status: {candidate.get('interpretation_status', 'unparsed')}",
                        f"supports_claim: {candidate.get('supports_claim') or '<not assessed>'}",
                        f"answers_question: {candidate.get('answers_question') or '<not assessed>'}",
                        f"implied_question: {candidate.get('implied_question') or '<not inferred>'}",
                        f"proposition: {candidate.get('proposition') or '<not inferred>'}",
                        "role_candidates: "
                        + (
                            ", ".join(candidate.get("role_candidates") or [])
                            or "<not inferred>"
                        ),
                        f"interpretation_confidence: {candidate.get('interpretation_confidence') if candidate.get('interpretation_confidence') is not None else '<not assessed>'}",
                        f"missing_nuance: {candidate.get('missing_nuance') or '<not inferred>'}",
                        "[END: interpretation]",
                        "[END: candidate]",
                    ]
                )
        call = calls_by_intent.get(context["intent_id"])
        if call:
            lines.extend(
                [
                    "[START: interpretation_call]",
                    f"parse_status: {call.get('parse_status', '')}",
                    f"record_count: {call.get('record_count', '')}",
                    f"errors: {json.dumps(call.get('parse_errors', []), ensure_ascii=False)}",
                    "raw_output:",
                    call.get("raw_output", "") or "<empty>",
                    "[END: interpretation_call]",
                ]
            )
        lines.append("[END: intent]")
    lines.append("[END: interpretation_audit]")
    return "\n".join(lines) + "\n"


def build_set_summary(audits: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Expose observed candidate sets without pretending they are gold labels."""

    sets: dict[str, set[str]] = {
        "selected": set(),
        "available_adjacent": set(),
        "direct_support": set(),
        "adjacent_support": set(),
        "contradictory": set(),
        "irrelevant": set(),
        "question_yes": set(),
        "question_partial": set(),
        "question_no": set(),
        "unknown_interpretation": set(),
    }
    for audit in audits:
        intent_id = audit["intent_id"]
        for candidate in audit.get("candidates", []):
            candidate_id = f"{intent_id}:{candidate['document_id']}"
            retrieval_status = candidate.get("retrieval_status")
            if retrieval_status in sets:
                sets[retrieval_status].add(candidate_id)
            interpretation = candidate.get("interpretation") or {}
            support = interpretation.get("supports_claim")
            question = interpretation.get("answers_question")
            support_key = {
                "direct": "direct_support",
                "adjacent": "adjacent_support",
                "contradictory": "contradictory",
                "irrelevant": "irrelevant",
            }.get(support)
            if support_key:
                sets[support_key].add(candidate_id)
            else:
                sets["unknown_interpretation"].add(candidate_id)
            question_key = {
                "yes": "question_yes",
                "partial": "question_partial",
                "no": "question_no",
            }.get(question)
            if question_key:
                sets[question_key].add(candidate_id)
    return {key: sorted(values) for key, values in sets.items()}


def audit_metadata(
    metadata_path: Path,
    *,
    runtime: GenerationRuntime | None,
    runtime_error: str | None,
    intent_ids: set[str] | None,
    max_candidates: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trace = metadata.get("paragraph_evidence_trace") or []
    selected_trace = [
        item
        for item in trace
        if not intent_ids or _as_text(item.get("intent_id")) in intent_ids
    ]
    skipped = [
        _as_text(item.get("intent_id"))
        for item in trace
        if intent_ids and _as_text(item.get("intent_id")) not in intent_ids
    ]
    audits: list[dict[str, Any]] = []
    calls_by_intent: dict[str, dict[str, Any]] = {}
    for intent_trace in selected_trace:
        context = _intent_context(intent_trace)
        candidates = candidate_union(intent_trace, max_candidates=max_candidates)
        candidate_ids = [
            _as_text(candidate.get("document_id")) for candidate in candidates
        ]
        prompt = build_interpretation_prompt(intent_trace, candidates)
        call_record: dict[str, Any] = {
            "stage": f"interpretation:{context['intent_id']}",
            "prompt": prompt,
            "prompt_tokens": None,
            "ok": False,
            "attempts": 0,
            "error": runtime_error,
            "raw_output": "",
            "parse_status": "not_run" if runtime is None else "not_parsed",
            "record_count": 0,
            "parse_errors": [],
        }
        if runtime is not None:
            try:
                call_record["prompt_tokens"] = runtime.count_tokens(prompt)
            except Exception as exc:
                call_record["error"] = (
                    f"token count failed: {type(exc).__name__}: {exc}"
                )
            call = run_generation_call(
                runtime.generate,
                prompt,
                max_new_tokens,
                temperature,
                stage=call_record["stage"],
                retries=0,
                fallback="",
            )
            call_record.update(call.as_dict())
            call_record["prompt"] = prompt
            parsed = parse_interpretation_output(call.text, candidate_ids)
            call_record.update(
                {
                    "raw_output": call.text,
                    "parse_status": parsed["status"],
                    "record_count": parsed["record_count"],
                    "parse_errors": parsed["errors"],
                }
            )
        else:
            parsed = {
                "status": "not_run",
                "records": {},
                "record_count": 0,
                "expected_count": len(candidate_ids),
                "missing_ids": candidate_ids,
                "duplicate_ids": [],
                "errors": ["model call not run"],
            }
        calls_by_intent[context["intent_id"]] = call_record
        audits.append(
            {
                "intent_id": context["intent_id"],
                "role": context["role"],
                "question": context["question"],
                "intended_claim": context["intended_claim"],
                "candidate_count": len(candidates),
                "candidates": [
                    {
                        "document_id": candidate["document_id"],
                        "retrieval_status": candidate.get("retrieval_status"),
                        "retrieval_rounds": candidate.get("retrieval_rounds", []),
                        "rank": candidate.get("rank"),
                        "score": candidate.get("score"),
                        "cosine_score": candidate.get("cosine_score"),
                        "lexical_score": candidate.get("lexical_score"),
                        "source_reference": candidate.get("source_reference", {}),
                        "source_span": candidate.get("source_span", ""),
                        "interpretation": _candidate_audit_fields(
                            parsed["records"].get(candidate["document_id"]),
                            parse_status=parsed["status"],
                        ),
                    }
                    for candidate in candidates
                ],
                "parse_status": parsed["status"],
                "parsed": parsed,
                "call": call_record,
            }
        )

    audits_by_intent = {audit["intent_id"]: audit for audit in audits}
    audited_trace = _merge_trace_interpretations(trace, audits_by_intent)
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_kind": "observation_only_candidate_interpretation",
        "reference_used_for_generation": False,
        "source_metadata": str(metadata_path),
        "source_reference": metadata.get("reference"),
        "generator_backend": runtime.backend if runtime is not None else None,
        "generator_model": runtime.model_name if runtime is not None else None,
        "runtime_used": runtime is not None,
        "runtime_error": runtime_error,
        "intent_ids_requested": sorted(intent_ids) if intent_ids else [],
        "intent_ids_audited": [audit["intent_id"] for audit in audits],
        "intent_ids_not_audited": skipped,
        "max_candidates": max_candidates,
        "intents": audits,
        "sets": build_set_summary(audits),
        "audited_evidence_trace": audited_trace,
        "interpretation_trace_debug": render_interpretation_debug(
            audited_trace, calls_by_intent
        ),
    }


def _output_name(metadata_path: Path) -> str:
    return metadata_path.parent.name or metadata_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an observation-only audit of retrieved evidence candidates."
    )
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Generation metadata JSON containing paragraph_evidence_trace; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for JSON and labelled trace outputs.",
    )
    parser.add_argument(
        "--intent-id",
        dest="intent_ids",
        action="append",
        help="Audit only this intent; repeat for multiple intents. Defaults to all.",
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--backend", choices=("mlx-lm", "mlx-vlm"), default="mlx-vlm"
    )
    parser.add_argument("--model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--model-cache", default="models/weights/huggingface")
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts and pending trace fields without loading MLX.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intent_ids = set(args.intent_ids or []) or None
    runtime: GenerationRuntime | None = None
    runtime_error: str | None = None
    if not args.dry_run:
        try:
            runtime = GenerationRuntime(args.backend, args.model, args.model_cache)
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"

    manifest: list[dict[str, Any]] = []
    for metadata_name in args.metadata:
        metadata_path = Path(metadata_name)
        result = audit_metadata(
            metadata_path,
            runtime=runtime,
            runtime_error=runtime_error,
            intent_ids=intent_ids,
            max_candidates=args.max_candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        name = _output_name(metadata_path)
        json_path = output_dir / f"{name}-interpretation.json"
        debug_path = output_dir / f"{name}-interpretation-trace.txt"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        debug_path.write_text(result["interpretation_trace_debug"], encoding="utf-8")
        manifest.append(
            {
                "source_metadata": str(metadata_path),
                "json_output": str(json_path),
                "debug_output": str(debug_path),
                "intent_ids_audited": result["intent_ids_audited"],
                "runtime_used": result["runtime_used"],
                "runtime_error": result["runtime_error"],
            }
        )
        print(f"Saved interpretation audit to {json_path}")
        print(f"Saved labelled interpretation trace to {debug_path}")

    manifest_path = output_dir / "interpretation-audit-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "audit_schema_version": AUDIT_SCHEMA_VERSION,
                "audit_kind": "observation_only_candidate_interpretation",
                "runtime_used": runtime is not None,
                "runtime_error": runtime_error,
                "runs": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved interpretation audit manifest to {manifest_path}")


if __name__ == "__main__":
    main()
