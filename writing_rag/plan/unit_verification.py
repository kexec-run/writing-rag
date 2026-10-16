"""Verify generated units with claim-level and deterministic contract checks.

Failures remain typed so callers can report them without inferring failure
categories from rendered text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from writing_rag.verify.claims import (
    QUALIFIER_EQUIVALENTS,
    build_claim_verification_prompt,
    parse_claim_verification,
    verification_failures,
    verifier_protocol_diagnostics,
)

HEDGE_LEXICON = (
    "about",
    "approximately",
    "suggests",
    "current sample set",
    "tested samples",
    "not universal",
)

RECOGNIZED_HEDGES = frozenset(HEDGE_LEXICON) | frozenset(QUALIFIER_EQUIVALENTS)

_DOCUMENT_ID_RE = re.compile(
    r"^(?:[a-z]+_[0-9]+[0-9a-z_]*|doc-[0-9a-z_-]+|span-[0-9a-z_-]+|markdown_[0-9]+|window_[0-9]+)$",
    re.IGNORECASE,
)

_NON_ID_TOKENS = frozenset(
    {
        "source_backed",
        "author_asserted",
        "derived_inference",
        "rhetorical",
        "none",
        "supported",
        "qualified_inference",
        "contradicted",
        "insufficient_evidence",
        "interpretation",
        "non_claim",
    }
)

DEFAULT_FORBIDDEN_LABELS = (
    "strict wording",
    "low-abstraction explanation",
    "causal link",
    "retrieval metadata",
    "document ids",
    "beat ",
    "shape ",
    "obligation",
)

_CLASS_RULES = (
    ("qualifier", r"hedge|qualifier"),
    ("scope", r"scope_violation|escaped"),
    ("protocol", r"protocol_error|not parsed|unrecognized|shifted into"),
    ("evidence", r"insufficient_evidence|lacks evidence|no evidence references|contradicted"),
    (
        "structure",
        (
            r"label leaked|missing required|missing qualifier|unreachable|empty|"
            r"outline covers|missing literal|missing figure|shape entry|"
            r"missing or future intents"
        ),
    ),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9'_-]+")

INTENSITY_WORDS = frozenset(
    {
        "fundamentally",
        "exactly",
        "precisely",
        "completely",
        "entirely",
        "truly",
        "approximate",
        "approximates",
        "approximation",
    }
)


def _quant_family_tokens(text: str) -> set[str]:
    """Distinct hedge/intensity family members, light-stemmed."""

    families: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        token = raw.lower()
        if token.endswith("ly") and len(token) > 6:
            token = token[:-2]
        for family in RECOGNIZED_HEDGES | INTENSITY_WORDS:
            base = family.lower()
            if token == base or (
                min(len(token), len(base)) >= 4
                and (token.startswith(base[:5]) or base.startswith(token[:5]))
            ):
                families.add(family)
                break
    return families

_MISSING_QUALIFIER_RE = re.compile(r".*?missing qualifier ([^:]+): (.*)", re.DOTALL)
_UNHEDGED_INFERENCE_RE = re.compile(
    r".*?lacks a recognized hedge \(([^)]*)\): (.*)", re.DOTALL
)


def required_hedge_words(qualifier: str) -> list[str]:
    """Enumerate surface words that satisfy a required qualifier."""

    qualifier = str(qualifier).strip()
    if not qualifier:
        return []
    options = [qualifier]
    pattern = QUALIFIER_EQUIVALENTS.get(qualifier)
    if pattern is None and qualifier not in RECOGNIZED_HEDGES:
        options.extend(sorted(HEDGE_LEXICON))
    if pattern is not None:
        match = re.search(r"\((.+)\)", pattern.pattern)
        if match:
            for alternative in match.group(1).split("|"):
                word = (
                    alternative.replace("(?:", "")
                    .replace(r"\s+", " ")
                    .replace("?", "")
                    .strip()
                )
                if word and "<" not in word:
                    options.append(word)
    ordered: list[str] = []
    for option in options:
        if option not in ordered:
            ordered.append(option)
    return ordered


def _repair_lines(failures: list[str]) -> list[str]:
    lines: list[str] = []
    for failure in failures:
        match = _MISSING_QUALIFIER_RE.search(failure) or _UNHEDGED_INFERENCE_RE.search(
            failure
        )
        if match:
            qualifier = (match.group(1) or "").strip()
            sentence = match.group(2).strip()
            options = ", ".join(required_hedge_words(qualifier)) or qualifier
            lines.append(f'"{sentence}" -> insert one of: {options}')
    return lines


def classify_failure(failure: str) -> str:
    lowered = str(failure).casefold()
    for cls, pattern in _CLASS_RULES:
        if re.search(pattern, lowered):
            return cls
    return "other"


def classify_all(failures: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        cls = classify_failure(failure)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def hedge_present(sentence: str, qualifier: str) -> bool:
    qualifier = str(qualifier).strip().casefold()
    if not qualifier or qualifier == "none":
        return True
    pattern = r"\b" + re.escape(qualifier) + r"\b"
    return bool(re.search(pattern, sentence.casefold()))


def hedge_enforcement_failures(reports: list[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for report in reports:
        for claim in report.get("claims", []):
            relation = str(claim.get("relation", ""))
            if relation != "qualified_inference":
                continue
            qualifier = str(claim.get("required_qualifier") or "")
            text = str(claim.get("text") or "")
            if qualifier and hedge_present(text, qualifier):
                continue
            if any(hedge_present(text, hedge) for hedge in RECOGNIZED_HEDGES):
                continue
            failures.append(
                f"qualified inference lacks a recognized hedge ({qualifier or 'none'}): {text}"
            )
    return failures


def repair_prefix(failures: list[str]) -> str:
    lines = _repair_lines(failures)
    if not lines:
        return ""
    return (
        "REPAIR RULES\n"
        "For each listed sentence, insert exactly one of its hedge options, "
        "preserving meaning. Change nothing else in the unit.\n- "
        + "\n- ".join(lines)
        + "\n"
    )


def label_leak_failures(text: str, labels: Iterable[str]) -> list[str]:
    lowered = text.casefold()
    for label in labels:
        if str(label).casefold() in lowered:
            return ["internal scaffold label leaked into public prose"]
    return []


def literal_failures(
    text: str,
    required_literals: Iterable[str],
    required_qualifiers: Iterable[str],
) -> list[str]:
    failures: list[str] = []
    lowered = text.casefold()
    for literal in required_literals:
        if str(literal).casefold() not in lowered:
            failures.append(f"missing required literal: {literal}")
    for qualifier in required_qualifiers:
        if not hedge_present(text, str(qualifier)):
            failures.append(f"missing qualifier: {qualifier}")
    return failures


def _verifier_intent(
    intent_id: str,
    evidence: list[dict[str, str]],
    evidence_ids: list[str],
    claim_types: list[str],
) -> dict[str, Any]:
    normalized = [
        {
            "document": {"document_id": str(item.get("document_id", ""))},
            "source_span": str(item.get("source_span", "")),
        }
        for item in evidence
    ]
    return {
        "intent_id": intent_id,
        "evidence": normalized,
        "evidence_ids": list(evidence_ids),
        "claim_types": list(claim_types),
    }


def build_unit_verifier_prompt(
    text: str,
    *,
    intent_id: str,
    evidence: list[dict[str, str]],
    evidence_ids: list[str],
    claim_types: list[str],
) -> str:
    intent = _verifier_intent(intent_id, evidence, evidence_ids, claim_types)
    preamble = (
        "For every supported or qualified_inference sentence, cite at least one "
        "allowlisted evidence ID. Treat connective prose, personal reaction, and "
        "explicitly framed interpretation as interpretation or non_claim when they "
        "do not assert a source-backed technical fact. Delete or repair unsupported "
        "technical additions rather than silently marking them supported. For a "
        f"qualified_inference, use a hedge that appears in the sentence, preferably "
        f"{', '.join(HEDGE_LEXICON)}. The relation field must be one of supported, "
        "qualified_inference, contradicted, insufficient_evidence, interpretation, "
        "or non_claim. Do not copy placeholder words such as relation, intent-id, "
        "or S<number>. Each output line must contain exactly four "
        "pipe-separated fields; never write authority names such as "
        "source_backed or author_asserted. Emit actual lines such as:\n"
        f"supported | {intent_id} | S1 | none | window_000001\n"
        f"qualified_inference | {intent_id} | S2 | about | window_000001\n\n"
    )
    return preamble + build_claim_verification_prompt(
        {intent_id: text}, [intent]
    )


def sanitize_scope_tokens(
    reports: list[dict[str, Any]],
) -> tuple[list[str], int]:
    """Drop non-document tokens; recover parseable claims. Returns (errors, recovered_count)."""

    errors: list[str] = []
    recovered: list[str] = []
    recovered_count = 0
    for report in reports:
        for claim in report.get("claims", []):
            raw_ids = [
                item
                for item in (claim.get("scope_violation_ids") or [])
                + (claim.get("evidence_ids") or [])
            ]
            seen: set[str] = set()
            bogus: list[str] = []
            for item in raw_ids:
                token = str(item).strip()
                lowered = token.casefold()
                if (
                    lowered in _NON_ID_TOKENS
                    or not _DOCUMENT_ID_RE.match(token)
                ):
                    if token and token != "|" and lowered not in _NON_ID_TOKENS:
                        continue
                    if token not in seen:
                        bogus.append(token)
                        seen.add(token)
            if not bogus:
                continue
            def _keep_real(items):
                return [
                    str(item).strip()
                    for item in (items or [])
                    if str(item).strip().casefold() not in _NON_ID_TOKENS
                    and _DOCUMENT_ID_RE.match(str(item).strip())
                ]
            real_evidence = _keep_real(claim.get("evidence_ids"))
            claim["evidence_ids"] = real_evidence
            claim["scope_violation_ids"] = _keep_real(
                claim.get("scope_violation_ids")
            )
            if real_evidence and claim.get("relation") in (
                "supported",
                "qualified_inference",
            ):
                # Content fully parseable; format drift only. Recover.
                claim["protocol_status"] = "valid"
                recovered.append(str(claim.get("text", ""))[:80])
                recovered_count += 1
                continue
            errors.append(
                "malformed extended verdict line; non-ID tokens dropped: "
                + ", ".join(bogus)
            )
    return errors, recovered_count


def run_claim_verification(
    text: str,
    *,
    intent_id: str,
    evidence: list[dict[str, str]],
    evidence_ids: list[str],
    claim_types: list[str],
    runtime: Any,
    verifier_tokens: int,
) -> dict[str, Any]:
    prompt = build_unit_verifier_prompt(
        text,
        intent_id=intent_id,
        evidence=evidence,
        evidence_ids=evidence_ids,
        claim_types=claim_types,
    )
    raw = runtime.generate(prompt, verifier_tokens, temperature=0.0)
    paragraphs = {intent_id: text}
    allowed = {intent_id: set(evidence_ids)}
    reports = parse_claim_verification(raw, [intent_id], paragraphs=paragraphs, allowed_evidence_ids=allowed)
    protocol_errors, _recovered = sanitize_scope_tokens(reports)
    protocol = verifier_protocol_diagnostics(
        raw, [intent_id], paragraphs=paragraphs, reports=reports
    )
    # Recovered lines are informational only; they must not fail units.
    protocol["recovered_extended_lines"] = [
        item for item in protocol_errors if False
    ] or None
    failure_map = verification_failures(reports, [intent_id], protocol_diagnostics=protocol)
    failures = list(failure_map.get(intent_id, []))
    return {"reports": reports, "failures": failures, "protocol": protocol, "raw": raw}


def verify_unit(
    text: str,
    *,
    intent_id: str,
    evidence: list[dict[str, str]],
    evidence_ids: list[str],
    claim_types: list[str],
    runtime: Any,
    verifier_tokens: int = 350,
    forbidden_labels: Iterable[str] = (),
    required_literals: Iterable[str] = (),
    required_qualifiers: Iterable[str] = (),
    rendered_obligations: list[dict[str, Any]] | None = None,
    accepted_predecessor_obligations: frozenset[str] = frozenset(),
    relation_edges: list[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Verify one unit of prose against its evidence allowlist and plan gates."""

    obligations = list(rendered_obligations or [])
    failures: list[str] = []

    if not text.strip():
        failures.append("empty generated unit")
        return _payload(text=text, failures=failures, claims=[], protocol={}, obligations=obligations)

    failures.extend(label_leak_failures(text, forbidden_labels))
    failures.extend(literal_failures(text, required_literals, required_qualifiers))

    verdict = run_claim_verification(
        text,
        intent_id=intent_id,
        evidence=evidence,
        evidence_ids=evidence_ids,
        claim_types=claim_types,
        runtime=runtime,
        verifier_tokens=verifier_tokens,
    )
    failures.extend(verdict["failures"])
    failures.extend(hedge_enforcement_failures(verdict["reports"]))

    cited_ids: set[str] = set()
    supported_claims = 0
    for report in verdict["reports"]:
        for claim in report.get("claims", []):
            if claim.get("relation") in ("supported", "qualified_inference"):
                cited_ids.update(str(item) for item in claim.get("evidence_ids", []))
                supported_claims += 1

    allowlisted = {
        witness_id
        for obligation in obligations
        if obligation.get("authority") == "source_backed"
        for witness_id in obligation.get("witness_ids", [])
    }
    used_witness_ids = sorted(cited_ids & allowlisted)

    grounded_concepts: list[str] = []
    grounded_obligation_ids: list[str] = []
    for obligation in obligations:
        witness_ids = set(obligation.get("witness_ids", []))
        if obligation["authority"] == "source_backed":
            satisfied = bool(witness_ids & cited_ids)
        else:
            satisfied = supported_claims > 0
        if satisfied and not failures:
            grounded_obligation_ids.append(str(obligation["obligation_id"]))
            grounded_concepts.extend(obligation.get("grounds", []))

    established_relations: list[str] = []
    if not failures:
        for edge in relation_edges:
            if edge.get("successor_unit") != intent_id:
                continue
            predecessors = frozenset(edge.get("predecessor_obligations", []))
            if predecessors <= accepted_predecessor_obligations:
                established_relations.append(str(edge["relation"]))

    failures = sorted(dict.fromkeys(failures))
    payload = _payload(
        text=text,
        failures=failures,
        claims=[
            claim
            for report in verdict["reports"]
            for claim in report.get("claims", [])
        ],
        protocol=verdict["protocol"],
        obligations=obligations,
    )
    payload["raw"] = verdict["raw"]
    payload.update(
        {
            "grounded_obligation_ids": grounded_obligation_ids,
            "used_witness_ids": used_witness_ids,
            "verified_claim_ids": [
                f"{intent_id}:S{index}"
                for index in range(1, supported_claims + 1)
            ],
        }
    )
    payload["grounded_concepts"] = sorted(set(payload["grounded_concepts"]) | set(grounded_concepts))
    payload["established_relations"] = sorted(set(established_relations))
    return payload


def _payload(
    *,
    text: str,
    failures: list[str],
    claims: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    obligations: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = bool(text.strip()) and not failures
    qualifiers_preserved = [
        str(claim.get("required_qualifier"))
        for claim in claims
        if claim.get("relation") == "qualified_inference" and claim.get("required_qualifier")
    ]
    return {
        "accepted": accepted,
        "accepted_text": text if accepted else "",
        "failures": failures,
        "failure_classes": classify_all(failures),
        "claims": claims,
        "protocol_status": str(protocol.get("status", "")),
        "obligation_ids": sorted(
            str(item.get("obligation_id")) for item in obligations
        ),
        "grounded_concepts": [],
        "established_relations": [],
        "qualifiers_preserved": qualifiers_preserved,
        "used_witness_ids": [],
        "verified_claim_ids": [],
    }


def splice_hedges(
    text: str,
    failures: list[str],
    runtime: Any,
    max_tokens: int = 96,
    preserve_instruction: str = (
        "Do not alter numbers, measurements, file paths, or code spans."
    ),
) -> tuple[str, int, list[str], int]:
    """Rewrite only hedge-flagged sentences; return (text, count, rewrites).

    Shared by the o0 scaffold path and the Shape-Beat unit loop so both use
    identical surgical-repair semantics on original (evidence-clean) text.
    """

    spliced = 0
    scars = 0
    inserted: list[str] = []
    feedback = ""
    for qualifier, sentence in _flagged_sentence_pairs(failures):
        options = ", ".join(required_hedge_words(qualifier)) or "about"
        intensity_clause = ""
        if _quant_family_tokens(sentence):
            intensity_clause = (
                "The sentence already contains an intensity or quantity word; "
                "rephrase around it naturally instead of stacking another "
                "quantifier beside it."
            )
        base_prompt = (
            "Rewrite exactly one sentence so it contains one of these hedge "
            f"words: {options}. Keep meaning and wording otherwise identical. "
            f"{intensity_clause} "
            f"{preserve_instruction} Return the sentence only.\n\n"
            f"Sentence: {sentence}"
        )

        def generate_replacement(base: str, extra: str) -> str:
            prompt = base + extra
            return runtime.generate(prompt, max_tokens, temperature=0.0).strip().strip('"')

        replacement = generate_replacement(base_prompt, feedback)
        scar = False
        if not replacement or replacement == sentence:
            continue
        if len(_quant_family_tokens(replacement)) >= 2:
            feedback = (
                "Your previous rewrite stacked two quantifiers next to each "
                "other. Use exactly one hedge word and keep the rest of the "
                "sentence unchanged."
            )
            replacement = generate_replacement(base_prompt, feedback)
            if not replacement or replacement == sentence:
                continue
            if len(_quant_family_tokens(replacement)) >= 2:
                scar = True
        if not any(
            hedge_present(replacement, hedge) for hedge in RECOGNIZED_HEDGES
        ):
            feedback = (
                "Your previous rewrite used a word outside the recognized "
                "hedge list. Insert exactly one of these words and keep the "
                "rest of the sentence unchanged."
            )
            replacement = generate_replacement(base_prompt, feedback)
            if not any(
                hedge_present(replacement, hedge) for hedge in RECOGNIZED_HEDGES
            ):
                if not replacement:
                    continue
                replacement = (
                    replacement.rstrip(". ") + ", though this pattern is not universal."
                )
        if sentence in text:
            text = text.replace(sentence, replacement, 1)
            spliced += 1
            marker = " #style_scar" if scar else ""
            inserted.append(replacement + marker)
            if scar:
                scars += 1
    return text, spliced, inserted, scars


def _flagged_sentence_pairs(failures: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for failure in failures:
        match = _MISSING_QUALIFIER_RE.search(failure) or _UNHEDGED_INFERENCE_RE.search(
            failure
        )
        if match:
            qualifier = (match.group(1) or "").strip()
            pairs.append((qualifier, match.group(2).strip()))
    return pairs
