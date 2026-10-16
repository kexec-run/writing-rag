"""Evidence-scoped paragraph planning, retrieval, verification, and repair helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from writing_rag.retrieve.facet_retrieve import FacetRetrievalConfig, retrieve_faceted_many
from writing_rag.retrieve.retrieve import EmbeddingSession, lexical_terms, retrieve_many
from writing_rag.plan.article_plan import (  # noqa: E402
    build_article_plan,
    format_scoped_draft_packets,
    sha256_text,
    validate_article_plan,
    validate_claim_trace,
)

from writing_rag.verify.evidence_trace import (
    build_candidate_trace,
    render_debug_trace,
    retrieval_trace_event,
)

from writing_rag.verify.claims import (
    build_claim_verification_prompt,
    build_local_repair_prompt,
    claim_units,
    infer_qualifier,
    merge_paragraph_repairs,
    parse_claim_verification,
    transition_audit,
    transition_failures_for_group,
    verification_failures,
    verifier_protocol_diagnostics,
)


ROLE_ORDER = (
    "observation",
    "motivation",
    "initial_hypothesis",
    "discarded_approach",
    "reframe",
    "experiment",
    "finding",
    "mechanism",
    "decision",
    "conclusion",
)
CLAIM_TYPES = (
    "personal_observation",
    "personal_action",
    "technical_fact",
    "technical_cause",
    "interpretation",
)

TRANSITION_MODES = (
    "answer",
    "extend",
    "redirect",
    "contrast",
    "return_to_narration",
)

DEFAULT_CLAIM_TYPES: dict[str, tuple[str, ...]] = {
    "observation": ("personal_observation", "technical_fact"),
    "motivation": ("personal_action", "interpretation"),
    "initial_hypothesis": ("interpretation",),
    "discarded_approach": ("interpretation",),
    "reframe": ("interpretation",),
    "experiment": ("technical_fact",),
    "finding": ("technical_fact",),
    "mechanism": ("technical_cause",),
    "decision": ("interpretation",),
    "conclusion": ("interpretation",),
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
ATOMIC_TERM_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?(?:x\d+)?|[A-Z]{2,}|[A-Za-z]+[A-Z][A-Za-z0-9]*)\b"
)
QUALIFIER_RE = re.compile(
    r"\b(maybe|could|likely|appears|suggests|uncertain|may|might|less|more)\b",
    re.IGNORECASE,
)
MARKED_PARAGRAPH_RE = re.compile(
    r"\[\[INTENT\s+([^\]]+)\]\]\s*(.*?)\s*\[\[/INTENT\]\]",
    re.IGNORECASE | re.DOTALL,
)
LOOSE_MARKED_PARAGRAPH_RE = re.compile(
    r"\[\[INTENT\s+([^\]]+)\]\]\s*(.*?)(?=\[\[(?:/INTENT|INTENT\s+)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
QUALIFIER_EQUIVALENTS = {
    "maybe": re.compile(
        r"\b(maybe|perhaps|possibly|may|might|could|suggests?|appears?)\b",
        re.IGNORECASE,
    ),
    "may": re.compile(r"\b(may|might|could|possibly|likely)\b", re.IGNORECASE),
    "might": re.compile(r"\b(might|may|could|possibly|likely)\b", re.IGNORECASE),
    "could": re.compile(r"\b(could|may|might|possibly)\b", re.IGNORECASE),
    "likely": re.compile(r"\b(likely|probably|appears?|suggests?)\b", re.IGNORECASE),
    "appears": re.compile(
        r"\b(appears?|seems?|looks?|suggests?|likely|may|might|could)\b",
        re.IGNORECASE,
    ),
    "suggests": re.compile(
        r"\b(suggests?|indicates?|implies?|appears?|seems?|likely)\b",
        re.IGNORECASE,
    ),
    "uncertain": re.compile(
        r"\b(uncertain|unclear|not sure|possibly|perhaps|may|might|could)\b",
        re.IGNORECASE,
    ),

    "ambiguous details": re.compile(
        r"\b(ambiguous|unclear|inaccessible|uncertainty|uncertain)\b",
        re.IGNORECASE,
    ),
    "reported_hypothesis": re.compile(
        r"\b(suspicion|suspected|hypothesis|assumed|initially thought)\b",
        re.IGNORECASE,
    ),
    "does not establish": re.compile(
        r"\b(does not|doesn't|cannot|can't)\s+(?:fully\s+)?establish\b",
        re.IGNORECASE,
    ),
    "inference_marker": re.compile(
        r"\b(suggest(?:s|ed|ing)?|indicat(?:e|es|ed|ing)|impl(?:y|ies|ied|ying))\b",
        re.IGNORECASE,
    ),
    "about": re.compile(r"\b(about|approximately|roughly)\b", re.IGNORECASE),
    "current sample set": re.compile(
        r"\b(current\s+(?:sample|test)\s+set|tested\s+(?:samples|set))\b",
        re.IGNORECASE,
    ),
    "not universal": re.compile(
        r"\b(not\s+(?:a\s+)?universal|specific\s+to\s+(?:the\s+)?(?:tested|current))\b",
        re.IGNORECASE,
    ),
    "not numerically close": re.compile(
        r"\bnot\s+numerically\s+close\b",
        re.IGNORECASE,
    ),
}
CONTROL_LEAK_RE = re.compile(
    r"(?im)^(?:#{1,6}\s+intent-\d+\b|Question:|Candidate claim:|"
    r"Required qualifiers:|Evidence:|Failures:|Current paragraph:)"
)


def _field(parts: list[str], name: str) -> str:
    prefix = f"{name}="
    for part in parts:
        if part.strip().casefold().startswith(prefix):
            return part.strip()[len(prefix) :].strip()
    return ""


def _field_any(parts: list[str], *names: str) -> str:
    for name in names:
        value = _field(parts, name)
        if value:
            return value
    return ""


def _parse_list(value: str) -> list[str]:
    cleaned = value.strip().strip("[](){}")
    return [
        item.strip()
        for item in re.split(r"[,;]", cleaned)
        if item.strip() and item.strip().casefold() not in {"none", "null", "nil", "-"}
    ]


def _normalize_claim_types(role: str, value: str = "") -> list[str]:
    requested = [
        item.casefold().replace("-", "_").replace(" ", "_")
        for item in _parse_list(value)
    ]
    selected = [item for item in requested if item in CLAIM_TYPES]
    if selected:
        return list(dict.fromkeys(selected))
    return list(DEFAULT_CLAIM_TYPES.get(role, ("interpretation",)))


def _infer_transition_mode(previous_role: str, role: str) -> str:
    if role in {"mechanism", "finding", "conclusion"}:
        return "answer"
    if role in {"observation", "motivation"} and previous_role in {
        "finding",
        "mechanism",
    }:
        return "return_to_narration"
    if role in {"discarded_approach", "reframe"}:
        return "redirect"
    return "extend"


def apply_intent_contract_defaults(
    intents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill a flexible beat contract without changing explicit model decisions."""

    def normalize_dependency(value: str) -> str:
        cleaned = value.strip()
        match = re.fullmatch(r"(?:intent[\s_-]*)?(\d+)", cleaned, re.IGNORECASE)
        if match:
            return f"intent-{int(match.group(1)):03d}"
        return cleaned

    for index, intent in enumerate(intents):
        role = str(intent.get("role", ""))
        intent["claim_types"] = _normalize_claim_types(
            role,
            str(intent.get("claim_types_raw", "")),
        )
        intent["opens_question"] = str(
            intent.get("opens_question") or intent.get("question") or ""
        ).strip()
        intent["closes_question"] = str(
            intent.get("closes_question")
            or intent.get("intended_claim")
            or intent.get("question")
            or ""
        ).strip()
        if index and not intent.get("depends_on"):
            intent["depends_on"] = [intents[index - 1]["intent_id"]]
        else:
            intent["depends_on"] = list(
                dict.fromkeys(
                    normalize_dependency(str(value))
                    for value in intent.get("depends_on", [])
                    if str(value).strip().casefold() not in {"", "none", "null"}
                )
            )
        previous_role = str(intents[index - 1].get("role", "")) if index else ""
        transition_mode = str(intent.get("transition_mode") or "").casefold()
        intent["transition_mode"] = (
            transition_mode
            if transition_mode in TRANSITION_MODES
            else (_infer_transition_mode(previous_role, role) if index else "")
        )
        intent.pop("claim_types_raw", None)
    return intents


def parse_paragraph_intents(outline: str) -> list[dict[str, Any]]:
    """Parse strict INTENT lines while accepting legacy role-labelled bullets."""
    intents: list[dict[str, Any]] = []
    for line in outline.splitlines():
        stripped = line.strip().lstrip("-* ")
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split("|")]
        role = _field(parts, "role").casefold().replace("-", "_").replace(" ", "_")
        if (
            not role
            and len(parts) > 1
            and re.fullmatch(r"INTENT\s+\d+", parts[0], re.IGNORECASE)
        ):
            bare_role = parts[1].strip().casefold().replace("-", "_").replace(" ", "_")
            if bare_role in ROLE_ORDER:
                role = bare_role
        if role not in ROLE_ORDER:
            head = parts[0].casefold().replace("-", "_").replace(" ", "_")
            role = next(
                (
                    candidate
                    for candidate in ROLE_ORDER
                    if head.startswith(candidate)
                    or head.startswith(f"{candidate}:")
                    or head.startswith(f"{candidate}=")
                ),
                "",
            )
        if not role:
            continue

        refs_value = _field(parts, "refs")
        refs = _parse_list(refs_value) if refs_value else []
        if not refs:
            for bracketed in re.findall(r"\[([^\]]+)\]", stripped):
                refs.extend(
                    value.strip()
                    for value in bracketed.split(",")
                    if value.strip() and not value.strip().isdigit()
                )
        question = _field(parts, "question")
        claim = _field(parts, "claim")
        if not claim:
            claim = re.sub(r"\[[^\]]+\]", "", parts[-1]).strip(" :-")
            claim = re.sub(
                r"^(?:INTENT\s+\d+\s*\|\s*)?", "", claim, flags=re.IGNORECASE
            )
            claim = re.sub(
                rf"^(?:role=)?{re.escape(role)}\s*[:=]?\s*",
                "",
                claim,
                flags=re.IGNORECASE,
            )
        if not question:
            question = claim or f"What evidence supports the {role.replace('_', ' ')}?"
        contract_fields = (
            "claims",
            "claim_types",
            "depends_on",
            "opens",
            "opens_question",
            "closes",
            "closes_question",
            "transition",
            "transition_mode",
        )
        intents.append(
            {
                "intent_id": f"intent-{len(intents) + 1:03d}",
                "order": len(intents) + 1,
                "role": role,
                "question": question,
                "intended_claim": claim,
                "outline_document_ids": list(dict.fromkeys(refs)),
                "claim_types_raw": _field_any(parts, "claims", "claim_types"),
                "depends_on": _parse_list(_field_any(parts, "depends_on", "depends")),
                "opens_question": _field_any(parts, "opens", "opens_question"),
                "closes_question": _field_any(parts, "closes", "closes_question"),
                "transition_mode": _field_any(parts, "transition", "transition_mode"),
                "_contract_explicit": any(
                    _field(parts, name) for name in contract_fields
                ),
            }
        )
    return apply_intent_contract_defaults(intents)


def intents_from_scaffold(scaffold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provide a deterministic fallback when a model outline is not machine-readable."""
    intents = []
    for item in scaffold:
        if item.get("status") != "retrieved":
            continue
        evidence_ids = [
            result["document"]["document_id"] for result in item.get("evidence", [])
        ]
        intents.append(
            {
                "intent_id": f"intent-{len(intents) + 1:03d}",
                "order": len(intents) + 1,
                "role": item["role"],
                "question": item["question"],
                "intended_claim": item["question"],
                "outline_document_ids": list(dict.fromkeys(evidence_ids)),
                "claim_types_raw": "",
                "depends_on": [],
                "opens_question": item["question"],
                "closes_question": item["question"],
                "transition_mode": "",
                "_contract_explicit": False,
            }
        )
    return apply_intent_contract_defaults(intents)


def parse_editorial_points(template: str) -> list[str]:
    """Extract numbered reader-facing coverage points, including wrapped lines."""
    points: list[str] = []
    current: list[str] = []
    for line in template.splitlines():
        match = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if match:
            if current:
                points.append(" ".join(current))
            current = [match.group(1).strip()]
        elif (
            current and (line.startswith(" ") or line.startswith("\t")) and line.strip()
        ):
            current.append(line.strip())
        elif current:
            points.append(" ".join(current))
            current = []
    if current:
        points.append(" ".join(current))
    return points


def _editorial_role(point: str) -> str:
    normalized = point.casefold()
    rules = (
        ("why the artifact mattered", "motivation"),
        ("guessing was not enough", "discarded_approach"),
        ("first principle", "reframe"),
        ("the accident", "observation"),
        ("problem:", "observation"),
        ("result:", "finding"),
        ("triggered", "mechanism"),
        ("does input", "experiment"),
        ("graphs and metrics", "finding"),
        ("reproduction options", "decision"),
    )
    return next((role for marker, role in rules if marker in normalized), "finding")


def intents_from_editorial_points(
    points: list[str], outline_intents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Make explicit editorial coverage primary while retaining model role hints."""
    intents = []
    for index, point in enumerate(points, start=1):
        point_terms = set(WORD_RE.findall(point.casefold()))
        best = max(
            outline_intents,
            key=lambda item: len(
                point_terms
                & set(
                    WORD_RE.findall(
                        f"{item.get('question', '')} {item.get('intended_claim', '')}".casefold()
                    )
                )
            ),
            default=None,
        )
        overlap = (
            len(
                point_terms
                & set(
                    WORD_RE.findall(
                        f"{best.get('question', '')} {best.get('intended_claim', '')}".casefold()
                    )
                )
            )
            if best
            else 0
        )
        intents.append(
            {
                "intent_id": f"intent-{index:03d}",
                "order": index,
                "role": best["role"]
                if best and overlap >= 2
                else _editorial_role(point),
                "question": point,
                "intended_claim": point,
                "outline_document_ids": (
                    best.get("outline_document_ids", [])
                    if best and overlap >= 2
                    else []
                ),
                "claim_types_raw": (
                    ",".join(best.get("claim_types", []))
                    if best and overlap >= 2
                    else ""
                ),
                "depends_on": [],
                "opens_question": point,
                "closes_question": point,
                "transition_mode": "",
                "editorial_point": index,
                "_contract_explicit": False,
            }
        )
    return apply_intent_contract_defaults(intents)


def _intent_retrieval_query(
    query: str,
    intent: dict[str, Any],
    *,
    extra: str = "",
) -> str:
    """Build a query from user intent, not the model's unverified claim."""
    return " ".join(
        part
        for part in (
            query,
            intent.get("question", ""),
            " ".join(intent.get("claim_types", [])),
            str(intent.get("role", "")).replace("_", " "),
            extra,
        )
        if part
    )


def _intent_anchor_query(query: str, intent: dict[str, Any]) -> str:
    """Build a compact query for local anchor selection."""
    terms = lexical_terms(intent.get("question", "")) - ANCHOR_GENERIC_TERMS
    return " ".join(sorted(terms or lexical_terms(query)))


ANCHOR_GENERIC_TERMS = {
    "answer",
    "assumes",
    "assumption",
    "assumptions",
    "does",
    "each",
    "exist",
    "exists",
    "fact",
    "function",
    "functions",
    "inherent",
    "mechanism",
    "order",
    "property",
    "question",
    "result",
    "something",
    "statement",
    "stated",
    "thing",
    "things",
    "truly",
    "understand",
    "understanding",
    "what",
}


def _intent_anchor_terms(query: str, intent: dict[str, Any]) -> set[str]:
    """Keep topic and claim-bearing terms for local anchor validation."""
    terms = lexical_terms(
        " ".join(
            part
            for part in (
                query,
                intent.get("question", ""),
                intent.get("intended_claim", ""),
            )
            if part
        )
    )
    return terms - ANCHOR_GENERIC_TERMS


def _select_intent_evidence(
    intent: dict[str, Any],
    retrieval_query: str,
    candidates: list[dict[str, Any]],
    *,
    span_selector: Callable[[str, str], str],
    top_k: int,
    minimum_score: float,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select evidence by evidence score, not by planner proximity.

    ``outline_document_ids`` are a bounded planning hint. They are not an
    evidence allowlist and must not outrank a stronger retrieved result. In
    particular, faceted retrieval exposes an RRF rank score separately from
    the underlying hybrid ``evidence_score``; using the RRF score here would
    turn structural proximity into apparent relevance.
    """
    preferred_ids = set(intent.get("outline_document_ids", []))
    excluded = excluded_ids or set()
    prepared: list[tuple[dict[str, Any], str, float, float]] = []
    query_terms = lexical_terms(retrieval_query)
    for result in candidates:
        document_id = result["document"]["document_id"]
        evidence_score = float(result.get("evidence_score", result.get("score", 0.0)))
        if document_id in excluded or evidence_score < minimum_score:
            continue
        source_span = span_selector(result["document"]["text"], retrieval_query)
        if not source_span:
            continue
        span_terms = lexical_terms(source_span)
        span_overlap = len(query_terms & span_terms) / max(1, len(query_terms))
        prepared.append((result, source_span, evidence_score, span_overlap))

    # The score that represents source relevance is primary. Span overlap is
    # only a deterministic tie-breaker; the lexical signal cannot override a
    # semantically stronger result. Planner references are the final tie-break
    # and therefore cannot inject structural neighbours as evidence.
    prepared.sort(
        key=lambda item: (
            -item[2],
            -item[3],
            item[0]["document"]["document_id"] not in preferred_ids,
            -float(item[0].get("score", 0.0)),
            item[0]["document"]["document_id"],
        )
    )
    evidence = []
    for result, source_span, _evidence_score, _span_overlap in prepared:
        document_id = result["document"]["document_id"]
        copied = dict(result)
        copied["source_span"] = source_span
        evidence.append(copied)
        if len(evidence) >= top_k:
            break
    return evidence


def _refresh_intent_evidence_metadata(intent: dict[str, Any]) -> None:
    evidence = intent.get("evidence", [])
    spans = [result.get("source_span", "") for result in evidence]
    intent["evidence_ids"] = [result["document"]["document_id"] for result in evidence]
    intent["evidence_allowlist"] = sorted(set(intent["evidence_ids"]))
    intent["required_qualifiers"] = sorted(
        {
            match.group(0).casefold()
            for span in spans
            for match in QUALIFIER_RE.finditer(span)
        }
    )
    if evidence:
        intent["status"] = "candidate_supported"
    elif intent.get("status") != "retrieved":
        intent["status"] = "unsupported"


def _attach_plan_source_spans(
    ledger: list[dict[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    """Bridge retrieved evidence records to the validated plan spans.

    Paragraph retrieval stores the selected excerpt as ``source_span``.  The
    article plan stores canonical, hash-checked spans under ``source_spans``.
    Keep both representations: the excerpt remains the verifier display text,
    while this field supplies the canonical provenance IDs for tracing.
    """

    spans_by_document = {
        str(packet.get("document_id")): [
            dict(span) for span in packet.get("source_spans", [])
        ]
        for packet in plan.get("source_packets", [])
    }
    for intent in ledger:
        for evidence in intent.get("evidence", []):
            document = evidence.get("document", {})
            document_id = str(document.get("document_id", ""))
            evidence["source_spans"] = [
                dict(span) for span in spans_by_document.get(document_id, [])
            ]


def _retrieve_intent_candidates(
    retrieval_queries: list[str],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any] | None,
    *,
    session: EmbeddingSession,
    top_k: int,
    candidate_top_k: int | None = None,
    retrieval_strategy: str,
    facet_config: FacetRetrievalConfig | None,
    anchor_queries: list[str] | None = None,
    anchor_term_sets: list[set[str]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Retrieve intent candidates while keeping RRF scores separate from evidence scores."""
    candidate_top_k = candidate_top_k or max(top_k * 3, 8)
    if retrieval_strategy == "baseline":
        return retrieve_many(
            retrieval_queries,
            documents,
            vectors,
            top_k=candidate_top_k,
            collection="content_memory",
            embedding_metadata=embedding_metadata,
            session=session,
        )
    if retrieval_strategy == "faceted":
        config = replace(
            facet_config or FacetRetrievalConfig(),
            top_k=candidate_top_k,
        )
        return retrieve_faceted_many(
            retrieval_queries,
            documents,
            vectors,
            config=config,
            collection="content_memory",
            embedding_metadata=embedding_metadata,
            session=session,
            anchor_queries=anchor_queries,
            anchor_term_sets=anchor_term_sets,
        )
    raise ValueError(
        "retrieval_strategy must be 'baseline' or 'faceted'"
    )


def retrieve_intent_evidence(
    query: str,
    intents: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any] | None,
    *,
    session: EmbeddingSession,
    span_selector: Callable[[str, str], str],
    top_k: int = 3,
    minimum_score: float = 0.20,
    retrieval_strategy: str = "baseline",
    facet_config: FacetRetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    """Retrieve a small, independent evidence set for every paragraph intent."""
    retrieval_queries = [_intent_retrieval_query(query, intent) for intent in intents]
    anchor_queries = [_intent_anchor_query(query, intent) for intent in intents]
    result_sets = _retrieve_intent_candidates(
        retrieval_queries,
        documents,
        vectors,
        embedding_metadata=embedding_metadata,
        session=session,
        top_k=top_k,
        retrieval_strategy=retrieval_strategy,
        facet_config=facet_config,
        anchor_queries=anchor_queries,
        anchor_term_sets=[_intent_anchor_terms(query, intent) for intent in intents],
    )
    ledger = []
    for intent, retrieval_query, candidates in zip(
        intents, retrieval_queries, result_sets
    ):
        evidence = _select_intent_evidence(
            intent,
            retrieval_query,
            candidates,
            span_selector=span_selector,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        candidate_trace = build_candidate_trace(
            intent,
            retrieval_query,
            candidates,
            span_selector=span_selector,
            minimum_score=minimum_score,
            selected_ids={
                result["document"]["document_id"] for result in evidence
            },
        )
        copied_intent = {
            **intent,
            "retrieval_query": retrieval_query,
            "retrieval_strategy": retrieval_strategy,
            "status": "candidate_supported" if evidence else "unsupported",
            "evidence": evidence,
            "retrieval_trace": [
                retrieval_trace_event(
                    intent=intent,
                    query=retrieval_query,
                    candidates=candidates,
                    candidate_trace=candidate_trace,
                    round_number=0,
                    top_k=top_k,
                    minimum_score=minimum_score,
                )
            ],
        }
        _refresh_intent_evidence_metadata(copied_intent)
        ledger.append(copied_intent)
    return ledger


def _retrieval_retry_requested(reasons: list[str]) -> bool:
    markers = (
        "insufficient_evidence",
        "unbacked",
        "unsupported",
        "no evidence",
        "lacks evidence",
        "contradicted",
    )
    normalized = " ".join(reasons).casefold()
    return any(marker in normalized for marker in markers)


def retry_intent_evidence(
    query: str,
    ledger: list[dict[str, Any]],
    failures: dict[str, list[str]],
    reports_by_id: dict[str, dict[str, Any]],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any] | None,
    *,
    session: EmbeddingSession,
    span_selector: Callable[[str, str], str],
    top_k: int = 3,
    minimum_score: float = 0.20,
    retry_round: int = 1,
    retrieval_strategy: str = "baseline",
    facet_config: FacetRetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    """Retrieve again only for claims whose evidence audit found a support gap."""
    targets = [
        intent
        for intent in ledger
        if intent["intent_id"] in failures
        and _retrieval_retry_requested(failures[intent["intent_id"]])
    ]
    if not targets:
        return []

    retrieval_queries = []
    for intent in targets:
        failed_claims = [
            str(claim.get("text", ""))
            for claim in reports_by_id.get(intent["intent_id"], {}).get("claims", [])
            if claim.get("relation") in {"insufficient_evidence", "contradicted"}
        ]
        extra = " ".join(
            ["retrieve direct support for the failed claim", *failed_claims]
        )
        retrieval_queries.append(_intent_retrieval_query(query, intent, extra=extra))

    result_sets = _retrieve_intent_candidates(
        retrieval_queries,
        documents,
        vectors,
        embedding_metadata=embedding_metadata,
        session=session,
        top_k=max(top_k, 4),
        candidate_top_k=max(top_k * 5, 12),
        retrieval_strategy=retrieval_strategy,
        facet_config=facet_config,
        anchor_queries=[_intent_anchor_query(query, intent) for intent in targets],
        anchor_term_sets=[_intent_anchor_terms(query, intent) for intent in targets],
    )
    retry_log: list[dict[str, Any]] = []
    for intent, retrieval_query, candidates in zip(
        targets, retrieval_queries, result_sets
    ):
        existing_ids = set(intent.get("evidence_ids", []))
        added = _select_intent_evidence(
            intent,
            retrieval_query,
            candidates,
            span_selector=span_selector,
            top_k=max(top_k, 2),
            minimum_score=minimum_score,
            excluded_ids=existing_ids,
        )
        if added:
            intent["evidence"] = [*intent.get("evidence", []), *added]
        intent["retrieval_retry_count"] = retry_round
        intent["retrieval_retry_query"] = retrieval_query
        intent["retrieval_strategy"] = retrieval_strategy
        retry_candidate_trace = build_candidate_trace(
            intent,
            retrieval_query,
            candidates,
            span_selector=span_selector,
            minimum_score=minimum_score,
            selected_ids={
                result["document"]["document_id"] for result in added
            },
            excluded_ids=existing_ids,
        )
        intent.setdefault("retrieval_trace", []).append(
            retrieval_trace_event(
                intent=intent,
                query=retrieval_query,
                candidates=candidates,
                candidate_trace=retry_candidate_trace,
                round_number=retry_round,
                top_k=max(top_k, 2),
                minimum_score=minimum_score,
            )
        )
        _refresh_intent_evidence_metadata(intent)
        retry_log.append(
            {
                "intent_id": intent["intent_id"],
                "round": retry_round,
                "query": retrieval_query,
                "added_document_ids": [
                    result["document"]["document_id"] for result in added
                ],
                "candidate_count": len(candidates),
                "evidence_count": len(intent.get("evidence", [])),
            }
        )
    return retry_log


def ledger_content_results(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deduplicated union for final draft validation and provenance."""
    by_id: dict[str, dict[str, Any]] = {}
    for intent in ledger:
        for result in intent.get("evidence", []):
            document_id = result["document"]["document_id"]
            current = by_id.get(document_id)
            if current is None or result["score"] > current["score"]:
                by_id[document_id] = result
    return sorted(
        by_id.values(),
        key=lambda result: (-result["score"], result["document"]["document_id"]),
    )


def group_intents(
    ledger: list[dict[str, Any]], group_size: int = 1
) -> list[list[dict[str, Any]]]:
    supported = [intent for intent in ledger if intent["status"] != "unsupported"]
    return [
        supported[index : index + group_size]
        for index in range(0, len(supported), group_size)
    ]


def format_intent_evidence(intents: list[dict[str, Any]]) -> str:
    blocks = []
    for intent in intents:
        evidence_lines = [
            f"- [{result['document']['document_id']}] {result['source_span']}"
            for result in intent.get("evidence", [])
        ] or ["- no evidence; omit this intent"]
        dependencies = ", ".join(intent.get("depends_on", [])) or "none"
        blocks.append(
            f"### {intent['intent_id']} | {intent['role']}\n"
            f"Reader question: {intent['question']}\n"
            f"Claim types: {', '.join(intent.get('claim_types', [])) or 'interpretation'}\n"
            f"Opens: {intent.get('opens_question') or intent['question']}\n"
            f"Closes: {intent.get('closes_question') or intent.get('intended_claim', '')}\n"
            f"Depends on: {dependencies}\n"
            f"Transition: {intent.get('transition_mode') or 'none'}\n"
            f"Candidate claim: {intent.get('intended_claim', '')}\n"
            f"Required qualifiers: {', '.join(intent.get('required_qualifiers', [])) or 'none'}\n"
            "Evidence:\n" + "\n".join(evidence_lines)
        )
    return "\n\n".join(blocks)


def format_intent_contract(intents: list[dict[str, Any]]) -> str:
    """Render authorial beat metadata without duplicating source evidence."""
    blocks = []
    for intent in intents:
        dependencies = ", ".join(intent.get("depends_on", [])) or "none"
        blocks.append(
            f"### {intent['intent_id']} | {intent['role']}\n"
            f"Reader question: {intent['question']}\n"
            f"Claim types: {', '.join(intent.get('claim_types', [])) or 'interpretation'}\n"
            f"Opens: {intent.get('opens_question') or intent['question']}\n"
            f"Closes: {intent.get('closes_question') or intent.get('intended_claim', '')}\n"
            f"Depends on: {dependencies}\n"
            f"Transition: {intent.get('transition_mode') or 'none'}\n"
            f"Authorial target: {intent.get('intended_claim', '')}\n"
            f"Required qualifiers: {', '.join(intent.get('required_qualifiers', [])) or 'none'}\n"
            "Source excerpts are supplied only in the scoped packets above; "
            "this beat contract is not source evidence."
        )
    return "\n\n".join(blocks)


def build_planning_inventory(
    results: list[dict[str, Any]],
    *,
    query: str,
    span_selector: Callable[[str, str], str],
    max_items: int = 24,
) -> tuple[str, set[str]]:
    """Expose a compact, diversified source map to planning without drafting from it."""
    selected = []
    seen_sections: set[tuple[str, str]] = set()
    for result in results:
        document = result["document"]
        metadata = document.get("metadata", {})
        section_key = (
            str(metadata.get("source_path", metadata.get("source_thread_id", ""))),
            str(metadata.get("heading", metadata.get("source_branch_id", ""))),
        )
        if section_key in seen_sections and len(selected) >= max_items // 2:
            continue
        seen_sections.add(section_key)
        selected.append(result)
        if len(selected) >= max_items:
            break
    lines = []
    ids = set()
    for result in selected:
        document = result["document"]
        metadata = document.get("metadata", {})
        document_id = document["document_id"]
        ids.add(document_id)
        label = (
            metadata.get("heading")
            or metadata.get("source_path")
            or metadata.get("source_thread_id")
            or "source"
        )
        span = span_selector(document["text"], query)
        lines.append(f"- [{document_id}] {label}: {span}")
    return "\n".join(lines), ids


def build_paragraph_group_prompt(
    intents: list[dict[str, Any]],
    *,
    query: str,
    style_profile: str,
    narrative_template: str,
    previous_tail: str = "",
    article_state: str = "",
    target_words_per_intent: int = 80,
    document_type: str = "investigative",
    draft_plan_context: str = "",
    draft_context_policy: str = "legacy",
) -> str:
    ids = ", ".join(intent["intent_id"] for intent in intents)
    document_form = (
        "MDX explanatory article. The reader question is planning metadata, not a required "
        "rhetorical question or heading. Prefer a definition, distinction, direct claim, or "
        "consequence as the opening form. Use a question only when the evidence naturally "
        "requires it."
        if document_type.casefold() == "mdx"
        else "Investigative article. The reader question may be used as an opening when it "
        "matches the evidence and the narrative contract."
    )
    if draft_context_policy in {
        "exploratory_v1",
        "episode_slice_v1",
        "episode_slice_c3_neighbor_v1",
    }:
        plan_context_section = (
            "Draft-time scoped source excerpts (optional grounding cues):\n"
            + draft_plan_context.strip()
            + "\n\n"
            if draft_plan_context.strip()
            else "Draft-time scoped source excerpts: none selected.\n\n"
        )
        draft_context_guidance = (
            "Use the scoped excerpts as optional grounding cues, not as a second outline. "
            "Before a claim, check whether an excerpt states it, suggests it, or does not support it. "
            "If a beat lacks support, use a short neutral transition or omit the unsupported detail. "
            "Match any hedge to the excerpt, and do not infer a mechanism merely because the outline asks for one. "
            "Do not convert adjacency, timestamps, or metadata into causality.\n\n"
        )
        intent_section_label = "Intent beat contract:"
        intent_section = format_intent_contract(intents)
    else:
        plan_context_section = (
            "Draft-time source-backed narrative packets:\n"
            + draft_plan_context.strip()
            + "\n\n"
            if draft_plan_context.strip()
            else ""
        )
        draft_context_guidance = ""
        intent_section_label = "Intent beat ledger:"
        intent_section = format_intent_evidence(intents)
    return (
        "Draft only the requested connected paragraph group. Content evidence controls every claim; "
        "the style profile controls voice only. The candidate claim is a retrieval intention, not evidence. "
        "Omit unsupported material, preserve qualifiers, and do not strengthen adjacency into causality. "
        "If the evidence does not directly answer an intent, write only a short neutral transition or omit "
        "the intent; do not turn the reader question into a factual answer. Do not repeat a prior paragraph's "
        "conclusion as if it answered a new question. "
        "Each factual or causal statement must be supported by the evidence under its own intent. "
        "Treat each intent as a beat contract: preserve its order, answer its reader question, and "
        "respect its claim types. Use the declared dependency and transition mode to connect paragraphs. "
        "A transition may be implicit; do not add a causal connector unless the evidence supports it. "
        "For an explicit answer, redirect, or contrast transition, either connect to a term in the preceding "
        "beat or use a restrained discourse marker such as 'however' or 'instead'. If neither is warranted, "
        "use an additive transition rather than inventing a relationship. "
        "A later intent may refer back, but an earlier intent must not introduce a fact owned by a later intent.\n\n"
        f"Return exactly one block for each of these intents, in order: {ids}. Use this format:\n"
        "[[INTENT intent-001]]\nParagraph text only.\n[[/INTENT]]\n"
        f"Aim for about {target_words_per_intent} words per intent when its evidence supports that detail. "
        "A block may begin with a concise Markdown heading when it starts a new reader-facing section. "
        "Do not include citations, document IDs, analysis, or any text outside the blocks.\n\n"
        f"Document form:\n{document_form}\n\n"
        f"Article goal:\n{query}\n\n"
        f"Narrative contract:\n{narrative_template.strip()}\n\n"
        f"Derived style profile:\n{style_profile.strip()}\n\n"
        f"Already-covered article state:\n{article_state or 'none'}\n\n"
        f"Previous accepted paragraph for continuity only:\n{previous_tail or 'none'}\n\n"
        + plan_context_section
        + draft_context_guidance
        + intent_section_label
        + "\n"
        + intent_section
    )


def parse_paragraph_group_output(
    output: str, expected_ids: list[str]
) -> dict[str, str]:
    def clean(text: str) -> bool:
        return (
            bool(text.strip())
            and not CONTROL_LEAK_RE.search(text)
            and "[[INTENT" not in text.upper()
        )

    marked = {
        intent_id.strip(): text.strip()
        for intent_id, text in MARKED_PARAGRAPH_RE.findall(output)
        if intent_id.strip() in expected_ids and clean(text)
    }
    if marked:
        return marked
    loose_marked = {
        intent_id.strip(): text.strip()
        for intent_id, text in LOOSE_MARKED_PARAGRAPH_RE.findall(output)
        if intent_id.strip() in expected_ids and clean(text)
    }
    if loose_marked:
        return loose_marked
    if "[[INTENT" in output.upper() or "[[/INTENT" in output.upper():
        return {}
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", output) if part.strip()]
    if not paragraphs:
        return {}
    if len(expected_ids) == 1:
        return {expected_ids[0]: output.strip()} if clean(output) else {}
    if len(paragraphs) >= len(expected_ids):
        return {
            intent_id: (
                paragraphs[index]
                if index < len(expected_ids) - 1
                else "\n\n".join(paragraphs[index:])
            )
            for index, intent_id in enumerate(expected_ids)
        }
    return {expected_ids[0]: output.strip()} if clean(output) else {}




def dependency_evidence_audit(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit source overlap without treating it as causal proof.

    A dependency is a planning relation, not proof of causality. Shared source
    IDs show only source overlap. Causal support requires an explicit relation
    record, which this ledger does not contain, so it remains false here.
    Endpoint-only edges remain visible as demoted diagnostics.
    """
    by_id = {str(intent.get("intent_id")): intent for intent in ledger}
    positions = {
        str(intent.get("intent_id")): index for index, intent in enumerate(ledger)
    }
    edges: list[dict[str, Any]] = []
    for intent in ledger:
        intent_id = str(intent.get("intent_id", ""))
        current_ids = set(intent.get("evidence_ids", []))
        for dependency in intent.get("depends_on", []):
            dependency_id = str(dependency)
            dependency_item = by_id.get(dependency_id)
            valid_order = (
                dependency_item is not None
                and positions.get(dependency_id, len(ledger))
                < positions.get(intent_id, 0)
            )
            dependency_ids = (
                set(dependency_item.get("evidence_ids", []))
                if dependency_item
                else set()
            )
            shared_ids = sorted(current_ids & dependency_ids)
            if not valid_order:
                status = "invalid"
            elif shared_ids:
                status = "shared_source_only"
            else:
                status = "endpoint_only"
            edges.append(
                {
                    "from_intent": dependency_id,
                    "to_intent": intent_id,
                    "status": status,
                    "shared_source_ids": shared_ids,
                    "from_evidence_ids": sorted(dependency_ids),
                    "to_evidence_ids": sorted(current_ids),
                    "source_overlap": bool(shared_ids),
                    "relation_evidence_ids": [],
                    "causal_support": False,
                    "causal_support_reason": (
                        "shared source IDs do not establish a causal relation"
                        if shared_ids
                        else "no explicit relation evidence"
                    ),
                }
            )
    counts = {
        status: sum(edge["status"] == status for edge in edges)
        for status in ("shared_source_only", "endpoint_only", "invalid")
    }
    return {
        "edge_count": len(edges),
        "shared_source_only_edge_count": counts["shared_source_only"],
        "endpoint_only_edge_count": counts["endpoint_only"],
        "invalid_edge_count": counts["invalid"],
        "edges": edges,
        "causal_support_is_explicit_only": True,
        "causal_support_requires_relation_evidence": True,
    }


def article_state(intents: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {intent['role']}: {intent.get('intended_claim') or intent['question']}"
        for intent in intents
    )


def evidence_fallback_paragraphs(intents: list[dict[str, Any]]) -> dict[str, str]:
    """Build source-only paragraph text when a draft call fails."""
    fallbacks: dict[str, str] = {}
    for intent in intents:
        spans: list[str] = []
        seen: set[str] = set()
        for result in intent.get("evidence", []):
            span = str(result.get("source_span") or "").strip()
            if span and span not in seen:
                spans.append(span)
                seen.add(span)
        if spans:
            fallbacks[intent["intent_id"]] = " ".join(spans)
    return fallbacks


def ledger_summary(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "intent_id": intent["intent_id"],
            "order": intent["order"],
            "role": intent["role"],
            "question": intent["question"],
            "intended_claim": intent.get("intended_claim", ""),
            "status": intent["status"],
            "evidence_ids": intent["evidence_ids"],
            "required_qualifiers": intent["required_qualifiers"],
            "claim_types": intent.get("claim_types", []),
            "depends_on": intent.get("depends_on", []),
            "transition_mode": intent.get("transition_mode", ""),
            "retrieval_retry_count": intent.get("retrieval_retry_count", 0),
            "retrieval_trace": intent.get("retrieval_trace", []),
        }
        for intent in ledger
    ]


def _atomic_terms(text: str) -> set[str]:
    return {match.group(0).casefold() for match in ATOMIC_TERM_RE.finditer(text)}


def audit_generation_invariants(
    ledger: list[dict[str, Any]],
    paragraphs: dict[str, str],
    reports_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare the declared paragraph skeleton with the generated claims.

    The ledger is the top-down set of eligible intent points. Paragraphs and
    verifier claims are the bottom-up realization. Set differences are kept as
    evidence for repair and debugging; lexical role checks are intentionally
    outside this acceptance audit.
    """
    eligible = [intent for intent in ledger if intent.get("status") != "unsupported"]
    top_order = tuple(intent["intent_id"] for intent in eligible)
    bottom_order = tuple(paragraphs)
    top_ids = set(top_order)
    bottom_ids = set(bottom_order)
    top_roles = {intent["intent_id"]: intent.get("role", "") for intent in eligible}
    bottom_roles = {
        intent_id: top_roles.get(intent_id, "") for intent_id in bottom_order
    }
    declared_edges = {
        (dependency, intent["intent_id"])
        for intent in eligible
        for dependency in intent.get("depends_on", [])
        if dependency in top_ids
    }
    adjacency_edges = {(left, right) for left, right in zip(top_order, top_order[1:])}
    top_edges = declared_edges or adjacency_edges
    bottom_declared_edges = {
        (dependency, intent_id)
        for intent_id in bottom_order
        for dependency in next(
            (
                item.get("depends_on", [])
                for item in eligible
                if item["intent_id"] == intent_id
            ),
            [],
        )
        if dependency in bottom_ids
    }
    bottom_edges = bottom_declared_edges or {
        (left, right) for left, right in zip(bottom_order, bottom_order[1:])
    }
    top_refs_by_id = {
        intent["intent_id"]: set(intent.get("evidence_ids", [])) for intent in eligible
    }
    bottom_refs_by_id = {
        intent_id: {
            evidence_id
            for claim in reports_by_id.get(intent_id, {}).get("claims", [])
            for evidence_id in claim.get("evidence_ids", [])
        }
        for intent_id in bottom_order
    }
    sentence_units_by_id = {
        intent_id: claim_units(paragraph) for intent_id, paragraph in paragraphs.items()
    }
    expected_sentence_keys = {
        (intent_id, number)
        for intent_id, sentences in sentence_units_by_id.items()
        for number in range(1, len(sentences) + 1)
    }
    reported_sentence_keys = {
        (intent_id, number)
        for intent_id, report in reports_by_id.items()
        for number, _claim in enumerate(report.get("claims", []), start=1)
    }
    top_atoms_by_id = {
        intent["intent_id"]: _atomic_terms(intent.get("intended_claim", ""))
        for intent in eligible
    }
    bottom_atoms_by_id = {
        intent_id: _atomic_terms(paragraph)
        for intent_id, paragraph in paragraphs.items()
    }
    evidence_atoms_by_id = {
        intent["intent_id"]: _atomic_terms(
            " ".join(
                result.get("source_span", "") for result in intent.get("evidence", [])
            )
        )
        for intent in eligible
    }
    missing_atoms_by_id = {
        intent_id: sorted(
            top_atoms_by_id.get(intent_id, set())
            - bottom_atoms_by_id.get(intent_id, set())
        )
        for intent_id in top_order
        if top_atoms_by_id.get(intent_id, set())
        - bottom_atoms_by_id.get(intent_id, set())
    }
    introduced_atoms_by_id = {
        intent_id: sorted(
            bottom_atoms_by_id.get(intent_id, set())
            - evidence_atoms_by_id.get(intent_id, set())
        )
        for intent_id in bottom_order
        if bottom_atoms_by_id.get(intent_id, set())
        - evidence_atoms_by_id.get(intent_id, set())
    }
    atom_owners: dict[str, set[str]] = {}
    for intent_id, atoms in top_atoms_by_id.items():
        for atom in atoms:
            atom_owners.setdefault(atom, set()).add(intent_id)
    unique_atom_owners = {
        atom: next(iter(owners))
        for atom, owners in atom_owners.items()
        if len(owners) == 1
    }
    order_index = {intent_id: index for index, intent_id in enumerate(top_order)}
    premature_atoms_by_id = {
        intent_id: sorted(
            atom
            for atom in bottom_atoms_by_id.get(intent_id, set())
            if atom in unique_atom_owners
            and intent_id in order_index
            and unique_atom_owners[atom] in order_index
            and order_index[intent_id] < order_index[unique_atom_owners[atom]]
        )
        for intent_id in bottom_order
        if any(
            atom in unique_atom_owners
            and intent_id in order_index
            and unique_atom_owners[atom] in order_index
            and order_index[intent_id] < order_index[unique_atom_owners[atom]]
            for atom in bottom_atoms_by_id.get(intent_id, set())
        )
    }
    unbacked_claims = sorted(
        f"{intent_id}:S{number}"
        for intent_id, report in reports_by_id.items()
        for number, claim in enumerate(report.get("claims", []), start=1)
        if claim.get("relation") in {"supported", "qualified_inference"}
        and not claim.get("evidence_ids")
    )
    qualifier_gaps = sorted(
        f"{intent['intent_id']}:{qualifier}"
        for intent in eligible
        for qualifier in intent.get("required_qualifiers", [])
        if not QUALIFIER_EQUIVALENTS.get(
            qualifier,
            re.compile(rf"\b{re.escape(qualifier)}\b", re.IGNORECASE),
        ).search(paragraphs.get(intent["intent_id"], ""))
    )
    transition_state = transition_audit(eligible, paragraphs)

    invariants = {
        "eligible_intents_covered": top_ids == bottom_ids,
        "skeleton_order_conserved": top_order == bottom_order,
        "role_order_conserved": tuple(
            top_roles.get(intent_id, "") for intent_id in top_order
        )
        == tuple(bottom_roles.get(intent_id, "") for intent_id in bottom_order),
        "dependency_edges_conserved": top_edges == bottom_edges,
        "sentence_verification_bijection": expected_sentence_keys
        == reported_sentence_keys,
        "evidence_scope_conserved": all(
            bottom_refs_by_id.get(intent_id, set())
            <= top_refs_by_id.get(intent_id, set())
            for intent_id in bottom_order
        ),
        "supported_claims_have_evidence": not unbacked_claims,
        "protected_atoms_preserved": not missing_atoms_by_id,
        "protected_atoms_backed": not introduced_atoms_by_id,
        "facts_arrive_after_their_owner": not premature_atoms_by_id,
        "qualifiers_preserved": not qualifier_gaps,
        "transition_contracts_satisfied": transition_state["passed"],
    }
    differences = {
        "intents_missing_from_bottom": sorted(top_ids - bottom_ids),
        "intents_added_at_bottom": sorted(bottom_ids - top_ids),
        "edges_missing_from_bottom": sorted(top_edges - bottom_edges),
        "edges_added_at_bottom": sorted(bottom_edges - top_edges),
        "refs_added_by_verifier": {
            intent_id: sorted(
                bottom_refs_by_id.get(intent_id, set())
                - top_refs_by_id.get(intent_id, set())
            )
            for intent_id in bottom_order
            if bottom_refs_by_id.get(intent_id, set())
            - top_refs_by_id.get(intent_id, set())
        },
        "retrieved_refs_unused": {
            intent_id: sorted(
                top_refs_by_id.get(intent_id, set())
                - bottom_refs_by_id.get(intent_id, set())
            )
            for intent_id in top_order
            if top_refs_by_id.get(intent_id, set())
            - bottom_refs_by_id.get(intent_id, set())
        },
        "sentences_missing_from_verifier": sorted(
            expected_sentence_keys - reported_sentence_keys
        ),
        "sentences_added_by_verifier": sorted(
            reported_sentence_keys - expected_sentence_keys
        ),
        "unbacked_claims": unbacked_claims,
        "missing_atoms_by_intent": missing_atoms_by_id,
        "introduced_atoms_by_intent": introduced_atoms_by_id,
        "premature_atoms_by_intent": premature_atoms_by_id,
        "qualifier_gaps": qualifier_gaps,
        "transition_issues": transition_state["issues"],
    }
    return {
        "top_down": {
            "intent_ids": list(top_order),
            "roles": [top_roles[intent_id] for intent_id in top_order],
            "edges": [list(edge) for edge in sorted(top_edges)],
            "evidence_ids_by_intent": {
                intent_id: sorted(refs) for intent_id, refs in top_refs_by_id.items()
            },
            "atomic_terms_by_intent": {
                intent_id: sorted(atoms) for intent_id, atoms in top_atoms_by_id.items()
            },
        },
        "bottom_up": {
            "intent_ids": list(bottom_order),
            "roles": [bottom_roles[intent_id] for intent_id in bottom_order],
            "edges": [list(edge) for edge in sorted(bottom_edges)],
            "sentence_counts": {
                intent_id: len(sentences)
                for intent_id, sentences in sentence_units_by_id.items()
            },
            "evidence_ids_by_intent": {
                intent_id: sorted(refs) for intent_id, refs in bottom_refs_by_id.items()
            },
            "atomic_terms_by_intent": {
                intent_id: sorted(atoms)
                for intent_id, atoms in bottom_atoms_by_id.items()
            },
        },
        "transition_audit": transition_state,
        "differences": differences,
        "invariants": invariants,
        "issues": [name for name, passed in invariants.items() if not passed],
    }


def generate_paragraph_article(
    *,
    query: str,
    outline: str,
    scaffold: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any] | None,
    session: EmbeddingSession,
    span_selector: Callable[[str, str], str],
    style_profile: str,
    narrative_template: str,
    generate: Callable[[str, int, float], str],
    count_prompt_tokens: Callable[[str], int],
    context_token_limit: int,
    max_tokens: int,
    temperature: float,
    evidence_top_k: int = 3,
    minimum_score: float = 0.20,
    retrieval_strategy: str = "baseline",
    facet_config: FacetRetrievalConfig | None = None,
    group_size: int = 1,
    max_repairs: int = 1,
    max_retrieval_retries: int = 1,
    minimum_total_words: int = 0,
    editorial_template: str = "",
    document_type: str = "investigative",
    article_plan: dict[str, Any] | None = None,
    article_plan_canonical_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    draft_plan_context_enabled: bool = False,
    draft_plan_context_token_cap: int | None = None,
    draft_context_policy: str = "legacy",
    draft_context_max_spans: int = 6,
    draft_context_min_term_overlap: int = 1,
    frozen_anchors: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate and verify evidence-scoped paragraph groups, freezing accepted text."""
    outline_intents = parse_paragraph_intents(outline)
    editorial_points = parse_editorial_points(editorial_template)
    intents = (
        outline_intents
        or (
            intents_from_editorial_points(editorial_points, outline_intents)
            if editorial_points
            else intents_from_scaffold(scaffold)
        )
    )
    ledger = retrieve_intent_evidence(
        query,
        intents,
        documents,
        vectors,
        embedding_metadata,
        session=session,
        span_selector=span_selector,
        top_k=evidence_top_k,
        minimum_score=minimum_score,
        retrieval_strategy=retrieval_strategy,
        facet_config=facet_config,
    )
    if article_plan is not None and article_plan_canonical_by_id is not None:
        article_plan = build_article_plan(
            ledger_content_results(ledger),
            article_plan_canonical_by_id,
            author_assertions=list(article_plan.get("author_assertions", [])),
            plan_id=f"{article_plan.get('plan_id', 'article-plan')}:paragraph",
        )
        plan_errors = validate_article_plan(article_plan)
        if plan_errors:
            return {
                "generated": "",
                "ledger": ledger,
                "paragraphs": {},
                "verification_reports": [],
                "verification_failures": {
                    "article": [
                        "article plan refresh failed: " + "; ".join(plan_errors)
                    ]
                },
                "invariant_audit": audit_generation_invariants(ledger, {}, {}),
                "dependency_evidence_audit": dependency_evidence_audit(ledger),
                "group_count": 0,
                "repair_count": 0,
                "pass_count": 0,
                "prompt_tokens": [],
                "verifier_outputs": [],
                "verifier_protocol_diagnostics": [],
                "generation_calls": [],
                "retrieval_retries": [],
                "retrieval_strategy": retrieval_strategy,
                "facet_config": None,
                "evidence_trace": ledger,
                "evidence_trace_debug": render_debug_trace(ledger),
                "article_plan": article_plan,
            }
        _attach_plan_source_spans(ledger, article_plan)
    groups = group_intents(ledger, group_size=group_size)
    if not groups:
        invariant_audit = audit_generation_invariants(ledger, {}, {})
        return {
            "generated": "",
            "ledger": ledger,
            "paragraphs": {},
            "verification_reports": [],
            "verification_failures": {
                "article": ["no paragraph intent had sufficient evidence"]
            },
            "invariant_audit": invariant_audit,
            "dependency_evidence_audit": dependency_evidence_audit(ledger),
            "group_count": 0,
            "repair_count": 0,
            "pass_count": 0,
            "prompt_tokens": [],
            "verifier_outputs": [],
            "verifier_protocol_diagnostics": [],
            "generation_calls": [],
            "retrieval_retries": [],
            "retrieval_strategy": retrieval_strategy,
            "facet_config": (
                {
                    "top_k": facet_config.top_k,
                    "anchor_pool_k": facet_config.anchor_pool_k,
                    "anchor_k": facet_config.anchor_k,
                    "facet_top_k": facet_config.facet_top_k,
                    "rrf_k": facet_config.rrf_k,
                    "base_weight": facet_config.base_weight,
                    "facet_weight": facet_config.facet_weight,
                    "min_anchor_lexical_score": facet_config.min_anchor_lexical_score,
                }
                if facet_config
                else None
            ),
            "evidence_trace": ledger,
            "evidence_trace_debug": render_debug_trace(ledger),
            "article_plan": article_plan,
        }

    paragraphs: dict[str, str] = {}
    reports_by_id: dict[str, dict[str, Any]] = {}
    final_failures: dict[str, list[str]] = {}
    completed_intents: list[dict[str, Any]] = []
    pass_count = 0
    repair_count = 0
    prompt_tokens: list[int] = []
    verifier_outputs: list[dict[str, Any]] = []
    verifier_protocol_diagnostics: list[dict[str, Any]] = []
    call_results: list[dict[str, object]] = []
    draft_plan_context_records: list[dict[str, Any]] = []
    draft_plan_context_tokens: list[int] = []
    previous_tail = ""
    target_words_per_intent = max(
        80,
        (
            minimum_total_words
            + len([item for item in ledger if item["status"] != "unsupported"])
            - 1
        )
        // max(1, len([item for item in ledger if item["status"] != "unsupported"])),
    )

    from paragraph_stages import (
        draft_group_stage,
        repair_group_stage,
        verify_group_stage,
    )

    retrieval_retries: list[dict[str, Any]] = []
    for group in groups:
        draft_plan_context = ""
        context_record: dict[str, Any] = {
            "intent_ids": [str(intent["intent_id"]) for intent in group],
            "intent_evidence_ids": {
                str(intent["intent_id"]): list(intent.get("evidence_ids", []))
                for intent in group
            },
            "enabled": bool(draft_plan_context_enabled),
            "packets": [],
        }
        if draft_plan_context_enabled:
            if article_plan is None:
                raise ValueError("draft plan context requires a validated article plan")
            packet_context = format_scoped_draft_packets(
                article_plan,
                group,
                document_type=document_type,
                draft_context_policy=draft_context_policy,
                draft_context_max_spans=draft_context_max_spans,
                draft_context_min_term_overlap=draft_context_min_term_overlap,
                frozen_anchors=frozen_anchors,
            )
            packet_records = list(packet_context["packets"])
            draft_plan_context = str(packet_context["text"])
            context_record["packets"] = packet_records
            context_record["unique_span_ids"] = list(
                packet_context["unique_span_ids"]
            )
            context_record["unique_edge_ids"] = list(
                packet_context["unique_edge_ids"]
            )
            context_record["text_sha256"] = sha256_text(draft_plan_context)
        else:
            context_record["text_sha256"] = sha256_text("")
        draft_plan_context_records.append(context_record)
        draft_stage = draft_group_stage(
            group,
            query=query,
            style_profile=style_profile,
            narrative_template=narrative_template,
            previous_tail=previous_tail,
            completed_intents=completed_intents,
            target_words_per_intent=target_words_per_intent,
            document_type=document_type,
            draft_plan_context=draft_plan_context,
            draft_context_policy=draft_context_policy,
            draft_plan_context_token_cap=draft_plan_context_token_cap,
            generate=generate,
            count_prompt_tokens=count_prompt_tokens,
            context_token_limit=context_token_limit,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        group_paragraphs = dict(draft_stage.paragraphs)
        paragraphs.update(group_paragraphs)
        stage_call_count = len(draft_stage.generation_calls)
        failures = {key: list(value) for key, value in draft_stage.failures.items()}
        prompt_tokens.extend(draft_stage.prompt_tokens)
        draft_plan_context_tokens.extend(draft_stage.draft_plan_context_tokens)
        call_results.extend(draft_stage.generation_calls)

        if group_paragraphs:
            verification_stage = verify_group_stage(
                group_paragraphs,
                group,
                generate=generate,
                count_prompt_tokens=count_prompt_tokens,
                context_token_limit=context_token_limit,
                max_tokens=max_tokens,
                article_plan=article_plan,
            )
            prompt_tokens.extend(verification_stage.prompt_tokens)
            call_results.extend(verification_stage.generation_calls)
            stage_call_count += len(verification_stage.generation_calls)
            verifier_outputs.extend(verification_stage.verifier_outputs)
            verifier_protocol_diagnostics.extend(
                verification_stage.verifier_protocol_diagnostics
            )
            reports_by_id.update(
                {
                    str(report.get("intent_id")): report
                    for report in verification_stage.reports
                    if report.get("intent_id")
                }
            )
            for intent_id, reasons in verification_stage.failures.items():
                failures.setdefault(intent_id, []).extend(reasons)

            for intent_id, reasons in transition_failures_for_group(
                completed_intents,
                group,
                paragraphs,
            ).items():
                failures.setdefault(intent_id, []).extend(reasons)

        if failures and max_retrieval_retries:
            ledger_by_id = {intent["intent_id"]: intent for intent in ledger}
            retry_failures = {
                intent_id: reasons
                for intent_id, reasons in failures.items()
                if ledger_by_id.get(intent_id, {}).get("retrieval_retry_count", 0)
                < max_retrieval_retries
            }
            retrieval_retries.extend(
                retry_intent_evidence(
                    query,
                    ledger,
                    retry_failures,
                    reports_by_id,
                    documents,
                    vectors,
                    embedding_metadata,
                    session=session,
                    span_selector=span_selector,
                    top_k=evidence_top_k,
                    minimum_score=minimum_score,
                    retry_round=1,
                    retrieval_strategy=retrieval_strategy,
                    facet_config=facet_config,
                )
            )

        if failures and max_repairs:
            repair_stage = repair_group_stage(
                group_paragraphs,
                failures,
                group,
                generate=generate,
                count_prompt_tokens=count_prompt_tokens,
                context_token_limit=context_token_limit,
                max_tokens=max_tokens,
                temperature=temperature,
                document_type=document_type,
                article_plan=article_plan,
            )
            prompt_tokens.extend(repair_stage.prompt_tokens)
            call_results.extend(repair_stage.generation_calls)
            stage_call_count += len(repair_stage.generation_calls)
            verifier_outputs.extend(repair_stage.verifier_outputs)
            verifier_protocol_diagnostics.extend(
                repair_stage.verifier_protocol_diagnostics
            )
            reports_by_id.update(
                {
                    str(report.get("intent_id")): report
                    for report in repair_stage.reports
                    if report.get("intent_id")
                }
            )
            repaired_ids = [
                intent_id
                for intent_id in failures
                if intent_id in repair_stage.paragraphs
            ]
            if repair_stage.ok and repair_stage.paragraphs:
                group_paragraphs = dict(repair_stage.paragraphs)
                paragraphs.update(group_paragraphs)
                for intent_id in repaired_ids:
                    failures.pop(intent_id, None)
            for intent_id, reasons in repair_stage.failures.items():
                failures.setdefault(intent_id, []).extend(reasons)
            if repair_stage.ok:
                repair_count += 1

        if group_paragraphs:
            for intent_id, reasons in transition_failures_for_group(
                completed_intents,
                group,
                paragraphs,
            ).items():
                failures.setdefault(intent_id, []).extend(reasons)

        pass_count += stage_call_count
        final_failures.update(failures)
        accepted_in_group = [
            intent for intent in group if intent["intent_id"] in paragraphs
        ]
        completed_intents.extend(accepted_in_group)
        if accepted_in_group:
            previous_tail = paragraphs[accepted_in_group[-1]["intent_id"]]
    ordered_text = [
        paragraphs[intent["intent_id"]]
        for intent in ledger
        if intent["intent_id"] in paragraphs
    ]
    invariant_audit = audit_generation_invariants(ledger, paragraphs, reports_by_id)
    dependency_audit = dependency_evidence_audit(ledger)
    if invariant_audit["issues"]:
        final_failures.setdefault("article", []).extend(
            f"generation invariant failed: {issue}"
            for issue in invariant_audit["issues"]
        )
    return {
        "generated": "\n\n".join(ordered_text).strip(),
        "ledger": ledger,
        "paragraphs": paragraphs,
        "verification_reports": [
            reports_by_id[intent["intent_id"]]
            for intent in ledger
            if intent["intent_id"] in reports_by_id
        ],
        "verification_failures": final_failures,
        "invariant_audit": invariant_audit,
        "dependency_evidence_audit": dependency_audit,
        "group_count": len(groups),
        "repair_count": repair_count,
        "pass_count": pass_count,
        "prompt_tokens": prompt_tokens,
        "verifier_outputs": verifier_outputs,
        "verifier_protocol_diagnostics": verifier_protocol_diagnostics,
        "generation_calls": call_results,
        "draft_plan_context_enabled": bool(draft_plan_context_enabled),
        "draft_context_policy": draft_context_policy,
        "draft_context_max_spans": draft_context_max_spans,
        "draft_context_min_term_overlap": draft_context_min_term_overlap,
        "draft_plan_context_records": draft_plan_context_records,
        "draft_plan_context_tokens": draft_plan_context_tokens,
        "retrieval_retries": retrieval_retries,
        "retrieval_strategy": retrieval_strategy,
        "facet_config": (
            {
                "top_k": facet_config.top_k,
                "anchor_pool_k": facet_config.anchor_pool_k,
                "anchor_k": facet_config.anchor_k,
                "facet_top_k": facet_config.facet_top_k,
                "rrf_k": facet_config.rrf_k,
                "base_weight": facet_config.base_weight,
                "facet_weight": facet_config.facet_weight,
                "min_anchor_lexical_score": facet_config.min_anchor_lexical_score,
            }
            if facet_config
            else None
        ),
        "evidence_trace": ledger,
        "evidence_trace_debug": render_debug_trace(ledger),
        "article_plan": article_plan,
    }
