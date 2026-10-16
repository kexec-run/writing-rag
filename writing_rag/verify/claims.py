"""Claim, transition, and verifier-protocol helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from writing_rag.plan.article_plan import validate_claim_trace

TRANSITION_MARKER_RE = re.compile(
    r"\b(at first|but|however|instead|the question became|the control showed|"
    r"this changed|in other words|the practical choice)\b",
    re.IGNORECASE,
)
CONTRACT_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "not",
    "one",
    "only",
    "our",
    "said",
    "some",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
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
ARTICLE_PLAN_LIMITATION_RE = re.compile(
    r"\b(?:does not converge|doesn't converge|lacks? conclusive|"
    r"not conclusive|unconfirmed|not established)\b",
    re.IGNORECASE,
)
RECOGNIZED_QUALIFIERS = {
    "maybe",
    "may",
    "might",
    "could",
    "likely",
    "appears",
    "suggests",
    "uncertain",
    "ambiguous details",
    "reported_hypothesis",
    "does not establish",
    "inference_marker",
    "about",
    "current sample set",
    "not universal",
    "not numerically close",
    "less",
    "more",
}
CONTROL_LEAK_RE = re.compile(
    r"(?im)^(?:#{1,6}\s+intent-\d+\b|Question:|Candidate claim:|"
    r"Required qualifiers:|Evidence:|Failures:|Current paragraph:)"
)
VERDICT_RELATIONS = (
    "supported",
    "qualified_inference",
    "contradicted",
    "insufficient_evidence",
    "interpretation",
    "non_claim",
)
VERDICT_ALIASES = {
    "personal_observation": "insufficient_evidence",
    "personal_action": "insufficient_evidence",
    "technical_fact": "supported",
    "technical_cause": "qualified_inference",
    "measurement": "supported",
    "evidence": "supported",
    "finding": "supported",
    "experiment": "supported",
    "observation": "interpretation",
    "motivation": "interpretation",
    "initial_hypothesis": "qualified_inference",
    "discarded_approach": "interpretation",
    "reframe": "interpretation",
    "mechanism": "qualified_inference",
    "decision": "interpretation",
    "conclusion": "interpretation",
    "question": "non_claim",
    "heading": "non_claim",
    "rhetorical_question": "non_claim",
}
VERDICT_INPUT_RELATIONS = (*VERDICT_RELATIONS, *VERDICT_ALIASES)
VERDICT_RELATION_PATTERN = "|".join(VERDICT_INPUT_RELATIONS)
VERDICT_RE = re.compile(
    rf"^VERDICT\s*\|\s*([^|]+)\s*\|\s*S(\d+)\s*\|\s*"
    rf"({VERDICT_RELATION_PATTERN})\s*\|\s*([^|]*)\s*\|\s*(.*)$",
    re.IGNORECASE,
)
RELATION_FIRST_VERDICT_RE = re.compile(
    rf"^({VERDICT_RELATION_PATTERN})\s*\|\s*"
    r"([^|]+)\s*\|\s*(?:S)?(\d+)\s*\|\s*([^|]*)\s*\|\s*(.*)$",
    re.IGNORECASE,
)
RELATION_PREFIX_VERDICT_RE = re.compile(
    rf"^relation\s*\|\s*([^|]+)\s*\|\s*(?:S)?(\d+)\s*\|\s*"
    rf"({VERDICT_RELATION_PATTERN})\s*\|\s*(.*)$",
    re.IGNORECASE,
)
COMPACT_VERDICT_RE = re.compile(
    rf"^({VERDICT_RELATION_PATTERN})\s*\|\s*none\s*\|\s*"
    r"(?:S)?(\d+)\s*\|\s*([^|]*)\s*\|\s*(.*)$",
    re.IGNORECASE,
)
INDEXED_COMPACT_VERDICT_RE = re.compile(
    rf"^(\d+)\s*\|\s*({VERDICT_RELATION_PATTERN})\s*\|\s*"
    r"(?:S)?(\d+)\s*\|\s*([^|]*)\s*\|\s*(.*)$",
    re.IGNORECASE,
)
FOUR_FIELD_COMPACT_VERDICT_RE = re.compile(
    rf"^({VERDICT_RELATION_PATTERN})\s*\|\s*none\s*\|\s*"
    r"(?:S)?(\d+)\s*\|\s*(.*)$",
    re.IGNORECASE,
)

def claim_units(paragraph: str) -> list[str]:
    """Return stable sentence-sized units so verifier output need not repeat prose."""
    prose = "\n".join(
        line for line in paragraph.splitlines() if not re.match(r"^\s*#{1,6}\s+", line)
    )
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", prose)
        if len(WORD_RE.findall(sentence)) >= 3
    ]


def infer_qualifier(text: str) -> str | None:
    for qualifier, pattern in QUALIFIER_EQUIVALENTS.items():
        if pattern.search(text):
            return qualifier
    return None


def build_claim_verification_prompt(
    paragraphs: dict[str, str],
    intents: list[dict[str, Any]],
    article_plan: dict[str, Any] | None = None,
) -> str:
    paragraph_blocks = []
    by_id = {intent["intent_id"]: intent for intent in intents}
    for intent_id, paragraph in paragraphs.items():
        intent = by_id[intent_id]
        units = "\n".join(
            f"[{intent_id}:S{index}] {sentence}"
            for index, sentence in enumerate(claim_units(paragraph), start=1)
        )
        evidence = "\n".join(
            f"[{result['document']['document_id']}] {result['source_span']}"
            for result in intent.get("evidence", [])
        )
        allowed_ids = ", ".join(
            str(evidence_id) for evidence_id in intent.get("evidence_ids", [])
        ) or "none"
        paragraph_blocks.append(
            f'<intent id="{intent_id}">\n'
            f"<claim_types>{', '.join(intent.get('claim_types', []))}</claim_types>\n"
            f"<allowed_evidence_ids>{allowed_ids}</allowed_evidence_ids>\n"
            f"<sentences>\n{units}\n</sentences>\n"
            f"<evidence>\n{evidence}\n</evidence>\n</intent>"
        )
    plan_instruction = ""
    if article_plan:
        plan_instruction = (
            "This run also has an article authority plan. Keep source-backed claims "
            "inside the listed source allowlists. For a source-backed claim, use the "
            "existing evidence field for source document IDs. For a causal claim, "
            "append the explicit narrative edge IDs and use no inferred context edge. "
            "For an author_asserted claim, append the supplied author assertion IDs; "
            "never invent an assertion ID. Extended verdict lines use: "
            "VERDICT | intent-id | S<number> | relation | evidence IDs or none | "
            "qualifier or none | authority | narrative edge IDs or none | "
            "author assertion IDs or none. Use authority none only for "
            "interpretation, insufficient_evidence, contradicted, or non_claim "
            "sentences that carry no provenance identifiers.\n\n"
        )
    verdict_format = (
        "VERDICT | intent-id | S<number> | relation | evidence IDs or none | "
        "qualifier or none | authority | narrative edge IDs or none | "
        "author assertion IDs or none\n"
        if article_plan
        else "relation | intent-id | S<number> | required qualifier or none | "
        "comma-separated evidence IDs or none\n"
    )
    return (
        "Audit each paragraph claim against only its attached evidence. Retrieval relevance is not support. "
        "Each intent has an exclusive evidence allowlist; cite only IDs inside that intent's "
        "<allowed_evidence_ids> field. An ID from another intent is a scope violation, even when its "
        "text seems relevant. Use evidence none for non_claim sentences. "
        "Use the declared claim types to judge the proposition: personal observations and actions need "
        "self-authored source evidence, technical facts need a supporting span, technical causes need "
        "evidence for the relation, and interpretations must remain interpretations unless directly supported. "
        "Split compound sentences into separate claims. Classify every factual or causal claim as exactly one "
        "of: supported, qualified_inference, contradicted, insufficient_evidence, interpretation, non_claim. "
        "Use non_claim for a rhetorical question or a heading that does not assert a proposition. "
        "Use interpretation when the sentence is explicitly framed as an interpretation rather than a fact. "
        "A causal relation requires evidence for the relation, not merely evidence for both endpoints. "
        "Judge the complete proposition and its framing: reporting 'we suspected X, but the test refuted it' "
        "does not assert X as true and is supported when the evidence records that investigative sequence. "
        "Do not judge voice or writing quality.\n\n"
        + plan_instruction
        + "Return exactly one compact line for every numbered sentence, and nothing else:\n"
        + verdict_format
        + "Do not repeat sentence text. Do not use JSON or Markdown. A qualified inference must name the "
        "qualifier that appears in the sentence. The qualifier field must contain only that hedge "
        "word or phrase, or none; never put a relation or claim type such as supported or "
        "qualified_inference in the qualifier field. The fields after the relation are always, "
        "in order: evidence IDs, qualifier, authority, narrative edge IDs, author assertion IDs. "
        "For example, a source-backed claim is: VERDICT | intent-001 | S1 | supported | "
        "doc-synthetic-001 | none | source_backed | none | none. An interpretation with no "
        "provenance is: VERDICT | intent-001 | S1 | interpretation | none | none | none | none | none. "
        "Do not put prose in the evidence or authority fields. Use source_backed whenever evidence "
        "IDs are present, author_asserted whenever author assertion IDs are present, and none only "
        "for a non-source-backed relation.\n\n"
        + "\n\n".join(paragraph_blocks)
    )


def parse_claim_verification(
    output: str,
    expected_ids: list[str],
    paragraphs: dict[str, str] | None = None,
    allowed_evidence_ids: dict[str, set[str]] | None = None,
    article_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    paragraphs = paragraphs or {}
    sentence_map = {
        intent_id: claim_units(paragraphs.get(intent_id, ""))
        for intent_id in expected_ids
    }
    line_claims: dict[str, dict[int, dict[str, Any]]] = {}
    compact_intent_index = 0
    compact_last_sentence_number = 0

    def compact_intent_id(sentence_number: int) -> str | None:
        nonlocal compact_intent_index, compact_last_sentence_number
        if len(expected_ids) == 1:
            return expected_ids[0]
        if compact_last_sentence_number and (
            sentence_number <= compact_last_sentence_number
        ):
            compact_intent_index += 1
            compact_last_sentence_number = 0
        if compact_intent_index >= len(expected_ids):
            return None
        compact_last_sentence_number = sentence_number
        return expected_ids[compact_intent_index]

    def normalized_evidence_ids(value: str) -> list[str]:
        evidence_ids = []
        for token in re.split(r"[,\s]+", value.strip()):
            token = token.strip()
            if not token or token.casefold() == "none":
                continue
            if re.fullmatch(r"\d+", token):
                token = f"markdown_{token}"
            evidence_ids.append(token)
        return list(dict.fromkeys(evidence_ids))

    def normalized_trace_ids(value: str) -> list[str]:
        return list(
            dict.fromkeys(
                token.strip()
                for token in re.split(r"[,\s]+", value.strip())
                if token.strip() and token.strip().casefold() != "none"
            )
        )

    def looks_like_evidence_field(value: str) -> bool:
        if value.strip().casefold() == "none":
            return True
        tokens = [token for token in re.split(r"[,\s]+", value.strip()) if token]
        if not tokens:
            return True
        return all(
            re.fullmatch(
                r"(?:markdown|window|mdx|d)[A-Za-z0-9_-]*",
                token,
                re.IGNORECASE,
            )
            for token in tokens
        )

    def constrain_claim(
        claim: dict[str, Any], intent_id: str
    ) -> dict[str, Any]:
        evidence_ids = list(claim.get("evidence_ids") or [])
        scope_violation_ids: list[str] = []
        allowed = (
            allowed_evidence_ids.get(intent_id)
            if allowed_evidence_ids is not None
            else None
        )
        if allowed is not None:
            scope_violation_ids = sorted(set(evidence_ids) - set(allowed))
            evidence_ids = [
                evidence_id for evidence_id in evidence_ids if evidence_id in allowed
            ]
        if claim.get("relation") == "non_claim" and evidence_ids:
            scope_violation_ids = sorted(
                set(scope_violation_ids) | set(evidence_ids)
            )
            evidence_ids = []
        claim["evidence_ids"] = evidence_ids
        claim["scope_violation_ids"] = scope_violation_ids
        return claim

    def extended_metadata(line: str) -> tuple[str, str, str] | None:
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 9 or fields[0].casefold() != "verdict":
            return None
        return fields[6], fields[7], fields[8]

    for line in output.splitlines():
        stripped = line.strip()
        qualifier_field_shifted = False
        metadata = extended_metadata(stripped)
        authority = "source_backed"
        edge_field = "none"
        assertion_field = "none"
        if metadata is not None:
            fields = [field.strip() for field in stripped.split("|")]
            authority, edge_field, assertion_field = metadata
            if (
                len(fields) >= 9
                and fields[4].casefold() != "none"
                and not looks_like_evidence_field(fields[4])
                and looks_like_evidence_field(fields[5])
            ):
                # A common model formatting error places the qualifier phrase
                # before the evidence value.  Recover the intended fields only
                # when the evidence value has a known document-ID shape.
                tail = fields[6:]
                authority_candidates = [
                    value
                    for value in tail
                    if value.casefold() in {"source_backed", "author_asserted"}
                ]
                authority = authority_candidates[0] if authority_candidates else (
                    tail[0] if tail and tail[0].casefold() == "none" else "none"
                )
                edge_candidates = [
                    value
                    for value in tail
                    if re.match(r"^(?:causal|temporal|edge)[:_-]", value, re.IGNORECASE)
                ]
                assertion_candidates = [
                    value
                    for value in tail
                    if re.match(r"^author[-_:]", value, re.IGNORECASE)
                ]
                edge_field = edge_candidates[0] if edge_candidates else "none"
                assertion_field = (
                    assertion_candidates[0] if assertion_candidates else "none"
                )
                fields = [
                    *fields[:4],
                    fields[5],
                    fields[4],
                    authority,
                    edge_field,
                    assertion_field,
                ]
                qualifier_field_shifted = True
            stripped = " | ".join(fields[:6])
        match = VERDICT_RE.match(stripped)
        if match:
            intent_id, sentence_number, relation, evidence_field, qualifier = (
                match.groups()
            )
        else:
            relation_prefix = RELATION_PREFIX_VERDICT_RE.match(stripped)
            if relation_prefix:
                intent_id, sentence_number, relation, evidence_field = (
                    relation_prefix.groups()
                )
                qualifier = "none"
            else:
                compact = COMPACT_VERDICT_RE.match(stripped)
                if compact:
                    relation, sentence_number, qualifier, evidence_field = compact.groups()
                    intent_id = compact_intent_id(int(sentence_number))
                    if intent_id is None:
                        continue
                else:
                    indexed_compact = INDEXED_COMPACT_VERDICT_RE.match(stripped)
                    four_field_compact = FOUR_FIELD_COMPACT_VERDICT_RE.match(stripped)
                    fields = [field.strip() for field in stripped.split("|")]
                    explicit_short = (
                        len(fields) == 4
                        and fields[0].casefold() in VERDICT_INPUT_RELATIONS
                        and re.fullmatch(r"intent-\d+", fields[1], re.IGNORECASE)
                    )
                    intentless_four = (
                        len(fields) == 4
                        and fields[0].casefold() in VERDICT_INPUT_RELATIONS
                        and not explicit_short
                        and not fields[1].casefold().startswith("intent-")
                        and re.fullmatch(r"(?:S)?\d+", fields[2], re.IGNORECASE)
                    )
                    if indexed_compact:
                        _, relation, sentence_number, qualifier, evidence_field = (
                            indexed_compact.groups()
                        )
                        intent_id = compact_intent_id(int(sentence_number))
                        if intent_id is None:
                            continue
                    elif four_field_compact or intentless_four:
                        if four_field_compact:
                            relation, sentence_number, evidence_field = (
                                four_field_compact.groups()
                            )
                            qualifier = "none"
                        else:
                            relation, qualifier, sentence_number, evidence_field = fields
                        intent_id = compact_intent_id(
                            int(re.sub(r"^[sS]", "", sentence_number))
                        )
                        if intent_id is None:
                            continue
                    elif explicit_short:
                        relation, intent_id, sentence_number, evidence_field = fields
                        qualifier = "none"
                    else:
                        relation_first = RELATION_FIRST_VERDICT_RE.match(stripped)
                        if relation_first:
                            (
                                relation,
                                intent_id,
                                sentence_number,
                                qualifier,
                                evidence_field,
                            ) = relation_first.groups()
                        else:
                            if (
                                not fields
                                or fields[0].casefold() not in VERDICT_INPUT_RELATIONS
                            ):
                                continue
                            located = next(
                                (
                                    re.search(
                                        r"\b(intent-\d+):S(\d+)\b", field, re.IGNORECASE
                                    )
                                    for field in fields[1:]
                                    if re.search(
                                        r"\bintent-\d+:S\d+\b", field, re.IGNORECASE
                                    )
                                ),
                                None,
                            )
                            if located is None:
                                continue
                            relation = fields[0]
                            intent_id, sentence_number = located.groups()
                            candidates = [
                                field
                                for field in fields[1:]
                                if field.casefold() != "none"
                                and not re.search(
                                    r"\bintent-\d+:S\d+\b", field, re.IGNORECASE
                                )
                                and not re.fullmatch(
                                    r"(?:markdown_\d+|window_\d+|mdx_[\w-]+_\d+_content)"
                                    r"(?:\s*,\s*(?:markdown_\d+|window_\d+|mdx_[\w-]+_\d+_content))*",
                                    field,
                                    re.IGNORECASE,
                                )
                            ]
                            qualifier = candidates[0] if candidates else "none"
                            evidence_field = (
                                ",".join(
                                    re.findall(
                                        r"(?:markdown_\d+|window_\d+|mdx_[\w-]+_\d+_content)",
                                        stripped,
                                        re.IGNORECASE,
                                    )
                                )
                                or "none"
                            )
        intent_id = intent_id.strip()
        if intent_id.casefold() in {"intent-id", "intent_id"}:
            if len(expected_ids) != 1:
                continue
            intent_id = expected_ids[0]
        number = int(re.sub(r"^[sS]", "", str(sentence_number).strip()))
        if intent_id not in expected_ids or number < 1:
            continue
        raw_relation = relation.strip().casefold()
        relation = VERDICT_ALIASES.get(raw_relation, raw_relation)
        sentences = sentence_map.get(intent_id, [])
        text = sentences[number - 1] if number <= len(sentences) else ""
        evidence_ids = normalized_evidence_ids(evidence_field)
        narrative_edge_ids = normalized_trace_ids(edge_field)
        author_assertion_ids = normalized_trace_ids(assertion_field)
        required_qualifier = (
            None
            if qualifier.strip().casefold() == "none"
            else qualifier.strip().casefold()
        )
        qualifier_field_is_relation = (
            required_qualifier is not None
            and required_qualifier in VERDICT_INPUT_RELATIONS
        )
        if qualifier_field_is_relation:
            # Some verifier generations copy the relation into the qualifier
            # field (for example, ``qualified_inference``).  That is a
            # protocol error, not a meaningful hedge.  Preserve the error in
            # the claim trace, but do not let the relation name become a
            # fabricated required qualifier.
            required_qualifier = None
        if relation.casefold() == "qualified_inference":
            required_qualifier = infer_qualifier(text) or required_qualifier
        claim = constrain_claim(
            {
                "text": text,
                "relation": relation.casefold(),
                "raw_relation": raw_relation,
                "protocol_status": (
                    "malformed_claim_type_in_relation_slot"
                    if raw_relation in VERDICT_ALIASES
                    else "malformed_extended_field_order"
                    if qualifier_field_shifted
                    else "malformed_qualifier_field"
                    if qualifier_field_is_relation
                    else "valid"
                ),
                "evidence_ids": evidence_ids,
                "authority": authority.casefold() or "source_backed",
                "narrative_edge_ids": narrative_edge_ids,
                "author_assertion_ids": author_assertion_ids,
                "required_qualifier": required_qualifier,
            },
            intent_id,
        )
        line_claims.setdefault(intent_id, {})[number] = claim
    if line_claims:
        reports = []
        for intent_id in expected_ids:
            expected = sentence_map.get(intent_id, [])
            claims = []
            for number, sentence in enumerate(expected, start=1):
                claims.append(
                    line_claims.get(intent_id, {}).get(
                        number,
                        {
                            "text": sentence,
                            "relation": "insufficient_evidence",
                            "evidence_ids": [],
                            "authority": "source_backed",
                            "narrative_edge_ids": [],
                            "author_assertion_ids": [],
                            "required_qualifier": None,
                        },
                    )
                )
            decision = (
                "repair"
                if any(
                    claim["relation"] in {"contradicted", "insufficient_evidence"}
                    for claim in claims
                )
                else "accept"
            )
            reports.append(
                {"intent_id": intent_id, "decision": decision, "claims": claims}
            )
        return reports

    # Backward-compatible JSON parsing for stored runs and older test fixtures.
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.IGNORECASE
    )
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            if isinstance(payload, list):
                reports = [
                    item for item in payload if isinstance(item, dict)
                ]
                for report in reports:
                    intent_id = str(report.get("intent_id", ""))
                    for claim in report.get("claims", []):
                        if isinstance(claim, dict):
                            constrain_claim(claim, intent_id)
                return reports
        except json.JSONDecodeError:
            pass
    return [
        {
            "intent_id": intent_id,
            "decision": "repair",
            "claims": [
                {
                    "text": "Verifier output did not contain valid verdict lines or JSON.",
                    "relation": "insufficient_evidence",
                    "evidence_ids": [],
                    "scope_violation_ids": [],
                    "authority": "source_backed",
                    "narrative_edge_ids": [],
                    "author_assertion_ids": [],
                    "required_qualifier": None,
                }
            ],
        }
        for intent_id in expected_ids
    ]


def verifier_protocol_diagnostics(
    output: str,
    expected_ids: list[str],
    paragraphs: dict[str, str] | None = None,
    reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Explain verifier protocol quality without deciding evidence quality."""

    paragraphs = paragraphs or {}
    expected_claim_count = sum(
        len(claim_units(paragraphs.get(intent_id, ""))) for intent_id in expected_ids
    )
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.IGNORECASE
    )
    json_payload = None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            json_payload = json.loads(cleaned)
        except json.JSONDecodeError:
            json_payload = None
    if isinstance(json_payload, list):
        parsed_claim_count = sum(
            len(report.get("claims", []))
            for report in json_payload
            if isinstance(report, dict)
        )
        status = "valid" if parsed_claim_count else "malformed"
        errors = [] if parsed_claim_count else ["JSON verifier response contained no claims"]
        return {
            "status": status,
            "format": "json",
            "recognized_line_count": 0,
            "expected_claim_count": expected_claim_count,
            "parsed_claim_count": parsed_claim_count,
            "unrecognized_lines": [],
            "alias_lines": [],
            "missing_claim_count": max(expected_claim_count - parsed_claim_count, 0),
            "errors": errors,
            "intent_errors": {
                intent_id: list(errors) for intent_id in expected_ids if errors
            },
        }

    recognized_line_count = 0
    alias_lines: list[dict[str, Any]] = []
    unrecognized_lines: list[dict[str, Any]] = []
    explicit_claim_keys: set[tuple[str, int]] = set()

    def line_identity(line: str) -> tuple[str | None, int | None, str | None]:
        match = VERDICT_RE.match(line)
        if match:
            return match.group(1).strip(), int(match.group(2)), match.group(3)
        match = RELATION_PREFIX_VERDICT_RE.match(line)
        if match:
            intent_id = match.group(1).strip()
            if intent_id.casefold() in {"intent-id", "intent_id"}:
                intent_id = expected_ids[0] if len(expected_ids) == 1 else None
            return intent_id, int(match.group(2)), match.group(3)
        match = RELATION_FIRST_VERDICT_RE.match(line)
        if match:
            return match.group(2).strip(), int(match.group(3)), match.group(1)
        match = INDEXED_COMPACT_VERDICT_RE.match(line)
        if match:
            return None, int(match.group(3)), match.group(2)
        match = COMPACT_VERDICT_RE.match(line)
        if match:
            return None, int(match.group(2)), match.group(1)
        match = FOUR_FIELD_COMPACT_VERDICT_RE.match(line)
        if match:
            return None, int(match.group(2)), match.group(1)
        fields = [field.strip() for field in line.split("|")]
        if len(fields) == 4 and fields[0].casefold() in VERDICT_INPUT_RELATIONS:
            if re.fullmatch(r"intent-\d+", fields[1], re.IGNORECASE):
                return fields[1], int(re.sub(r"^[sS]", "", fields[2])), fields[0]
            if re.fullmatch(r"(?:S)?\d+", fields[2], re.IGNORECASE):
                return None, int(re.sub(r"^[sS]", "", fields[2])), fields[0]
        located = re.search(r"\b(intent-\d+):S(\d+)\b", line, re.IGNORECASE)
        if located and fields and fields[0].casefold() in VERDICT_INPUT_RELATIONS:
            return located.group(1), int(located.group(2)), fields[0]
        return None, None, None

    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        intent_id, sentence_number, raw_relation = line_identity(line)
        if raw_relation is None:
            unrecognized_lines.append({"line": line_number, "text": line})
            continue
        recognized_line_count += 1
        if intent_id is not None and sentence_number is not None:
            explicit_claim_keys.add((intent_id, sentence_number))
        relation_key = raw_relation.casefold()
        if relation_key in VERDICT_ALIASES:
            alias_lines.append(
                {
                    "line": line_number,
                    "text": line,
                    "intent_id": intent_id,
                    "sentence": sentence_number,
                    "raw_relation": relation_key,
                    "normalized_relation": VERDICT_ALIASES[relation_key],
                }
            )

    parsed_claim_count = sum(
        len(
            [
                claim
                for claim in report.get("claims", [])
                if isinstance(claim, dict) and "raw_relation" in claim
            ]
        )
        for report in (reports or [])
        if isinstance(report, dict)
    )
    malformed_qualifier_claims = [
        {
            "intent_id": str(report.get("intent_id")),
            "text": str(claim.get("text", "")),
        }
        for report in (reports or [])
        if isinstance(report, dict)
        for claim in report.get("claims", [])
        if isinstance(claim, dict)
        and claim.get("protocol_status") == "malformed_qualifier_field"
    ]
    malformed_extended_field_claims = [
        {
            "intent_id": str(report.get("intent_id")),
            "text": str(claim.get("text", "")),
        }
        for report in (reports or [])
        if isinstance(report, dict)
        for claim in report.get("claims", [])
        if isinstance(claim, dict)
        and claim.get("protocol_status") == "malformed_extended_field_order"
    ]
    missing_claim_count = max(expected_claim_count - parsed_claim_count, 0)
    errors: list[str] = []
    if not recognized_line_count:
        errors.append("no recognized verifier verdict lines")
    if alias_lines:
        errors.append("claim type or narrative role used in the verdict relation slot")
    if malformed_qualifier_claims:
        errors.append("relation used in the verifier qualifier field")
    if malformed_extended_field_claims:
        errors.append("qualifier phrase was shifted into the verifier evidence field")
    if unrecognized_lines:
        errors.append(f"{len(unrecognized_lines)} unrecognized verifier line(s)")
    if missing_claim_count:
        errors.append(f"{missing_claim_count} expected sentence verdict(s) were not parsed")
    if not recognized_line_count:
        status = "malformed"
    elif (
        alias_lines
        or unrecognized_lines
        or malformed_qualifier_claims
        or malformed_extended_field_claims
    ):
        status = "protocol_malformed"
    elif missing_claim_count:
        status = "incomplete"
    else:
        status = "valid"

    intent_errors: dict[str, list[str]] = {intent_id: [] for intent_id in expected_ids}
    for alias in alias_lines:
        targets = (
            [alias["intent_id"]]
            if alias.get("intent_id") in intent_errors
            else expected_ids
        )
        for intent_id in targets:
            intent_errors[intent_id].append(
                "claim type/role in verdict slot: " + str(alias["raw_relation"])
            )
    for malformed in malformed_qualifier_claims:
        intent_id = malformed["intent_id"]
        targets = [intent_id] if intent_id in intent_errors else expected_ids
        for target in targets:
            intent_errors[target].append(
                "relation used in verifier qualifier field"
            )
    for malformed in malformed_extended_field_claims:
        intent_id = malformed["intent_id"]
        targets = [intent_id] if intent_id in intent_errors else expected_ids
        for target in targets:
            intent_errors[target].append(
                "qualifier phrase was shifted into verifier evidence field"
            )
    if unrecognized_lines or not recognized_line_count:
        for intent_id in expected_ids:
            intent_errors[intent_id].append("unrecognized or missing verifier protocol line")
    if missing_claim_count:
        for intent_id in expected_ids:
            intent_errors[intent_id].append("one or more sentence verdicts were not parsed")
    intent_errors = {
        intent_id: list(dict.fromkeys(messages))
        for intent_id, messages in intent_errors.items()
        if messages
    }
    return {
        "status": status,
        "format": "lines",
        "recognized_line_count": recognized_line_count,
        "expected_claim_count": expected_claim_count,
        "parsed_claim_count": parsed_claim_count,
        "explicit_claim_keys": sorted(
            [list(key) for key in explicit_claim_keys]
        ),
        "unrecognized_lines": unrecognized_lines,
        "alias_lines": alias_lines,
        "malformed_qualifier_claims": malformed_qualifier_claims,
        "malformed_qualifier_count": len(malformed_qualifier_claims),
        "malformed_extended_field_claims": malformed_extended_field_claims,
        "malformed_extended_field_count": len(malformed_extended_field_claims),
        "missing_claim_count": missing_claim_count,
        "errors": errors,
        "intent_errors": intent_errors,
    }


def verification_failures(
    reports: list[dict[str, Any]],
    expected_ids: list[str],
    *,
    protocol_diagnostics: dict[str, Any] | None = None,
    article_plan: dict[str, Any] | None = None,
    intent_claim_types: dict[str, set[str]] | None = None,
    intent_article_plans: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    by_id = {str(report.get("intent_id")): report for report in reports}
    failures: dict[str, list[str]] = {}
    for intent_id in expected_ids:
        report = by_id.get(intent_id)
        if report is None:
            failures[intent_id] = ["claim verifier omitted this paragraph"]
            continue
        reasons = []
        protocol_errors = (protocol_diagnostics or {}).get("intent_errors", {}).get(
            intent_id, []
        )
        reasons.extend(f"verifier_protocol_error: {error}" for error in protocol_errors)
        skip_claim_checks = (
            protocol_diagnostics is not None
            and protocol_diagnostics.get("recognized_line_count", 0) == 0
        )
        for claim in report.get("claims", []):
            if claim.get("protocol_status") == "malformed_claim_type_in_relation_slot":
                if protocol_diagnostics is not None:
                    reasons.append(
                        "verifier_protocol_error: relation slot contained "
                        f"{claim.get('raw_relation')}"
                    )
                    continue
            if claim.get("protocol_status") == "malformed_qualifier_field":
                reasons.append(
                    "verifier_protocol_error: qualifier field contained relation "
                    f"{claim.get('raw_relation')}"
                )
            if claim.get("protocol_status") == "malformed_extended_field_order":
                reasons.append(
                    "verifier_protocol_error: qualifier phrase was shifted into "
                    "the evidence field"
                )
            if skip_claim_checks:
                continue
            relation = claim.get("relation")
            scope_violation_ids = claim.get("scope_violation_ids") or []
            if scope_violation_ids:
                reasons.append(
                    "evidence_scope_violation: "
                    + ", ".join(str(item) for item in scope_violation_ids)
                )
            if relation in {"contradicted", "insufficient_evidence"}:
                reasons.append(f"{relation}: {claim.get('text', '')}")
            if (
                relation in {"supported", "qualified_inference"}
                and not claim.get("evidence_ids")
            ):
                reasons.append(
                    f"{relation} lacks evidence: {claim.get('text', '')}"
                )
            required = str(claim.get("required_qualifier") or "").casefold()
            if (
                article_plan
                and relation == "qualified_inference"
                and not required
                and ARTICLE_PLAN_LIMITATION_RE.search(str(claim.get("text", "")))
            ):
                required = "uncertain"
            if relation == "qualified_inference" and not required:
                reasons.append(
                    f"qualified inference missing qualifier assessment: {claim.get('text', '')}"
                )
            literal_qualifier = bool(
                article_plan
                and required
                and re.search(
                    rf"\b{re.escape(required)}\b",
                    str(claim.get("text", "")),
                    re.IGNORECASE,
                )
            )
            if (
                relation == "qualified_inference"
                and required not in RECOGNIZED_QUALIFIERS
                and not literal_qualifier
            ):
                reasons.append(
                    f"qualified inference lacks a recognized hedge ({required}): "
                    f"{claim.get('text', '')}"
                )
            if (
                relation == "qualified_inference"
                and (required in RECOGNIZED_QUALIFIERS or literal_qualifier)
            ):
                pattern = QUALIFIER_EQUIVALENTS.get(
                    required, re.compile(rf"\b{re.escape(required)}\b", re.IGNORECASE)
                )
                if not pattern.search(str(claim.get("text", ""))) and not (
                    required == "uncertain"
                    and ARTICLE_PLAN_LIMITATION_RE.search(str(claim.get("text", "")))
                ):
                    reasons.append(
                        f"missing qualifier {required}: {claim.get('text', '')}"
                    )
            if article_plan:
                traced_claim = dict(claim)
                claim_types = (intent_claim_types or {}).get(intent_id, set())
                if "technical_cause" in claim_types:
                    traced_claim["causal"] = True
                claim_plan = (intent_article_plans or {}).get(intent_id, article_plan)
                claim_slot = intent_id if intent_article_plans else "article"
                plan_errors = validate_claim_trace(
                    traced_claim,
                    claim_plan,
                    slot_id=claim_slot,
                )
                reasons.extend(f"article_plan: {error}" for error in plan_errors)
        if report.get("decision") == "repair" and not reasons:
            reasons.append("claim verifier requested repair")
        if reasons:
            failures[intent_id] = list(dict.fromkeys(reasons))
    return failures


def build_local_repair_prompt(
    paragraphs: dict[str, str],
    failures: dict[str, list[str]],
    intents: list[dict[str, Any]],
    document_type: str = "investigative",
) -> str:
    form_instruction = (
        " The reader question is planning metadata, not a required rhetorical question or heading. "
        "Prefer a definition, direct claim, or consequence as the opening form."
        if document_type.casefold() == "mdx"
        else ""
    )
    by_id = {intent["intent_id"]: intent for intent in intents}
    blocks = []
    for intent_id, reasons in failures.items():
        intent = by_id[intent_id]
        evidence = "\n".join(
            f'<span id="{result["document"]["document_id"]}">{result["source_span"]}</span>'
            for result in intent.get("evidence", [])
        )
        blocks.append(
            f'<repair intent="{intent_id}">\n<problems>{" ; ".join(reasons)}</problems>\n'
            f"<draft>{paragraphs.get(intent_id, '')}</draft>\n"
            f"<evidence>{evidence}</evidence>\n</repair>"
        )
    return (
        "Repair only the listed paragraphs. Delete unsupported claims, preserve supported details and "
        "qualifiers, and qualify an inference only when the evidence permits it. Do not alter accepted "
        "paragraphs and do not add facts. Never copy XML labels, evidence wrappers, failure descriptions, "
        "or internal intent headings into prose." + form_instruction
        + " Return one [[INTENT ...]] marked block per listed intent and "
        "nothing else.\n\n" + "\n\n".join(blocks)
    )


def merge_paragraph_repairs(
    paragraphs: dict[str, str], repairs: dict[str, str]
) -> dict[str, str]:
    merged = dict(paragraphs)
    for intent_id, text in repairs.items():
        if intent_id in merged and text.strip():
            merged[intent_id] = text.strip()
    return merged


def _contract_terms(text: str) -> set[str]:
    return {
        word.casefold()
        for word in WORD_RE.findall(text)
        if len(word) >= 4 and word.casefold() not in CONTRACT_STOPWORDS
    }


def transition_audit(
    intents: list[dict[str, Any]],
    paragraphs: dict[str, str],
) -> dict[str, Any]:
    """Check declared beat dependencies without pretending coherence is similarity."""
    by_id = {intent["intent_id"]: intent for intent in intents}
    positions = {intent["intent_id"]: index for index, intent in enumerate(intents)}
    details = []
    issues: list[str] = []
    for intent in intents:
        intent_id = intent["intent_id"]
        if intent_id not in paragraphs:
            continue
        dependencies = list(intent.get("depends_on", []))
        invalid_dependencies = [
            dependency
            for dependency in dependencies
            if dependency not in by_id
            or positions.get(dependency, len(intents)) >= positions.get(intent_id, 0)
        ]
        if invalid_dependencies:
            issue = f"{intent_id} depends on missing or later beats: " + ", ".join(
                invalid_dependencies
            )
            issues.append(issue)
            details.append(
                {
                    "intent_id": intent_id,
                    "mode": intent.get("transition_mode", ""),
                    "passed": False,
                    "reason": issue,
                    "overlap": 0,
                }
            )
            continue

        dependency = by_id.get(dependencies[-1]) if dependencies else None
        mode = intent.get("transition_mode", "")
        anchor = (dependency.get("closes_question", "") if dependency else "") or (
            dependency.get("question", "") if dependency else ""
        )
        overlap = len(_contract_terms(anchor) & _contract_terms(paragraphs[intent_id]))
        marker = bool(TRANSITION_MARKER_RE.search(paragraphs[intent_id]))
        semantic_check = bool(intent.get("_contract_explicit")) and mode in {
            "answer",
            "redirect",
            "contrast",
        }
        passed = not semantic_check or bool(overlap or marker)
        reason = (
            ""
            if passed
            else (
                f"{intent_id} declares transition={mode} but does not connect to "
                f"the preceding beat's question or use a discourse marker"
            )
        )
        if reason:
            issues.append(reason)
        details.append(
            {
                "intent_id": intent_id,
                "depends_on": dependencies,
                "mode": mode,
                "passed": passed,
                "overlap": overlap,
                "discourse_marker": marker,
                "reason": reason,
            }
        )
    return {
        "passed": not issues,
        "issues": issues,
        "details": details,
        "dependency_edges": [
            [dependency, intent["intent_id"]]
            for intent in intents
            for dependency in intent.get("depends_on", [])
        ],
    }


def transition_failures_for_group(
    completed_intents: list[dict[str, Any]],
    group: list[dict[str, Any]],
    paragraphs: dict[str, str],
) -> dict[str, list[str]]:
    intents = [*completed_intents, *group]
    audit = transition_audit(intents, paragraphs)
    group_ids = {intent["intent_id"] for intent in group}
    return {
        detail["intent_id"]: [detail["reason"]]
        for detail in audit["details"]
        if detail["intent_id"] in group_ids
        and not detail["passed"]
        and detail["reason"]
    }
