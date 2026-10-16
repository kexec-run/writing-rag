"""Empirical, user-relative support scores for causal edges and chains.

These scores rank candidate narrative links against adjacent reasoning found in
non-holdout user writing. They are risk proxies, not entailment probabilities.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

from writing_rag.ingest.build_markdown_content import narrative_roles


WORD_RE = re.compile(r"[A-Za-z0-9']+")
CONNECTOR_RE = re.compile(
    r"\b(because|therefore|so|which means|as a result|leads to|hence|thus|since)\b",
    re.IGNORECASE,
)
QUALIFIER_RE = re.compile(
    r"\b(maybe|could|likely|appears|suggests|uncertain|not sure|may|might)\b",
    re.IGNORECASE,
)
DISCOURSE_RE = re.compile(
    r"\b(but|however|instead|although|while|yet|first|second|then|after|before|finally|once)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "about", "after", "again", "also", "and", "because", "before", "being",
    "between", "both", "but", "could", "does", "each", "from", "have", "into",
    "just", "more", "most", "only", "other", "over", "same", "should", "some",
    "such", "than", "that", "their", "then", "there", "these", "they", "this",
    "those", "through", "under", "very", "what", "when", "where", "which", "while",
    "with", "would", "your",
}
ROLE_ORDER = {
    role: index
    for index, role in enumerate(
        (
            "observation", "motivation", "initial_hypothesis", "discarded_approach",
            "reframe", "experiment", "measurement", "finding", "mechanism",
            "decision", "conclusion", "evidence",
        )
    )
}


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile_rank(value: float, values: list[float]) -> float:
    if not values:
        return 1.0
    below = sum(candidate < value for candidate in values)
    equal = sum(candidate == value for candidate in values)
    return (below + 0.5 * equal) / len(values)


def sentence_units(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
        if len(WORD_RE.findall(sentence)) >= 4
    ]


def content_terms(text: str) -> set[str]:
    return {
        term.casefold()
        for term in WORD_RE.findall(text)
        if len(term) >= 4 and term.casefold() not in STOPWORDS
    }


def role_progression(left: str, right: str) -> float:
    left_roles = narrative_roles(left, None)
    right_roles = narrative_roles(right, None)
    if left_roles == ["evidence"] or right_roles == ["evidence"]:
        return 0.5
    left_index = min(ROLE_ORDER.get(role, ROLE_ORDER["evidence"]) for role in left_roles)
    right_index = min(ROLE_ORDER.get(role, ROLE_ORDER["evidence"]) for role in right_roles)
    if right_index >= left_index:
        return 1.0
    return max(0.0, 1.0 - (left_index - right_index) / 5.0)


def edge_features(
    left: str,
    right: str,
    *,
    bridge: str = "",
    source_directness: float = 0.85,
    endpoint_support: float = 1.0,
) -> dict[str, float]:
    left_terms = content_terms(left)
    right_terms = content_terms(right)
    denominator = max(1, min(len(left_terms), len(right_terms)))
    continuity = min(1.0, len(left_terms & right_terms) / denominator)
    boundary = f"{left[-120:]} {bridge} {right[:160]}"
    return {
        "source_directness": max(0.0, min(1.0, source_directness)),
        "endpoint_support": max(0.0, min(1.0, endpoint_support)),
        "role_progression": role_progression(left, right),
        "lexical_continuity": continuity,
        "explicit_connector": float(bool(CONNECTOR_RE.search(boundary))),
        "qualifier_present": float(bool(QUALIFIER_RE.search(boundary))),
        "discourse_marker": float(bool(DISCOURSE_RE.search(boundary))),
    }


def edge_score(features: dict[str, float]) -> float:
    return (
        0.30 * features["source_directness"]
        + 0.20 * features["endpoint_support"]
        + 0.18 * features["role_progression"]
        + 0.14 * features["lexical_continuity"]
        + 0.10 * features["explicit_connector"]
        + 0.05 * features["qualifier_present"]
        + 0.03 * features["discourse_marker"]
    )


def aggregate_chain_scores(scores: Iterable[float]) -> float:
    values = [max(0.01, min(1.0, score)) for score in scores]
    if not values:
        return 0.0
    geometric_mean = math.exp(sum(math.log(value) for value in values) / len(values))
    weak_link = min(values)
    return geometric_mean * (0.75 + 0.25 * weak_link)


def build_edge_calibration(documents: list[dict[str, Any]]) -> dict[str, Any]:
    feature_rows: list[dict[str, float]] = []
    chain_scores: list[float] = []
    document_count = 0
    for document in documents:
        if document.get("collection") != "content_memory":
            continue
        text = document.get("text", "")
        units = sentence_units(text)
        if len(units) < 2:
            continue
        rows = [
            edge_features(left, right, source_directness=0.85, endpoint_support=1.0)
            for left, right in zip(units, units[1:])
        ]
        scores = [edge_score(row) for row in rows]
        feature_rows.extend(rows)
        chain_scores.append(aggregate_chain_scores(scores))
        document_count += 1

    feature_keys = tuple(edge_features("one claim here", "another claim here"))
    feature_values = {
        key: [row[key] for row in feature_rows]
        for key in feature_keys
    }
    edge_scores = [edge_score(row) for row in feature_rows]
    return {
        "sample_count": len(edge_scores),
        "document_count": document_count,
        "feature_keys": list(feature_keys),
        "feature_values": feature_values,
        "score_values": edge_scores,
        "chain_score_values": chain_scores,
        "accept_threshold": quantile(edge_scores, 0.10) if edge_scores else 0.55,
        "warning_threshold": quantile(edge_scores, 0.02) if edge_scores else 0.40,
        "chain_accept_threshold": quantile(chain_scores, 0.10) if chain_scores else 0.50,
        "chain_warning_threshold": quantile(chain_scores, 0.02) if chain_scores else 0.35,
        "feature_medians": {
            key: quantile(values, 0.50) for key, values in feature_values.items()
        },
    }


def classify_score(score: float, calibration: dict[str, Any], *, chain: bool = False) -> str:
    accept_key = "chain_accept_threshold" if chain else "accept_threshold"
    warning_key = "chain_warning_threshold" if chain else "warning_threshold"
    if score >= calibration[accept_key]:
        return "accepted"
    if score >= calibration[warning_key]:
        return "qualified"
    return "demoted"


def score_candidate_edge(
    left: str,
    right: str,
    *,
    bridge: str,
    shared_source: bool,
    endpoint_support: float,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    source_directness = 1.0 if shared_source and CONNECTOR_RE.search(bridge) else 0.85 if shared_source else 0.55
    features = edge_features(
        left,
        right,
        bridge=bridge,
        source_directness=source_directness,
        endpoint_support=endpoint_support,
    )
    score = edge_score(features)
    probability = percentile_rank(score, calibration.get("score_values", []))
    band = classify_score(score, calibration)
    reward_multiplier = {
        "accepted": 1.0 + 0.20 * probability,
        "qualified": 0.85 + 0.15 * probability,
        "demoted": 0.55 + 0.20 * probability,
    }[band]
    return {
        "score": round(score, 6),
        "probability": round(probability, 6),
        "band": band,
        "reward_multiplier": round(reward_multiplier, 6),
        "features": {key: round(value, 6) for key, value in features.items()},
    }


def score_candidate_chain(
    edge_scores: list[float],
    calibration: dict[str, Any],
    *,
    fallback_score: float = 0.0,
) -> dict[str, Any]:
    score = aggregate_chain_scores(edge_scores) if edge_scores else fallback_score
    probability = percentile_rank(score, calibration.get("chain_score_values", []))
    band = classify_score(score, calibration, chain=True)
    demoted_edges = sum(
        edge_score_value < calibration["warning_threshold"]
        for edge_score_value in edge_scores
    )
    if demoted_edges >= 2:
        band = "demoted"
    elif demoted_edges == 1 and band == "accepted":
        band = "qualified"
    return {
        "score": round(score, 6),
        "probability": round(probability, 6),
        "band": band,
        "demoted_edge_count": demoted_edges,
        "weakest_edge_score": round(min(edge_scores), 6) if edge_scores else 0.0,
    }
