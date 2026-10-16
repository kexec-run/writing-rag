"""Retrieve evidence with relevance-gated source-span query facets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from writing_rag.retrieve.retrieve import EmbeddingSession, lexical_overlap, lexical_terms, retrieve_many


@dataclass(frozen=True)
class FacetRetrievalConfig:
    """Controls conservative query-facet expansion and rank fusion."""

    top_k: int = 8
    anchor_pool_k: int = 10
    anchor_k: int = 2
    facet_top_k: int = 20
    rrf_k: int = 60
    base_weight: float = 1.0
    facet_weight: float = 1.0
    min_anchor_lexical_score: float = 0.5
    min_anchor_term_hits: int = 1
    max_facet_words: int = 80


def _facet_sentences(text: str) -> list[str]:
    """Mirror the evidence boundary used by generation provenance."""
    candidates: list[str] = []
    parts = re.split(r"```[^\n]*\n?(.*?)```", text, flags=re.DOTALL)
    for part_index, part in enumerate(parts):
        if part_index % 2:
            code_lines = [line.strip() for line in part.splitlines() if line.strip()]
            if code_lines:
                candidates.append(" ".join(code_lines))
            continue
        blocks = re.split(r"\n\s*\n", part)
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            units = lines if any(line.startswith("|") for line in lines) else [
                " ".join(lines)
            ]
            for unit in units:
                cleaned = re.sub(r"^(?:>|[-*]|\d+[.)])\s*", "", unit).strip()
                if not cleaned or re.fullmatch(r"\|?[\s:-]+\|?", cleaned):
                    continue
                candidates.extend(
                    sentence.strip()
                    for sentence in re.split(
                        r"(?<=[.!?])\s+|\s*[;]\s*", cleaned
                    )
                    if len(re.findall(r"\b[\w'-]+\b", sentence)) >= 3
                )
    return candidates


def source_span_facet(text: str, query: str, *, max_words: int = 80) -> str:
    """Select a complete query-bearing source span for facet expansion."""
    candidates = _facet_sentences(text)
    if not candidates:
        return ""
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (-lexical_overlap(query, pair[1]), pair[0]),
    )

    def with_antecedent(index: int, sentence: str) -> str:
        """Keep a local antecedent without cutting the selected evidence."""
        if index > 0 and re.match(
            r"^(this|that|these|those|it|this approach|this option)\b",
            sentence,
            re.IGNORECASE,
        ):
            return f"{candidates[index - 1]} {sentence}"
        if sentence.endswith(":") and index + 1 < len(candidates):
            return f"{sentence} {candidates[index + 1]}"
        return sentence

    spans = [
        with_antecedent(index, sentence)
        for index, sentence in ranked
    ]
    for sentence in spans:
        words = re.findall(r"\b[\w'-]+\b", sentence)
        if len(words) <= max_words:
            return " ".join(words)

    # A long source span is preferable to a truncated qualifier or causal
    # relation. The limit is a preference for compact facets, not a boundary
    # that permits cutting evidence.
    sentence = spans[0]
    words = re.findall(r"\b[\w'-]+\b", sentence)
    return " ".join(words)


def _validate_config(config: FacetRetrievalConfig) -> None:
    if config.top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if config.anchor_pool_k <= 0 or config.anchor_k <= 0:
        raise ValueError("anchor limits must be greater than zero")
    if config.facet_top_k <= 0 or config.rrf_k <= 0:
        raise ValueError("facet_top_k and rrf_k must be greater than zero")
    if config.base_weight < 0 or config.facet_weight < 0:
        raise ValueError("fusion weights cannot be negative")
    if not 0.0 <= config.min_anchor_lexical_score <= 1.0:
        raise ValueError("min_anchor_lexical_score must be between zero and one")
    if config.min_anchor_term_hits <= 0:
        raise ValueError("min_anchor_term_hits must be greater than zero")
    if config.max_facet_words <= 0:
        raise ValueError("max_facet_words must be greater than zero")


def _fallback_results(
    results: list[dict[str, Any]],
    *,
    reason: str,
    anchor_query: str = "",
    required_anchor_terms: set[str] | None = None,
) -> list[dict[str, Any]]:
    fallback = []
    for item in results:
        copied = dict(item)
        copied["retrieval_strategy"] = "facet_fallback"
        copied["facet_trace"] = {
            "status": "fallback",
            "reason": reason,
            "anchor_document_ids": [],
            "anchor_queries": [anchor_query] if anchor_query else [],
            "required_anchor_terms": sorted(required_anchor_terms or set()),
            "facet_queries": [],
            "rank_contributions": [],
        }
        fallback.append(copied)
    return fallback


def retrieve_faceted_many(
    queries: list[str],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    *,
    config: FacetRetrievalConfig | None = None,
    view: str | None = None,
    collection: str | None = None,
    embedding_metadata: dict[str, Any] | None = None,
    session: EmbeddingSession | None = None,
    anchor_queries: list[str] | None = None,
    anchor_term_sets: list[set[str]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Fuse base and anchor-derived facet rankings without forced expansion."""
    config = config or FacetRetrievalConfig()
    _validate_config(config)
    if not queries:
        return []
    active_session = session or EmbeddingSession(vectors, embedding_metadata)
    base_results = retrieve_many(
        queries,
        documents,
        vectors,
        top_k=max(config.top_k, config.anchor_pool_k),
        view=view,
        collection=collection,
        embedding_metadata=embedding_metadata,
        session=active_session,
    )
    fused_results: list[list[dict[str, Any]]] = []
    if anchor_queries is not None and len(anchor_queries) != len(queries):
        raise ValueError("anchor_queries must have one query per retrieval query")
    if anchor_term_sets is not None and len(anchor_term_sets) != len(queries):
        raise ValueError("anchor_term_sets must have one set per retrieval query")
    for query_index, (query, base) in enumerate(zip(queries, base_results)):
        anchor_query = (
            anchor_queries[query_index] if anchor_queries is not None else query
        )
        required_anchor_terms = (
            anchor_term_sets[query_index] if anchor_term_sets is not None else set()
        )
        qualified_anchors: list[tuple[dict[str, Any], str]] = []
        for item in base[: config.anchor_pool_k]:
            text = item["document"].get("text", "")
            if (
                lexical_overlap(anchor_query, text)
                < config.min_anchor_lexical_score
            ):
                continue
            facet = source_span_facet(
                text,
                anchor_query,
                max_words=config.max_facet_words,
            )
            facet_terms = lexical_terms(facet)
            term_hits = len(facet_terms & required_anchor_terms)
            if required_anchor_terms and term_hits < config.min_anchor_term_hits:
                continue
            if not facet:
                continue
            qualified_anchors.append((item, facet))
            if len(qualified_anchors) >= config.anchor_k:
                break
        anchors = [item for item, _facet in qualified_anchors]
        facets = [facet for _item, facet in qualified_anchors]
        if not facets:
            fused_results.append(
                _fallback_results(
                    base[: config.top_k],
                    reason="no_qualified_anchors",
                    anchor_query=anchor_query,
                    required_anchor_terms=required_anchor_terms,
                )
            )
            continue
        facet_queries = [f"{query} {facet}" for facet in facets]
        facet_results = retrieve_many(
            facet_queries,
            documents,
            vectors,
            top_k=config.facet_top_k,
            view=view,
            collection=collection,
            embedding_metadata=embedding_metadata,
            session=active_session,
        )
        base_by_id: dict[str, dict[str, Any]] = {}
        facet_by_id: dict[str, dict[str, Any]] = {}
        rank_contributions: dict[str, list[dict[str, Any]]] = {}
        rrf_scores: dict[str, float] = {}

        for rank, item in enumerate(base, start=1):
            document_id = item["document"]["document_id"]
            rrf_scores[document_id] = rrf_scores.get(document_id, 0.0) + (
                config.base_weight / (config.rrf_k + rank)
            )
            base_by_id.setdefault(document_id, item)
            rank_contributions.setdefault(document_id, []).append(
                {"query": query, "rank": rank, "kind": "base"}
            )
        for facet_query, results in zip(facet_queries, facet_results):
            for rank, item in enumerate(results, start=1):
                document_id = item["document"]["document_id"]
                rrf_scores[document_id] = rrf_scores.get(document_id, 0.0) + (
                    config.facet_weight / (config.rrf_k + rank)
                )
                current = facet_by_id.get(document_id)
                if current is None or item["score"] > current["score"]:
                    facet_by_id[document_id] = item
                rank_contributions.setdefault(document_id, []).append(
                    {"query": facet_query, "rank": rank, "kind": "facet"}
                )

        display_items = {
            document_id: (
                base_by_id.get(document_id) or facet_by_id.get(document_id)
            )
            for document_id in rrf_scores
        }
        ranked = sorted(
            display_items.values(),
            key=lambda item: (
                -rrf_scores[item["document"]["document_id"]],
                -max(
                    float(
                        (
                            base_by_id.get(item["document"]["document_id"]) or {}
                        ).get("score", float("-inf"))
                    ),
                    float(
                        (
                            facet_by_id.get(item["document"]["document_id"]) or {}
                        ).get("score", float("-inf"))
                    ),
                ),
                item["document"]["document_id"],
            ),
        )
        selected = []
        anchor_ids = [item["document"]["document_id"] for item in anchors]
        for rank, item in enumerate(ranked[: config.top_k], start=1):
            document_id = item["document"]["document_id"]
            copied = dict(item)
            original_query_score = (
                base_by_id.get(document_id, {}).get("score")
                if document_id in base_by_id
                else None
            )
            facet_query_score = (
                facet_by_id.get(document_id, {}).get("score")
                if document_id in facet_by_id
                else None
            )
            copied["original_query_score"] = original_query_score
            copied["facet_query_score"] = facet_query_score
            copied["base_score"] = original_query_score
            # Evidence thresholds need a comparable direct score, while the
            # separate fields preserve whether it came from the original or
            # facet query.
            copied["evidence_score"] = (
                original_query_score
                if original_query_score is not None
                else facet_query_score
            )
            copied["evidence_score_source"] = (
                "original_query" if original_query_score is not None else "facet_query"
            )
            copied["score"] = round(rrf_scores[document_id], 6)
            copied["rrf_score"] = copied["score"]
            copied["rank"] = rank
            copied["retrieval_strategy"] = "facet_rrf"
            copied["facet_trace"] = {
                "status": "expanded",
                "anchor_document_ids": anchor_ids,
                "anchor_queries": [anchor_query],
                "required_anchor_terms": sorted(required_anchor_terms),
                "facet_queries": facet_queries,
                "rank_contributions": rank_contributions[document_id],
            }
            selected.append(copied)
        fused_results.append(selected)
    return fused_results


def retrieve_faceted(
    query: str,
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Single-query convenience wrapper."""
    return retrieve_faceted_many([query], documents, vectors, **kwargs)[0]
