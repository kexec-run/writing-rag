"""Retrieve grounded writing-memory documents with collection boundaries."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

from writing_rag.retrieve.embed_chunks import (
    DEFAULT_QWEN_CUSTOM_QUERY,
    QWEN_QUERY_POLICIES,
    create_backend,
    hash_vector,
    tokenize,
)
from writing_rag.runtime.pipeline_common import read_json, write_json


NARRATIVE_ROLE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("opening", ("observation", "motivation")),
    ("turn", ("discarded_approach", "initial_hypothesis", "reframe")),
    ("method", ("experiment", "measurement")),
    ("finding", ("finding", "mechanism")),
    ("decision", ("decision", "conclusion")),
)


LEXICAL_STOPWORDS = frozenset(
    """
    a about after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more
    most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with you your yours yourself yourselves truly
    """.split()
)

TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
)


def _tokenizer_file_hashes(snapshot: str | Path | None) -> dict[str, str]:
    if snapshot is None:
        return {}
    root = Path(snapshot)
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name in TOKENIZER_FILES
    }


class EmbeddingSession:
    """Reuse one query encoder and cache across a complete retrieval run."""

    def __init__(
        self,
        vectors: list[dict[str, Any]],
        embedding_metadata: dict[str, Any] | None,
        *,
        encoder_session: "EmbeddingSession | None" = None,
    ) -> None:
        self.vectors = vectors
        self.metadata = embedding_metadata or {"embedding_backend": "hashing"}
        self.dimension = len(vectors[0]["embedding"]) if vectors else 0
        if any(len(item["embedding"]) != self.dimension for item in vectors):
            raise ValueError("stored embedding dimensions are inconsistent")
        declared_dimension = self.metadata.get("dimension")
        if declared_dimension is not None and int(declared_dimension) != self.dimension:
            raise ValueError(
                f"stored embedding dimension {self.dimension} does not match metadata "
                f"dimension {declared_dimension}"
            )
        if (
            self.metadata.get("embedding_backend") == "qwen3"
            and self.metadata.get("normalized") is False
        ):
            raise ValueError("Qwen retrieval requires normalized document vectors")
        self.vector_by_id = {
            item["document_id"]: item["embedding"] for item in vectors
        }
        self._backend: Any = None
        self._encoder_session = encoder_session
        self.query_policy = str(self.metadata.get("query_encoding_policy", "plain"))
        if self.query_policy not in QWEN_QUERY_POLICIES:
            raise ValueError(f"Unsupported query encoding policy: {self.query_policy}")
        self.custom_query_prompt = self.metadata.get("query_encoding_prompt")
        if self.query_policy == "qwen_custom_query" and not self.custom_query_prompt:
            self.custom_query_prompt = DEFAULT_QWEN_CUSTOM_QUERY
        self.query_prompt_sha256 = hashlib.sha256(
            str(self.custom_query_prompt or self.metadata.get("qwen_query_prompt") or "")
            .encode("utf-8")
        ).hexdigest()
        self._cache: dict[tuple[str, str, str], list[float]] = {}

    def _load_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self._encoder_session is not None:
            self._backend = self._encoder_session._load_backend()
            return self._backend
        requested_device = self.metadata.get("device")
        device = None if requested_device in {None, "", "auto"} else requested_device
        self._backend, backend_metadata = create_backend(
            "qwen3",
            model_name=self.metadata.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
            device=device,
            dimension=self.dimension,
            model_cache=self.metadata.get("model_cache"),
        )
        expected_revision = self.metadata.get("model_snapshot_revision")
        resolved_model = getattr(self._backend, "resolved_model", None)
        actual_revision = Path(resolved_model).name if resolved_model else None
        if expected_revision and expected_revision != actual_revision:
            raise RuntimeError(
                f"model snapshot mismatch: expected {expected_revision}, "
                f"loaded {actual_revision}"
            )
        expected_pooling = self.metadata.get("pooling_config_sha256")
        actual_pooling = backend_metadata.get("pooling_config_sha256")
        if expected_pooling and expected_pooling != actual_pooling:
            raise RuntimeError("pooling configuration mismatch")
        expected_tokenizers = self.metadata.get("tokenizer_files")
        if expected_tokenizers and expected_tokenizers != _tokenizer_file_hashes(resolved_model):
            raise RuntimeError("tokenizer file mismatch")
        return self._backend

    def encode(self, queries: list[str]) -> list[list[float]]:
        if not queries or not self.dimension:
            return [[] for _ in queries]
        backend: Any = None
        if self.metadata.get("embedding_backend") == "qwen3":
            backend = self._load_backend()
            if self.query_policy == "qwen_card_query":
                model_prompt = getattr(backend, "query_prompt", None)
                if not model_prompt:
                    raise RuntimeError(
                        "Qwen card query policy requires a model-defined 'query' prompt"
                    )
                self.query_prompt_sha256 = hashlib.sha256(
                    model_prompt.encode("utf-8")
                ).hexdigest()
        def cache_key(query: str) -> tuple[str, str, str]:
            return (query, self.query_policy, self.query_prompt_sha256)

        missing = list(
            dict.fromkeys(query for query in queries if cache_key(query) not in self._cache)
        )
        if missing:
            if self.metadata.get("embedding_backend") == "hashing":
                encoded = [hash_vector(query, self.dimension) for query in missing]
            elif self.metadata.get("embedding_backend") == "qwen3":
                if hasattr(backend, "encode_queries"):
                    encoded = backend.encode_queries(
                        missing,
                        batch_size=min(16, len(missing)),
                        policy=self.query_policy,
                        custom_prompt=self.custom_query_prompt,
                    )
                elif self.query_policy == "plain":
                    encoded = backend.encode(missing, batch_size=min(16, len(missing)))
                else:
                    raise RuntimeError(
                        "Configured Qwen query policy requires encode_queries support"
                    )
            else:
                raise ValueError("Unknown embedding backend in metadata")
            for vector in encoded:
                if len(vector) != self.dimension:
                    raise ValueError(
                        "query embedding dimension does not match stored document vectors"
                    )
                if self.metadata.get("normalized") is True or self.metadata.get(
                    "embedding_backend"
                ) == "qwen3":
                    norm = math.sqrt(sum(value * value for value in vector))
                    if not math.isclose(norm, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                        raise ValueError("query embedding is not normalized")
            self._cache.update(
                (cache_key(query), vector) for query, vector in zip(missing, encoded)
            )
        return [self._cache[cache_key(query)] for query in queries]


def lexical_terms(text: str) -> set[str]:
    """Return content-bearing retrieval terms without discourse stopwords."""
    return {
        term for term in tokenize(text) if term not in LEXICAL_STOPWORDS
    }


def is_content_evidence_candidate(document: dict[str, Any]) -> bool:
    """Exclude assistant-only ChatGPT windows from content evidence."""
    metadata = document.get("metadata", {})
    return not (
        metadata.get("source", "").casefold() == "chatgpt"
        and not str(metadata.get("self_message_ids", "")).strip()
    )


def lexical_overlap(
    query: str,
    text: str,
    *,
    idf: dict[str, float] | None = None,
) -> float:
    query_terms = lexical_terms(query)
    document_terms = lexical_terms(text)
    if not query_terms:
        return 0.0
    weights = {
        term: (idf.get(term, 1.0) if idf is not None else 1.0)
        for term in query_terms
    }
    denominator = sum(weights.values())
    return sum(
        weight for term, weight in weights.items() if term in document_terms
    ) / denominator


def query_embedding(
    query: str,
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any] | None,
    session: EmbeddingSession | None = None,
) -> list[float]:
    active_session = session or EmbeddingSession(vectors, embedding_metadata)
    return active_session.encode([query])[0]


def query_embeddings(
    queries: list[str],
    vectors: list[dict[str, Any]],
    embedding_metadata: dict[str, Any] | None,
    session: EmbeddingSession | None = None,
) -> list[list[float]]:
    """Embed several queries with one backend instance.

    Causal retrieval asks several related questions. Loading the embedding model
    for every question is both slow and needlessly memory-intensive.
    """
    active_session = session or EmbeddingSession(vectors, embedding_metadata)
    return active_session.encode(queries)


def retrieve(
    query: str,
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    *,
    top_k: int = 8,
    view: str | None = None,
    collection: str | None = None,
    embedding_metadata: dict[str, Any] | None = None,
    session: EmbeddingSession | None = None,
    ranking_mode: str = "hybrid",
) -> list[dict[str, Any]]:
    return retrieve_many(
        [query],
        documents,
        vectors,
        top_k=top_k,
        view=view,
        collection=collection,
        embedding_metadata=embedding_metadata,
        session=session,
        ranking_mode=ranking_mode,
    )[0]


def retrieve_many(
    queries: list[str],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    *,
    top_k: int = 8,
    view: str | None = None,
    collection: str | None = None,
    embedding_metadata: dict[str, Any] | None = None,
    session: EmbeddingSession | None = None,
    ranking_mode: str = "hybrid",
) -> list[list[dict[str, Any]]]:
    """Retrieve queries with stopword-aware, IDF-weighted lexical evidence."""
    if top_k <= 0:
        return [[] for _ in queries]
    if ranking_mode not in {"dense_only", "hybrid"}:
        raise ValueError(f"Unsupported ranking mode: {ranking_mode}")
    active_session = session or EmbeddingSession(vectors, embedding_metadata)
    vector_by_id = active_session.vector_by_id
    query_vectors = query_embeddings(
        queries,
        vectors,
        embedding_metadata,
        session=active_session,
    )
    eligible_documents = [
        document
        for document in documents
        if (not view or document.get("view") == view)
        and (not collection or document.get("collection") == collection)
        and (
            document.get("collection") != "content_memory"
            or document.get("view") != "content"
            or is_content_evidence_candidate(document)
        )
    ]
    term_sets = {
        document["document_id"]: lexical_terms(document["text"])
        for document in eligible_documents
    }
    results: list[list[dict[str, Any]]] = []
    for query, query_vector in zip(queries, query_vectors):
        query_terms = lexical_terms(query)
        document_frequency = {
            term: sum(term in terms for terms in term_sets.values())
            for term in query_terms
        }
        document_count = len(eligible_documents)
        idf = {
            term: math.log((document_count + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        ranked = []
        for document in eligible_documents:
            embedding = vector_by_id.get(document["document_id"])
            if embedding is None:
                continue
            cosine_score = sum(
                left * right for left, right in zip(query_vector, embedding)
            )
            lexical_score = lexical_overlap(
                query,
                document["text"],
                idf=idf,
            )
            score = (
                cosine_score
                if ranking_mode == "dense_only"
                else 0.65 * cosine_score + 0.35 * lexical_score
            )
            ranked.append(
                {
                    "score": round(score, 6),
                    "cosine_score": round(cosine_score, 6),
                    "lexical_score": round(lexical_score, 6),
                    "ranking_mode": ranking_mode,
                    "document": document,
                }
            )
        ranked.sort(
            key=lambda item: (-item["score"], item["document"]["document_id"])
        )
        for rank, item in enumerate(ranked[:top_k], start=1):
            item["rank"] = rank
        results.append(ranked[:top_k])
    return results


def retrieve_multi_query(
    queries: list[str],
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    *,
    top_k: int = 8,
    candidate_k: int | None = None,
    rrf_k: int = 60,
    fusion: str = "max",
    view: str | None = None,
    collection: str | None = None,
    embedding_metadata: dict[str, Any] | None = None,
    session: EmbeddingSession | None = None,
) -> list[dict[str, Any]]:
    """Fuse bounded query-specific rankings while preserving evidence scores."""
    if top_k <= 0 or rrf_k <= 0:
        return []
    if fusion not in {"max", "rrf"}:
        raise ValueError("fusion must be 'max' or 'rrf'")
    unique_queries = list(
        dict.fromkeys(query.strip() for query in queries if query.strip())
    )
    if not unique_queries:
        return []
    candidate_k = max(top_k, candidate_k or top_k * 4)
    per_query = retrieve_many(
        unique_queries,
        documents,
        vectors,
        top_k=candidate_k,
        view=view,
        collection=collection,
        embedding_metadata=embedding_metadata,
        session=session,
    )
    by_id: dict[str, dict[str, Any]] = {}
    fusion_scores: dict[str, float] = {}
    contributions: dict[str, list[dict[str, Any]]] = {}
    for query, ranked in zip(unique_queries, per_query, strict=True):
        for rank, item in enumerate(ranked, start=1):
            document_id = item["document"]["document_id"]
            if fusion == "rrf":
                fusion_scores[document_id] = fusion_scores.get(document_id, 0.0) + (
                    1.0 / (rrf_k + rank)
                )
            else:
                fusion_scores[document_id] = max(
                    fusion_scores.get(document_id, 0.0),
                    item["score"],
                )
            current = by_id.get(document_id)
            if current is None or item["score"] > current["score"]:
                by_id[document_id] = item
            contributions.setdefault(document_id, []).append(
                {
                    "query": query,
                    "rank": rank,
                    "score": item["score"],
                }
            )
    ranked = sorted(
        by_id.values(),
        key=lambda item: (
            -fusion_scores[item["document"]["document_id"]],
            -len(contributions[item["document"]["document_id"]]),
            -item["score"],
            item["document"]["document_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked[:top_k], start=1):
        document_id = item["document"]["document_id"]
        copied = dict(item)
        copied["evidence_score"] = item["score"]
        copied["score"] = round(fusion_scores[document_id], 6)
        copied["rrf_score"] = (
            copied["score"] if fusion == "rrf" else None
        )
        copied["rank"] = rank
        copied["retrieval_strategy"] = f"multi_query_{fusion}"
        copied["retrieval_trace"] = {
            "queries": unique_queries,
            "candidate_k": candidate_k,
            "fusion": fusion,
            "rrf_k": rrf_k,
            "rank_contributions": contributions[document_id],
        }
        selected.append(copied)
    return selected


def narrative_roles(document: dict[str, Any]) -> set[str]:
    value = document.get("metadata", {}).get("narrative_roles", "")
    if isinstance(value, list):
        return set(value)
    return {role for role in str(value).split(",") if role}


def retrieve_narrative(
    query: str,
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    *,
    top_k: int = 5,
    embedding_metadata: dict[str, Any] | None = None,
    session: EmbeddingSession | None = None,
) -> list[dict[str, Any]]:
    """Select a score-ranked story spine when role metadata is available."""
    if top_k <= 0:
        return []
    has_role_metadata = any(
        narrative_roles(document)
        for document in documents
        if document.get("collection") == "content_memory"
    )
    pool = retrieve(
        query,
        documents,
        vectors,
        top_k=len(documents) if has_role_metadata else max(top_k * 4, 20),
        collection="content_memory",
        embedding_metadata=embedding_metadata,
        session=session,
    )
    if not has_role_metadata:
        return pool[:top_k]

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for group_name, wanted_roles in NARRATIVE_ROLE_GROUPS:
        candidates = [
            item
            for item in pool
            if item["document"]["document_id"] not in selected_ids
            and narrative_roles(item["document"]) & set(wanted_roles)
        ]
        candidates.sort(
            key=lambda item: next(
                (
                    index
                    for index, role in enumerate(wanted_roles)
                    if role in narrative_roles(item["document"])
                ),
                len(wanted_roles),
            )
        )
        if not candidates:
            continue
        chosen = candidates[0]
        chosen["narrative_role_group"] = group_name
        selected.append(chosen)
        selected_ids.add(chosen["document"]["document_id"])
        if len(selected) >= top_k:
            break

    for item in pool:
        if len(selected) >= top_k:
            break
        document_id = item["document"]["document_id"]
        if document_id not in selected_ids:
            item["narrative_role_group"] = "supporting"
            selected.append(item)
            selected_ids.add(document_id)

    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    return selected[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--vectors", default="data/embeddings/vectors.json")
    parser.add_argument("--metadata", default="data/embeddings/metadata.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--collection", choices=("content_memory", "style_memory"))
    parser.add_argument("--view", choices=("content", "summary", "reasoning", "style", "knowledge"))
    parser.add_argument("--narrative", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output")
    args = parser.parse_args()

    documents = read_json(args.documents)
    vectors = read_json(args.vectors)
    metadata = read_json(args.metadata)
    results = (
        retrieve_narrative(
            args.query,
            documents,
            vectors,
            top_k=args.top_k,
            embedding_metadata=metadata,
        )
        if args.narrative
        else retrieve(
            args.query,
            documents,
            vectors,
            top_k=args.top_k,
            view=args.view,
            collection=args.collection,
            embedding_metadata=metadata,
        )
    )
    if args.output:
        write_json(args.output, results)
    else:
        for result in results:
            document = result["document"]
            role = result.get("narrative_role_group", "")
            suffix = f" role={role}" if role else ""
            print(
                f"{result['rank']}. {result['score']:.4f} "
                f"{document['document_id']} [{document['collection']}/{document['view']}]"
                f"{suffix}"
            )
            print(document["text"][:500].replace("\n", " "))


if __name__ == "__main__":
    main()
