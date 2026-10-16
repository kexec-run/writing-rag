"""Qwen retrieval and embedding (no Jina)."""

from __future__ import annotations

from .embed_chunks import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    QwenEmbeddingBackend,
    build_vectors,
    create_backend,
    hash_vector,
)
from .retrieve import (
    EmbeddingSession,
    is_content_evidence_candidate,
    lexical_overlap,
    lexical_terms,
    query_embedding,
    query_embeddings,
    retrieve,
    retrieve_many,
    retrieve_multi_query,
    retrieve_narrative,
)

__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "EmbeddingSession",
    "QwenEmbeddingBackend",
    "build_vectors",
    "create_backend",
    "hash_vector",
    "is_content_evidence_candidate",
    "lexical_overlap",
    "lexical_terms",
    "query_embedding",
    "query_embeddings",
    "retrieve",
    "retrieve_many",
    "retrieve_multi_query",
    "retrieve_narrative",
]
