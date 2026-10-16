"""Build embeddings for retrieval documents.

Qwen3-Embedding-0.6B is the preferred semantic backend. Hashing is retained
only as an explicit/offline fallback so the artifact contract remains usable
when the model runtime or weights are unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Any

from writing_rag.runtime.model_paths import configure_model_cache, resolve_cached_model_path
from writing_rag.runtime.pipeline_common import read_json, write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")
DEFAULT_DIMENSION = 256
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_QUERY_POLICIES = ("plain", "qwen_card_query", "qwen_custom_query")
DEFAULT_QWEN_CUSTOM_QUERY = (
    "Given a question for a personal writing-memory corpus, retrieve passages "
    "that directly provide evidence for answering the question."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def hash_vector(text: str, dimension: int = DEFAULT_DIMENSION) -> list[float]:
    if dimension <= 0:
        raise ValueError("dimension must be greater than zero")
    vector = [0.0] * dimension
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [round(value / norm, 8) for value in vector] if norm else vector


class QwenEmbeddingBackend:
    name = "qwen3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        model_cache: str | Path | None = None,
    ) -> None:
        self.model_cache = str(configure_model_cache(model_cache))
        resolved_model = resolve_cached_model_path(model_name, self.model_cache)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen embeddings require sentence-transformers. "
                "Install requirements-qwen.txt or use --backend hashing explicitly."
            ) from exc
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device:
            kwargs["device"] = device
        try:
            self.model = SentenceTransformer(str(resolved_model), **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {model_name} from {resolved_model}. "
                "Check local model weights or install the required runtime."
            ) from exc
        self.model_name = model_name
        self.resolved_model = str(resolved_model)
        self.device = device or "auto"
        self.last_output_dtypes: dict[str, str] = {}

    @property
    def query_prompt(self) -> str | None:
        prompts = getattr(self.model, "prompts", None) or {}
        prompt = prompts.get("query")
        return str(prompt) if prompt is not None else None

    @property
    def pooling_config_sha256(self) -> str | None:
        path = Path(self.resolved_model) / "1_Pooling" / "config.json"
        return sha256_file(path) if path.is_file() else None

    def encode_documents(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        return self._encode(texts, batch_size=batch_size)

    def encode_queries(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        policy: str = "plain",
        custom_prompt: str | None = None,
    ) -> list[list[float]]:
        if policy not in QWEN_QUERY_POLICIES:
            raise ValueError(f"Unsupported Qwen query encoding policy: {policy}")
        if policy == "qwen_card_query":
            if not self.query_prompt:
                raise RuntimeError(
                    "Qwen card query policy requires a model-defined 'query' prompt"
                )
            return self._encode(texts, batch_size=batch_size, prompt_name="query")
        if policy == "qwen_custom_query":
            prompt = custom_prompt or DEFAULT_QWEN_CUSTOM_QUERY
            return self._encode(texts, batch_size=batch_size, prompt=prompt)
        return self._encode(texts, batch_size=batch_size)

    def _encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        if prompt is not None:
            kwargs["prompt"] = prompt
        vectors = self.model.encode(texts, **kwargs)
        output_dtype = str(getattr(vectors, "dtype", "unknown"))
        self.last_output_dtype = output_dtype
        if not hasattr(self, "last_output_dtypes"):
            self.last_output_dtypes = {}
        self.last_output_dtypes["latest"] = output_dtype
        return vectors.tolist()

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Backward-compatible plain encoding, used for documents and Q0 queries."""
        return self.encode_documents(texts, batch_size=batch_size)


def create_backend(
    backend: str,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    dimension: int = DEFAULT_DIMENSION,
    model_cache: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    if backend == "hashing":
        return (
            None,
            {
                "embedding_backend": "hashing",
                "embedding_model": "hashing-v1",
                "embedding_kind": "deterministic_local_baseline",
                "fallback_used": True,
                "fallback_reason": "explicit_hashing_backend",
                "dimension": dimension,
                "normalized": True,
            },
        )
    if backend not in {"auto", "qwen3"}:
        raise ValueError(f"Unsupported embedding backend: {backend}")

    try:
        qwen = QwenEmbeddingBackend(model_name, device, model_cache)
    except RuntimeError as exc:
        if backend == "qwen3":
            raise
        return (
            None,
            {
                "embedding_backend": "hashing",
                "embedding_model": "hashing-v1",
                "embedding_kind": "deterministic_local_baseline",
                "fallback_used": True,
                "fallback_reason": str(exc),
                "dimension": dimension,
                "normalized": True,
            },
        )
    return (
        qwen,
        {
            "embedding_backend": "qwen3",
            "embedding_model": model_name,
            "embedding_kind": "semantic_sentence_transformer",
            "fallback_used": False,
            "dimension": None,
            "normalized": True,
            "device": qwen.device,
            "model_cache": qwen.model_cache,
            "qwen_query_prompt": qwen.query_prompt,
            "qwen_query_prompt_sha256": (
                hashlib.sha256(qwen.query_prompt.encode("utf-8")).hexdigest()
                if qwen.query_prompt
                else None
            ),
            "pooling_config_sha256": qwen.pooling_config_sha256,
        },
    )


def build_vectors(
    documents: list[dict[str, Any]],
    dimension: int = DEFAULT_DIMENSION,
    *,
    backend: Any = None,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    texts = [document["text"] for document in documents]
    if backend is None:
        embeddings = [hash_vector(text, dimension) for text in texts]
    else:
        embeddings = backend.encode(texts, batch_size=batch_size)
    return [
        {"document_id": document["document_id"], "embedding": embedding}
        for document, embedding in zip(documents, embeddings)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--vectors", default="data/embeddings/vectors.json")
    parser.add_argument("--metadata", default="data/embeddings/metadata.json")
    parser.add_argument(
        "--backend",
        choices=("qwen3", "auto", "hashing"),
        default="qwen3",
        help="Use Qwen for retrieval by default; choose auto or hashing explicitly for diagnostics.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-cache", help="Hugging Face home for downloaded model weights."
    )
    parser.add_argument("--device")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use 1 by default to bound Qwen attention memory on 16-GB Apple Silicon.",
    )
    args = parser.parse_args()
    model_cache = configure_model_cache(args.model_cache)

    documents = read_json(args.input)
    backend, metadata = create_backend(
        args.backend,
        model_name=args.model,
        device=args.device,
        dimension=args.dimension,
        model_cache=model_cache,
    )
    vectors = build_vectors(
        documents,
        args.dimension,
        backend=backend,
        batch_size=args.batch_size,
    )
    if vectors:
        metadata["dimension"] = len(vectors[0]["embedding"])
    metadata.update(
        {
            "document_count": len(vectors),
            "source_documents": args.input,
            "vectors_file": args.vectors,
            "model_cache": str(model_cache),
        }
    )
    write_json(args.vectors, vectors)
    write_json(args.metadata, metadata)
    print(f"Embedded documents: {len(vectors)}")
    print(f"Embedding backend: {metadata['embedding_backend']}")
    if metadata["fallback_used"]:
        print(f"Fallback reason: {metadata['fallback_reason']}")
    print(f"Embedding dimension: {metadata['dimension']}")
    print(f"Saved vectors to {args.vectors}")


if __name__ == "__main__":
    main()
