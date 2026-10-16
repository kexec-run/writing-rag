from __future__ import annotations

import sys
import unittest
from pathlib import Path

from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from embed_chunks import QwenEmbeddingBackend  # noqa: E402
from retrieve import EmbeddingSession, retrieve  # noqa: E402


class QwenQueryEncodingTests(unittest.TestCase):
    def backend(self, prompt: str | None) -> QwenEmbeddingBackend:
        backend = object.__new__(QwenEmbeddingBackend)
        backend.model = Mock()
        backend.model.prompts = {"query": prompt} if prompt is not None else {}
        backend.model.encode.return_value.tolist.return_value = [[1.0, 0.0]]
        return backend

    def test_documents_are_plain_and_qwen_card_queries_are_prompted(self) -> None:
        backend = self.backend("Instruct: retrieve evidence\nQuery:")
        backend.encode_documents(["document"])
        backend.encode_queries(["query"], policy="qwen_card_query")
        calls = backend.model.encode.call_args_list
        self.assertNotIn("prompt_name", calls[0].kwargs)
        self.assertEqual(calls[1].kwargs["prompt_name"], "query")

    def test_qwen_card_policy_fails_without_model_prompt(self) -> None:
        backend = self.backend(None)
        with self.assertRaisesRegex(RuntimeError, "model-defined 'query' prompt"):
            backend.encode_queries(["query"], policy="qwen_card_query")

    def test_custom_policy_passes_the_frozen_prompt(self) -> None:
        backend = self.backend(None)
        backend.encode_queries(["query"], policy="qwen_custom_query", custom_prompt="frozen")
        self.assertEqual(backend.model.encode.call_args.kwargs["prompt"], "frozen")

    def test_query_cache_key_contains_policy_and_prompt_hash(self) -> None:
        session = EmbeddingSession(
            [{"document_id": "d1", "embedding": [1.0, 0.0]}],
            {
                "embedding_backend": "hashing",
                "query_encoding_policy": "qwen_card_query",
            },
        )
        session.encode(["same query"])
        self.assertEqual(len(session._cache), 1)
        key = next(iter(session._cache))
        self.assertEqual(key[0], "same query")
        self.assertEqual(key[1], "qwen_card_query")
        self.assertTrue(key[2])

    def test_query_cache_keys_do_not_collide_across_policies(self) -> None:
        session = EmbeddingSession(
            [{"document_id": "d1", "embedding": [1.0, 0.0]}],
            {"embedding_backend": "hashing"},
        )
        session.encode(["same query"])
        session.query_policy = "qwen_card_query"
        session.encode(["same query"])
        self.assertEqual(len(session._cache), 2)

    def test_dimension_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match metadata dimension"):
            EmbeddingSession(
                [{"document_id": "d1", "embedding": [1.0, 0.0]}],
                {"embedding_backend": "qwen3", "dimension": 3},
            )

    def test_non_normalized_qwen_vectors_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized document vectors"):
            EmbeddingSession(
                [{"document_id": "d1", "embedding": [1.0, 0.0]}],
                {"embedding_backend": "qwen3", "normalized": False},
            )

    def test_dense_only_ranking_is_separate_from_hybrid_scoring(self) -> None:
        documents = [
            {
                "document_id": "dense-hit",
                "collection": "content_memory",
                "view": "content",
                "text": "evidence",
                "metadata": {},
            },
            {
                "document_id": "lexical-hit",
                "collection": "content_memory",
                "view": "content",
                "text": "evidence evidence evidence evidence",
                "metadata": {},
            },
        ]
        vectors = [
            {"document_id": "dense-hit", "embedding": [1.0, 0.0]},
            {"document_id": "lexical-hit", "embedding": [0.0, 1.0]},
        ]

        class FixedSession:
            vector_by_id = {"dense-hit": [1.0, 0.0], "lexical-hit": [0.0, 1.0]}

            def encode(self, queries: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in queries]

        dense = retrieve(
            "evidence",
            documents,
            vectors,
            top_k=2,
            embedding_metadata={"embedding_backend": "hashing"},
            session=FixedSession(),
            ranking_mode="dense_only",
        )
        self.assertEqual(dense[0]["document"]["document_id"], "dense-hit")
        self.assertEqual(dense[0]["ranking_mode"], "dense_only")

    def test_model_snapshot_mismatch_fails_closed(self) -> None:
        class FakeBackend:
            resolved_model = "/model/actual-revision"

        session = EmbeddingSession(
            [{"document_id": "d1", "embedding": [1.0, 0.0]}],
            {
                "embedding_backend": "qwen3",
                "model_snapshot_revision": "expected-revision",
            },
        )
        with patch(
            "writing_rag.retrieve.retrieve.create_backend",
            return_value=(FakeBackend(), {}),
        ):
            with self.assertRaisesRegex(RuntimeError, "model snapshot mismatch"):
                session.encode(["query"])


if __name__ == "__main__":
    unittest.main()
