import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from embed_chunks import build_vectors  # noqa: E402
from graph_retrieve import (  # noqa: E402
    GraphRetrievalConfig,
    graph_retrieve_many,
    relationship_features,
)


def document(
    document_id: str,
    text: str,
    *,
    thread: str = "thread-a",
    start: int = 0,
    end: int | None = None,
    graph_path: str | None = None,
    message_ids: str = "",
) -> dict:
    return {
        "document_id": document_id,
        "collection": "content_memory",
        "view": "content",
        "text": text,
        "metadata": {
            "source_thread_id": thread,
            "start_timestamp_ms": start,
            "end_timestamp_ms": end if end is not None else start,
            "graph_path_id": graph_path,
            "message_ids": message_ids,
        },
    }


class GraphRetrievalTests(unittest.TestCase):
    def test_relationship_requires_time_bound_for_same_thread(self) -> None:
        seed = document("seed", "anchor", start=0, message_ids="m1")
        near = document("near", "reply", start=10_000, message_ids="m2")
        far = document("far", "reply", start=3_600_000, message_ids="m3")

        near_features = relationship_features(seed, near, max_time_gap_ms=60_000)
        far_features = relationship_features(seed, far, max_time_gap_ms=60_000)

        self.assertEqual(near_features["relation_types"], ["same_thread_time_window"])
        self.assertFalse(far_features["relation_types"])

    def test_graph_path_and_message_overlap_are_explainable(self) -> None:
        seed = document(
            "seed",
            "anchor",
            graph_path="path-1",
            message_ids="m1,m2",
        )
        candidate = document(
            "candidate",
            "reply",
            thread="other-thread",
            graph_path="path-1",
            message_ids="m2,m3",
        )

        features = relationship_features(seed, candidate, max_time_gap_ms=1)

        self.assertEqual(
            features["relation_types"], ["same_graph_path", "shared_message"]
        )
        self.assertEqual(features["shared_message_ids"], ["m2"])

    def test_graph_retrieval_expands_a_nearby_candidate(self) -> None:
        documents = [
            document("seed", "anchor topic", start=0),
            document("near", "short reply", start=10_000),
            document("far", "unrelated archive", start=3_600_000),
            document("other", "short reply", thread="thread-b", start=10_000),
        ]
        vectors = build_vectors(documents, dimension=64)
        results = graph_retrieve_many(
            ["anchor"],
            documents,
            vectors,
            config=GraphRetrievalConfig(
                top_k=3,
                candidate_k=1,
                min_base_score_ratio=0.0,
                min_base_score=0.0,
                max_expanded_per_seed=4,
                max_time_gap_ms=60_000,
            ),
            embedding_metadata={"embedding_backend": "hashing"},
        )[0]

        result_by_id = {item["document"]["document_id"]: item for item in results}
        self.assertIn("seed", result_by_id)
        self.assertIn("near", result_by_id)
        self.assertEqual(result_by_id["near"]["retrieval_strategy"], "graph_time_expansion")
        self.assertIn(
            "same_thread_time_window",
            result_by_id["near"]["graph_trace"]["relation_types"],
        )
        self.assertNotIn("far", result_by_id)
        self.assertNotIn("other", result_by_id)

    def test_graph_linked_candidate_inside_pool_receives_graph_score(self) -> None:
        documents = [
            document("seed", "anchor topic", start=0),
            document("near", "anchor reply", start=10_000),
        ]
        vectors = build_vectors(documents, dimension=64)
        results = graph_retrieve_many(
            ["anchor topic"],
            documents,
            vectors,
            config=GraphRetrievalConfig(
                top_k=2,
                candidate_k=2,
                anchor_k=1,
                min_base_score_ratio=0.0,
                max_time_gap_ms=60_000,
            ),
            embedding_metadata={"embedding_backend": "hashing"},
        )[0]

        near = next(
            item for item in results if item["document"]["document_id"] == "near"
        )
        self.assertEqual(near["retrieval_strategy"], "graph_time_expansion")
        self.assertTrue(near["graph_trace"]["candidate_pool"])
        self.assertGreater(near["score"], near["base_score"])


    def test_structural_proximity_does_not_override_score_floor(self) -> None:
        documents = [
            document("seed", "anchor topic", start=0),
            document("irrelevant", "firmware version rollout", start=10_000),
            document("other", "unrelated archive", start=20_000),
        ]
        vectors = build_vectors(documents, dimension=64)
        results = graph_retrieve_many(
            ["anchor topic"],
            documents,
            vectors,
            config=GraphRetrievalConfig(
                top_k=3,
                candidate_k=1,
                max_time_gap_ms=60_000,
            ),
            collection="content_memory",
            embedding_metadata={"embedding_backend": "hashing"},
        )[0]

        self.assertEqual(
            [item["document"]["document_id"] for item in results], ["seed"]
        )
        self.assertTrue(
            all(not item["retrieval_strategy"].startswith("graph_") for item in results)
        )

    def test_graph_path_expansion_preserves_full_provenance(self) -> None:
        documents = [
            document("seed", "anchor topic", graph_path="path-1"),
            document(
                "path",
                "path companion",
                thread="other-thread",
                graph_path="path-1",
            ),
            document("other", "unrelated archive", graph_path="path-2"),
        ]
        vectors = build_vectors(documents, dimension=64)
        results = graph_retrieve_many(
            ["anchor topic"],
            documents,
            vectors,
            config=GraphRetrievalConfig(
                top_k=2,
                candidate_k=1,
                min_base_score_ratio=0.0,
                min_base_score=0.0,
                max_time_gap_ms=1,
            ),
            collection="content_memory",
            embedding_metadata={"embedding_backend": "hashing"},
        )[0]

        path_result = next(
            item for item in results if item["document"]["document_id"] == "path"
        )
        self.assertEqual(path_result["retrieval_strategy"], "graph_path_expansion")
        self.assertEqual(
            path_result["graph_trace"]["selection_reason"],
            "base_score_above_anchor_floor_and_structurally_linked",
        )
        self.assertEqual(path_result["graph_trace"]["seed_document_ids"], ["seed"])
        self.assertEqual(path_result["graph_trace"]["relation_types"], ["same_graph_path"])
    def test_collection_and_view_filters_apply_before_expansion(self) -> None:
        documents = [
            document("content", "anchor topic"),
            document("style", "short reply"),
        ]
        documents[1]["collection"] = "style_memory"
        documents[1]["view"] = "style"
        vectors = build_vectors(documents, dimension=64)

        results = graph_retrieve_many(
            ["anchor"],
            documents,
            vectors,
            config=GraphRetrievalConfig(top_k=4),
            collection="content_memory",
            view="content",
            embedding_metadata={"embedding_backend": "hashing"},
        )[0]

        self.assertEqual(
            [item["document"]["document_id"] for item in results], ["content"]
        )


if __name__ == "__main__":
    unittest.main()
