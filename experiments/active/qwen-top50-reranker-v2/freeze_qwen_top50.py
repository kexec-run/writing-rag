"""Freeze exact per-query Qwen top-50 reranker substrates.

The retriever remains Qwen3-Embedding-0.6B. This experiment persists the exact
query and candidate strings so later rerankers cannot silently see a different
representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data/generated/instagram-jina-candidate-selection-v1/jina-selection-results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/qwen-top50-reranker-v2/frozen-top50.json"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models/weights/huggingface"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
CONDITIONS = ("baseline", "c1_full", "jina_selected")
QUERY_COUNT = 60
POOL_SIZE = 50


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _retrieval_text(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        str(by_id[message_id].get("content", ""))
        for message_id in row["primary"].get("self_message_ids", [])
    ).strip()


def _metrics(ranks: list[int]) -> dict[str, float | int]:
    count = len(ranks)
    return {
        "query_count": count,
        "recall_at_5": round(sum(rank <= 5 for rank in ranks) / count, 6),
        "recall_at_10": round(sum(rank <= 10 for rank in ranks) / count, 6),
        "recall_at_50": round(sum(rank <= 50 for rank in ranks) / count, 6),
        "mrr_at_50": round(
            sum(1.0 / rank if rank <= 50 else 0.0 for rank in ranks) / count,
            6,
        ),
        "ndcg_at_10": round(
            sum(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks)
            / count,
            6,
        ),
    }


def _source_result(path: Path, validation_path: Path | None = None) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("structural", {}).get("passed") is not True:
        if validation_path is None:
            raise ValueError("source structural checks did not pass")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if validation.get("source_sha256") != expected_hash or validation.get("passed") is not True:
            raise ValueError("confirmation selection validation did not pass")
    qwen = result.get("qwen_retrieval")
    if qwen is not None:
        if qwen.get("backend", {}).get("embedding_backend") != "qwen3":
            raise ValueError("source retrieval backend is not Qwen")
        if qwen.get("query_policy") != "qwen_card_query":
            raise ValueError("source does not use the Qwen card query policy")
    query_count = len(result.get("queries", {}))
    if query_count <= 0:
        raise ValueError("source contains no queries")
    for condition in CONDITIONS:
        if len(result.get("rows", {}).get(condition, [])) != query_count:
            raise ValueError(f"source row count mismatch for {condition}")
    return result


def build_frozen(
    source_path: Path,
    *,
    model_cache: Path,
    batch_size: int,
    validation_path: Path | None = None,
) -> dict[str, Any]:
    source = _source_result(source_path, validation_path=validation_path)
    messages_path = Path(source["scope"]["messages_path"])
    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    by_id = {str(message["message_id"]): message for message in messages}
    queries = source["queries"]
    source_rows = source["rows"]
    document_texts = {
        condition: {
            str(row["anchor_message_id"]): _retrieval_text(row, by_id)
            for row in source_rows[condition]
        }
        for condition in CONDITIONS
    }
    if any(not text for rows in document_texts.values() for text in rows.values()):
        raise ValueError("cannot freeze an empty candidate text")

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from embed_chunks import create_backend

    backend, backend_metadata = create_backend(
        "qwen3",
        model_name=MODEL_ID,
        device="cpu",
        model_cache=model_cache,
    )
    query_ids = sorted(queries)
    query_texts = [str(queries[query_id]) for query_id in query_ids]
    query_vectors = backend.encode_queries(
        query_texts,
        batch_size=batch_size,
        policy="qwen_card_query",
    )
    unique_doc_ids = sorted(
        f"{condition}:{candidate_id}"
        for condition in CONDITIONS
        for candidate_id in document_texts[condition]
    )
    document_vectors = backend.encode_documents(
        [
            document_texts[condition][candidate_id]
            for condition, candidate_id in (
                item.split(":", 1) for item in unique_doc_ids
            )
        ],
        batch_size=batch_size,
    )
    vector_by_id = dict(zip(unique_doc_ids, document_vectors))
    frozen_conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        candidate_ids = sorted(document_texts[condition])
        source_rows_by_id = {
            str(row["anchor_message_id"]): row
            for row in source_rows[condition]
        }
        query_rows: list[dict[str, Any]] = []
        gold_ranks: list[int] = []
        for query_index, query_id in enumerate(query_ids):
            ranked = sorted(
                candidate_ids,
                key=lambda candidate_id: (
                    -_cosine(
                        query_vectors[query_index],
                        vector_by_id[f"{condition}:{candidate_id}"],
                    ),
                    candidate_id,
                ),
            )
            scores = {
                candidate_id: _cosine(
                    query_vectors[query_index],
                    vector_by_id[f"{condition}:{candidate_id}"],
                )
                for candidate_id in candidate_ids
            }
            gold_rank = ranked.index(query_id) + 1
            gold_ranks.append(gold_rank)
            query_rows.append(
                {
                    "query_id": query_id,
                    "query_text": str(queries[query_id]),
                    "query_sha256": _sha256_text(str(queries[query_id])),
                    "fold": source_rows_by_id[query_id]["fold"],
                    "band": source_rows_by_id[query_id]["band"],
                    "gold_candidate_id": query_id,
                    "gold_native_rank": gold_rank,
                    "native_top5": gold_rank <= 5,
                    "native_top10": gold_rank <= 10,
                    "candidates": [
                        {
                            "candidate_id": candidate_id,
                            "candidate_text": document_texts[condition][candidate_id],
                            "candidate_sha256": _sha256_text(
                                document_texts[condition][candidate_id]
                            ),
                            "native_score": round(scores[candidate_id], 8),
                            "native_rank": rank,
                            "is_gold": candidate_id == query_id,
                        }
                        for rank, candidate_id in enumerate(ranked[:POOL_SIZE], 1)
                    ],
                }
            )
        frozen_conditions[condition] = {
            "query_count": len(query_rows),
            "pool_size": POOL_SIZE,
            "native_metrics": _metrics(gold_ranks),
            "queries": query_rows,
        }

    return {
        "schema_version": "qwen-top50-frozen-substrate-v1",
        "status": "frozen",
        "source_artifact": str(source_path),
        "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "scope": {
            "messages_path": str(messages_path),
            "condition_order": list(CONDITIONS),
            "query_count": len(query_ids),
            "pool_size": POOL_SIZE,
            "gold_definition": "query anchor message is the single relevant candidate",
        },
        "retriever": {
            "backend": backend_metadata,
            "model": MODEL_ID,
            "query_policy": "qwen_card_query",
            "device": "cpu",
            "batch_size": batch_size,
            "tie_break": "candidate_id ascending after descending cosine",
        },
        "invariants": {
            "candidate_text_frozen": True,
            "query_text_frozen": True,
            "gold_labels_frozen": True,
            "rerankers_may_only_change_order": True,
            "jina_is_not_retriever": True,
        },
        "conditions": frozen_conditions,
    }


def validate_frozen(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != "qwen-top50-frozen-substrate-v1":
        raise ValueError("unexpected frozen substrate schema")
    if artifact.get("status") != "frozen":
        raise ValueError("substrate is not marked frozen")
    scope = artifact.get("scope", {})
    query_count = scope.get("query_count")
    if not isinstance(query_count, int) or query_count <= 0 or scope.get("pool_size") != POOL_SIZE:
        raise ValueError("frozen scope cardinality is invalid")
    if artifact.get("invariants", {}).get("rerankers_may_only_change_order") is not True:
        raise ValueError("reranker order-only invariant is missing")
    for condition in CONDITIONS:
        condition_data = artifact.get("conditions", {}).get(condition)
        if not condition_data or condition_data.get("query_count") != query_count:
            raise ValueError(f"invalid condition: {condition}")
        for row in condition_data.get("queries", []):
            query_text = row.get("query_text")
            if row.get("query_sha256") != _sha256_text(str(query_text)):
                raise ValueError(f"query hash mismatch: {condition}/{row.get('query_id')}")
            candidates = row.get("candidates", [])
            if len(candidates) != POOL_SIZE:
                raise ValueError(f"candidate pool size mismatch: {condition}/{row.get('query_id')}")
            ids = [candidate.get("candidate_id") for candidate in candidates]
            if len(set(ids)) != POOL_SIZE:
                raise ValueError(f"duplicate candidate ID: {condition}/{row.get('query_id')}")
            if [candidate.get("native_rank") for candidate in candidates] != list(range(1, POOL_SIZE + 1)):
                raise ValueError(f"native order mismatch: {condition}/{row.get('query_id')}")
            for candidate in candidates:
                text = str(candidate.get("candidate_text"))
                if candidate.get("candidate_sha256") != _sha256_text(text):
                    raise ValueError(
                        f"candidate hash mismatch: {condition}/{row.get('query_id')}"
                    )
            gold_id = row.get("gold_candidate_id")
            gold_rank = int(row.get("gold_native_rank"))
            gold_in_pool = gold_id in ids
            if gold_in_pool != (gold_rank <= POOL_SIZE):
                raise ValueError(f"gold pool membership mismatch: {condition}/{row.get('query_id')}")
            if gold_in_pool and candidates[gold_rank - 1].get("candidate_id") != gold_id:
                raise ValueError(f"gold rank mismatch: {condition}/{row.get('query_id')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-cache", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selection-validation")
    args = parser.parse_args()
    artifact = build_frozen(
        Path(args.input),
        model_cache=Path(args.model_cache),
        batch_size=args.batch_size,
        validation_path=Path(args.selection_validation) if args.selection_validation else None,
    )
    validate_frozen(artifact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": artifact["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
