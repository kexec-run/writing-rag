"""Run a small Instagram-only Qwen versus zembed retrieval screen.

This reuses the frozen 60-anchor B0/C1 episode artifact.  It does not alter
Instagram ingestion, episode construction, the Qwen index, or production
retrieval.  The document text and query policy match the existing Instagram
episode experiment: self-authored text is the retrieval document and the
other-speaker neighbourhood is the fixed query.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EPISODES = ROOT / "data/generated/instagram-conversation-episodes-v2-gated-qwen-rerun/gated-results.json"
DEFAULT_MESSAGES = ROOT / "data/normalized/messages.json"
DEFAULT_OUTPUT = ROOT / "data/generated/zembed-instagram-screen-v1"
DEFAULT_CACHE = ROOT / "models/weights/huggingface"
ZEMBED_BACKEND = Path(__file__).with_name("zembed_mlx_backend.py")
sys.path.insert(0, str(ROOT / "scripts"))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _retrieval_text(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        str(by_id[message_id].get("content", ""))
        for message_id in row["primary"].get("self_message_ids", [])
    ).strip()


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _metrics(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        return {"query_count": 0, "mrr": 0.0, "ndcg_at_10": 0.0, "recall_at_10": 0.0,
                "recall_at_20": 0.0, "recall_at_50": 0.0}
    return {
        "query_count": len(ranks),
        "mrr": round(sum(1.0 / rank for rank in ranks) / len(ranks), 6),
        "ndcg_at_10": round(
            sum(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks)
            / len(ranks),
            6,
        ),
        "recall_at_10": round(sum(rank <= 10 for rank in ranks) / len(ranks), 6),
        "recall_at_20": round(sum(rank <= 20 for rank in ranks) / len(ranks), 6),
        "recall_at_50": round(sum(rank <= 50 for rank in ranks) / len(ranks), 6),
    }


def _load_zembed_backend() -> Any:
    spec = importlib.util.spec_from_file_location("zembed_instagram_backend", ZEMBED_BACKEND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ZEMBED_BACKEND}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ZembedMLXBackend


def _build_cases(
    episode_result: dict[str, Any], messages: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    by_id = {str(message["message_id"]): message for message in messages}
    queries = {str(anchor_id): str(query) for anchor_id, query in episode_result["queries"].items()}
    conditions: dict[str, dict[str, Any]] = {}
    for condition in ("baseline", "c1_full"):
        rows = episode_result["rows"][condition]
        row_by_anchor = {str(row["anchor_message_id"]): row for row in rows}
        if set(row_by_anchor) != set(queries):
            raise ValueError(f"{condition} query/document anchor IDs do not match")
        conditions[condition] = row_by_anchor
    documents = {
        f"{condition}:{anchor_id}": _retrieval_text(row, by_id)
        for condition, rows in conditions.items()
        for anchor_id, row in rows.items()
    }
    if any(not text for text in documents.values()):
        raise ValueError("empty Instagram retrieval document")
    return conditions, queries, documents


def _rank_condition(
    condition: str,
    rows: dict[str, dict[str, Any]],
    queries: dict[str, str],
    document_vectors: dict[str, list[float]],
    query_vectors: dict[str, list[float]],
) -> dict[str, Any]:
    candidate_ids = sorted(rows)
    result_rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    for anchor_id in sorted(queries):
        ranked = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -_cosine(query_vectors[anchor_id], document_vectors[f"{condition}:{candidate_id}"]),
                candidate_id,
            ),
        )
        rank = ranked.index(anchor_id) + 1
        ranks.append(rank)
        result_rows.append({
            "anchor_message_id": anchor_id,
            "fold": rows[anchor_id]["fold"],
            "band": rows[anchor_id]["band"],
            "rank": rank,
            "top10_anchor_message_ids": ranked[:10],
        })
    result: dict[str, Any] = {**_metrics(ranks), "vector_dimension": len(next(iter(document_vectors.values()))), "rows": result_rows}
    for grouping in ("fold", "band"):
        groups: dict[str, list[int]] = {}
        for row in result_rows:
            groups.setdefault(str(row[grouping]), []).append(int(row["rank"]))
        result[f"by_{grouping}"] = {key: _metrics(value) for key, value in sorted(groups.items())}
    return result


def _compare(qwen: dict[str, Any], zembed: dict[str, Any]) -> dict[str, Any]:
    qwen_rows = {row["anchor_message_id"]: row for row in qwen["rows"]}
    zembed_rows = {row["anchor_message_id"]: row for row in zembed["rows"]}
    effects: list[dict[str, Any]] = []
    for anchor_id in sorted(qwen_rows):
        q_rank = int(qwen_rows[anchor_id]["rank"])
        z_rank = int(zembed_rows[anchor_id]["rank"])
        effects.append({"anchor_message_id": anchor_id, "qwen_rank": q_rank, "zembed_rank": z_rank, "rank_delta": z_rank - q_rank,
                        "qwen_top10": q_rank <= 10, "zembed_top10": z_rank <= 10})
    return {
        "qwen_to_zembed": {
            "improved": sum(effect["rank_delta"] < 0 for effect in effects),
            "same": sum(effect["rank_delta"] == 0 for effect in effects),
            "regressed": sum(effect["rank_delta"] > 0 for effect in effects),
            "top10_rescues": sum(not effect["qwen_top10"] and effect["zembed_top10"] for effect in effects),
            "top10_damages": sum(effect["qwen_top10"] and not effect["zembed_top10"] for effect in effects),
            "max_regression": max(effect["rank_delta"] for effect in effects),
        },
        "rows": effects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default=str(DEFAULT_EPISODES))
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--model-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--qwen-batch-size", type=int, default=32)
    parser.add_argument("--zembed-batch-size", type=int, default=1)
    args = parser.parse_args()

    episode_result = _load(Path(args.episodes))
    messages = _load(Path(args.messages))
    conditions, queries, documents = _build_cases(episode_result, messages)
    condition_names = ("baseline", "c1_full")
    query_ids = sorted(queries)
    query_texts = [queries[anchor_id] for anchor_id in query_ids]

    from embed_chunks import create_backend  # type: ignore

    qwen_started = time.perf_counter()
    qwen_backend, qwen_metadata = create_backend(
        "qwen3", model_name="Qwen/Qwen3-Embedding-0.6B", device="cpu",
        model_cache=Path(args.model_cache),
    )
    qwen_queries = qwen_backend.encode_queries(query_texts, batch_size=args.qwen_batch_size, policy="qwen_card_query")
    qwen_documents = qwen_backend.encode_documents(list(documents.values()), batch_size=args.qwen_batch_size)
    qwen_elapsed = time.perf_counter() - qwen_started
    qwen_query_vectors = dict(zip(query_ids, qwen_queries))
    qwen_document_vectors = dict(zip(documents, qwen_documents))

    del qwen_backend
    ZembedMLXBackend = _load_zembed_backend()
    zembed_started = time.perf_counter()
    zembed_backend = ZembedMLXBackend("lexrivera/zembed-1-embedding-mlx-6Bit", model_cache=Path(args.model_cache), batch_size=args.zembed_batch_size)
    zembed_queries = zembed_backend.encode_queries(query_texts, batch_size=args.zembed_batch_size)
    zembed_documents = zembed_backend.encode_documents(list(documents.values()), batch_size=args.zembed_batch_size)
    zembed_elapsed = time.perf_counter() - zembed_started
    zembed_query_vectors = dict(zip(query_ids, zembed_queries))
    zembed_document_vectors = dict(zip(documents, zembed_documents))

    result: dict[str, Any] = {
        "schema_version": "zembed-instagram-screen-v1",
        "status": "exploratory",
        "scope": {
            "source": "instagram",
            "anchor_count": len(query_ids),
            "fold_count": 5,
            "bands": ["p90_band", "p95_band", "p99_band"],
            "conditions": list(condition_names),
            "document_policy": "self_authored_messages_only",
            "query_policy": "fixed_other_speaker_neighbourhood",
            "same_documents_and_queries": True,
            "document_text_hash": _hash_texts(list(documents.values())),
            "query_text_hash": _hash_texts(query_texts),
        },
        "models": {
            "qwen": {"metadata": qwen_metadata, "elapsed_seconds": round(qwen_elapsed, 6), "batch_size": args.qwen_batch_size},
            "zembed": {"metadata": zembed_backend.contract_metadata(), "elapsed_seconds": round(zembed_elapsed, 6), "batch_size": args.zembed_batch_size},
        },
        "conditions": {},
    }
    for condition in condition_names:
        qwen_result = _rank_condition(condition, conditions[condition], queries, qwen_document_vectors, qwen_query_vectors)
        zembed_result = _rank_condition(condition, conditions[condition], queries, zembed_document_vectors, zembed_query_vectors)
        result["conditions"][condition] = {
            "qwen": qwen_result,
            "zembed": zembed_result,
            "comparison": _compare(qwen_result, zembed_result),
        }

    output = Path(args.output)
    _write(output / "instagram-screen.json", result)
    print(json.dumps({"output": str(output / "instagram-screen.json"), "anchor_count": len(query_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
