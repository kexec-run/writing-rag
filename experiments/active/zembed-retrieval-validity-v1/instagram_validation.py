"""Run full-ranking Qwen versus zembed validation on a small Instagram artifact.

The input is an episode artifact produced by the existing Instagram episode
builder.  Only its B0 and C1 60-document pools are evaluated.  The complete
local ranking is persisted because top-10 output is insufficient to diagnose
rank regressions.
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
SCREEN_PATH = Path(__file__).with_name("instagram_screen.py")
DEFAULT_EPISODES = ROOT / "data/generated/instagram-conversation-episodes-v2-gated-qwen-rerun/gated-results.json"
DEFAULT_MESSAGES = ROOT / "data/normalized/messages.json"
DEFAULT_OUTPUT = ROOT / "data/generated/zembed-instagram-validation-v1"
DEFAULT_CACHE = ROOT / "models/weights/huggingface"
sys.path.insert(0, str(ROOT / "scripts"))


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _metrics(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        return {"query_count": 0, "mrr": 0.0, "ndcg_at_10": 0.0,
                "recall_at_10": 0.0, "recall_at_20": 0.0, "recall_at_50": 0.0}
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


def _retrieval_models(args: argparse.Namespace, query_texts: list[str], document_texts: list[str]) -> dict[str, Any]:
    from embed_chunks import create_backend

    qwen_started = time.perf_counter()
    qwen_backend, qwen_metadata = create_backend(
        "qwen3",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cpu",
        model_cache=Path(args.model_cache),
    )
    qwen_queries = qwen_backend.encode_queries(
        query_texts, batch_size=args.qwen_batch_size, policy="qwen_card_query"
    )
    qwen_documents = qwen_backend.encode_documents(
        document_texts, batch_size=args.qwen_batch_size
    )
    qwen_elapsed = time.perf_counter() - qwen_started
    del qwen_backend

    backend_module = _load_module(
        Path(__file__).with_name("zembed_mlx_backend.py"), "instagram_zembed_backend"
    )
    zembed_started = time.perf_counter()
    zembed_backend = backend_module.ZembedMLXBackend(
        "lexrivera/zembed-1-embedding-mlx-6Bit",
        model_cache=Path(args.model_cache),
        batch_size=args.zembed_batch_size,
    )
    zembed_queries = zembed_backend.encode_queries(
        query_texts, batch_size=args.zembed_batch_size
    )
    zembed_documents = zembed_backend.encode_documents(
        document_texts, batch_size=args.zembed_batch_size
    )
    zembed_elapsed = time.perf_counter() - zembed_started
    return {
        "qwen": {
            "query_vectors": qwen_queries,
            "document_vectors": qwen_documents,
            "metadata": qwen_metadata,
            "elapsed_seconds": round(qwen_elapsed, 6),
            "batch_size": args.qwen_batch_size,
        },
        "zembed": {
            "query_vectors": zembed_queries,
            "document_vectors": zembed_documents,
            "metadata": zembed_backend.contract_metadata(),
            "elapsed_seconds": round(zembed_elapsed, 6),
            "batch_size": args.zembed_batch_size,
        },
    }


def _rank(
    condition: str,
    rows: dict[str, dict[str, Any]],
    query_ids: list[str],
    document_ids: list[str],
    query_vectors: list[list[float]],
    document_vectors: list[list[float]],
) -> dict[str, Any]:
    query_by_id = dict(zip(query_ids, query_vectors))
    document_by_id = dict(zip(document_ids, document_vectors))
    result_rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    for anchor_id in query_ids:
        ranked = sorted(
            document_ids,
            key=lambda document_id: (
                -_cosine(query_by_id[anchor_id], document_by_id[document_id]),
                document_id,
            ),
        )
        rank = ranked.index(f"{condition}:{anchor_id}") + 1
        ranks.append(rank)
        result_rows.append({
            "anchor_message_id": anchor_id,
            "fold": rows[anchor_id]["fold"],
            "band": rows[anchor_id]["band"],
            "rank": rank,
            "ranking": [
                {
                    "document_id": document_id,
                    "anchor_message_id": document_id.split(":", 1)[1],
                    "rank": index + 1,
                    "score": round(_cosine(query_by_id[anchor_id], document_by_id[document_id]), 9),
                }
                for index, document_id in enumerate(ranked)
            ],
        })
    result: dict[str, Any] = {**_metrics(ranks), "vector_dimension": len(document_vectors[0]), "rows": result_rows}
    for grouping in ("fold", "band"):
        grouped: dict[str, list[int]] = {}
        for row in result_rows:
            grouped.setdefault(str(row[grouping]), []).append(int(row["rank"]))
        result[f"by_{grouping}"] = {key: _metrics(value) for key, value in sorted(grouped.items())}
    return result


def _comparison(qwen: dict[str, Any], zembed: dict[str, Any]) -> dict[str, Any]:
    qwen_rows = {row["anchor_message_id"]: row for row in qwen["rows"]}
    zembed_rows = {row["anchor_message_id"]: row for row in zembed["rows"]}
    effects = []
    for anchor_id in sorted(qwen_rows):
        qwen_rank = int(qwen_rows[anchor_id]["rank"])
        zembed_rank = int(zembed_rows[anchor_id]["rank"])
        effects.append({
            "anchor_message_id": anchor_id,
            "qwen_rank": qwen_rank,
            "zembed_rank": zembed_rank,
            "rank_delta": zembed_rank - qwen_rank,
            "qwen_top10": qwen_rank <= 10,
            "zembed_top10": zembed_rank <= 10,
        })
    return {
        "summary": {
            "improved": sum(row["rank_delta"] < 0 for row in effects),
            "same": sum(row["rank_delta"] == 0 for row in effects),
            "regressed": sum(row["rank_delta"] > 0 for row in effects),
            "top10_rescues": sum(not row["qwen_top10"] and row["zembed_top10"] for row in effects),
            "top10_damages": sum(row["qwen_top10"] and not row["zembed_top10"] for row in effects),
            "max_regression": max(row["rank_delta"] for row in effects),
        },
        "rows": effects,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    screen = _load_module(SCREEN_PATH, "instagram_screen_contract")
    episode_result = _load(Path(args.episodes))
    messages = _load(Path(args.messages))
    conditions, queries, documents = screen._build_cases(episode_result, messages)
    query_ids = sorted(queries)
    query_texts = [queries[anchor_id] for anchor_id in query_ids]
    output: dict[str, Any] = {
        "schema_version": "zembed-instagram-validation-v1",
        "status": "exploratory",
        "source_artifact": str(Path(args.episodes)),
        "scope": {
            "source": "instagram",
            "anchor_count": len(query_ids),
            "fold_count": 5,
            "bands": ["p90_band", "p95_band", "p99_band"],
            "conditions": ["baseline", "c1_full"],
            "document_policy": "self_authored_messages_only",
            "query_policy": "fixed_other_speaker_neighbourhood",
            "same_documents_and_queries": True,
            "query_text_hash": _hash_texts(query_texts),
            "document_text_hashes": {
                document_id: _sha256_text(text)
                for document_id, text in documents.items()
            },
        },
    }
    models = _retrieval_models(args, query_texts, list(documents.values()))
    output["models"] = {
        name: {key: value for key, value in model.items() if key not in {"query_vectors", "document_vectors"}}
        for name, model in models.items()
    }
    for condition, rows in conditions.items():
        condition_document_ids = sorted(
            document_id for document_id in documents if document_id.startswith(f"{condition}:")
        )
        document_indexes = [list(documents).index(document_id) for document_id in condition_document_ids]
        output.setdefault("conditions", {})[condition] = {}
        for model_name, model in models.items():
            ranking = _rank(
                condition,
                rows,
                query_ids,
                condition_document_ids,
                model["query_vectors"],
                [model["document_vectors"][index] for index in document_indexes],
            )
            output["conditions"][condition][model_name] = ranking
        output["conditions"][condition]["comparison"] = _comparison(
            output["conditions"][condition]["qwen"],
            output["conditions"][condition]["zembed"],
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default=str(DEFAULT_EPISODES))
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--model-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--qwen-batch-size", type=int, default=32)
    parser.add_argument("--zembed-batch-size", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    _write(output / "validation.json", result)
    print(json.dumps({"output": str(output / "validation.json"), "anchor_count": result["scope"]["anchor_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
