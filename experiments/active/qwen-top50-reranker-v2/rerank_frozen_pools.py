"""Rerank an immutable Qwen top-50 substrate and record transitions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBSTRATE = PROJECT_ROOT / "data/generated/qwen-top50-reranker-v2/frozen-top50.json"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models/weights/huggingface/hub"
QWEN_MODEL_ID = "mku64/Qwen3-Reranker-0.6B-mlx-8Bit"
JINA_MODEL_ID = "jinaai/jina-reranker-v3.5-mlx"
CONDITIONS = ("baseline", "c1_full", "jina_selected")
POOL_SIZE = 50


def _load_freeze_module():
    path = Path(__file__).with_name("freeze_qwen_top50.py")
    spec = importlib.util.spec_from_file_location("freeze_qwen_top50", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load freeze validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = _load_freeze_module()


def _resolve_snapshot(model_cache: Path, model_id: str) -> tuple[str, str]:
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(model_cache),
        local_files_only=True,
    )
    return path, Path(path).name


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
            sum(1.0 / __import__("math").log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks)
            / count,
            6,
        ),
    }


def _rank_delta(native_rank: int, reranked_rank: int) -> int:
    return native_rank - reranked_rank


def _transition(native_rank: int, reranked_rank: int, cutoff: int) -> str:
    if native_rank > cutoff and reranked_rank <= cutoff:
        return "rescue"
    if native_rank <= cutoff and reranked_rank > cutoff:
        return "damage"
    if reranked_rank < native_rank:
        return "improved_inside_cutoff" if reranked_rank <= cutoff else "improved_outside_cutoff"
    if reranked_rank > native_rank:
        return "worsened_inside_cutoff" if native_rank <= cutoff else "worsened_outside_cutoff"
    return "unchanged"


def _load_qwen_reranker(model_path: str):
    runner_dir = PROJECT_ROOT / "experiments/active/qwen-top50-reranker"
    sys.path.insert(0, str(runner_dir))
    from rerank_frozen_top50 import MLXReranker

    return MLXReranker(model_path, max_length=8192)


def _load_jina_reranker(model_path: str):
    sys.path.insert(0, model_path)
    from rerank import MLXReranker

    projector = Path(model_path) / "projector.safetensors"
    return MLXReranker(model_path=model_path, projector_path=str(projector))


def _qwen_rank(model: Any, rows: list[dict[str, Any]], batch_size: int) -> dict[str, list[str]]:
    pairs = [
        (row["query_text"], candidate["candidate_text"])
        for row in rows
        for candidate in row["candidates"]
    ]
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    rankings: dict[str, list[str]] = {}
    offset = 0
    for row in rows:
        candidate_ids = [candidate["candidate_id"] for candidate in row["candidates"]]
        row_scores = scores[offset : offset + len(candidate_ids)]
        offset += len(candidate_ids)
        rankings[row["query_id"]] = [
            candidate_id
            for candidate_id, _ in sorted(
                zip(candidate_ids, row_scores), key=lambda item: (-item[1], item[0])
            )
        ]
    return rankings


def _jina_rank(model: Any, rows: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    rankings: dict[str, list[str]] = {}
    usage: dict[str, Any] = {}
    from rerank import _format_prompt

    for index, row in enumerate(rows, 1):
        documents = [candidate["candidate_text"] for candidate in row["candidates"]]
        query, truncated_documents, blocks = model._make_blocks(row["query_text"], documents)
        if len(blocks) != 1:
            raise ValueError(
                f"Jina split frozen pool for {row['query_id']}: {len(blocks)} blocks"
            )
        prompt = _format_prompt(query, blocks[0])
        input_tokens = len(model._tokenize_prompt(prompt))
        results = model.rerank(
            row["query_text"], documents, top_n=len(documents), return_embeddings=False
        )
        ranked_indices = [int(result["index"]) for result in results]
        if sorted(ranked_indices) != list(range(len(documents))):
            raise ValueError(f"Jina did not return a complete ranking for {row['query_id']}")
        rankings[row["query_id"]] = [
            row["candidates"][candidate_index]["candidate_id"]
            for candidate_index in ranked_indices
        ]
        usage[row["query_id"]] = {
            "candidate_count": len(documents),
            "result_count": len(results),
            "input_characters": len(row["query_text"]) + sum(map(len, documents)),
            "input_tokens": input_tokens,
            "truncated_document_count": sum(
                original != truncated
                for original, truncated in zip(documents, truncated_documents)
            ),
            "block_count": len(blocks),
        }
        if index % 10 == 0:
            print(f"scored {index}/{len(rows)} listwise queries", flush=True)
    return rankings, usage


def _subgroup_metrics(trace_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in trace_rows:
        groups.setdefault(str(row[field]), []).append(row)
    result: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        native = _metrics([int(row["native_rank"]) for row in rows])
        reranked = _metrics([int(row["reranked_rank"]) for row in rows])
        result[name] = {
            "query_count": len(rows),
            "native_metrics": native,
            "reranked_metrics": reranked,
            "deltas": {
                key: round(reranked[key] - native[key], 6)
                for key in ("recall_at_5", "recall_at_10", "mrr_at_50", "ndcg_at_10")
            },
        }
    return result


def _evaluate(rows: list[dict[str, Any]], rankings: dict[str, list[str]]) -> dict[str, Any]:
    trace_rows: list[dict[str, Any]] = []
    native_ranks: list[int] = []
    reranked_ranks: list[int] = []
    for row in rows:
        query_id = row["query_id"]
        native_rank = int(row["gold_native_rank"])
        ranked = rankings[query_id]
        gold_candidate_id = row["gold_candidate_id"]
        reranked_rank = (
            ranked.index(gold_candidate_id) + 1
            if gold_candidate_id in ranked
            else len(ranked) + 1
        )
        native_ranks.append(native_rank)
        reranked_ranks.append(reranked_rank)
        trace_rows.append(
            {
                "query_id": query_id,
                "fold": row["fold"],
                "band": row["band"],
                "native_rank": native_rank,
                "reranked_rank": reranked_rank,
                "rank_delta_native_minus_reranked": _rank_delta(native_rank, reranked_rank),
                "native_top5": native_rank <= 5,
                "native_top10": native_rank <= 10,
                "reranked_top5": reranked_rank <= 5,
                "reranked_top10": reranked_rank <= 10,
                "transition_at_5": _transition(native_rank, reranked_rank, 5),
                "transition_at_10": _transition(native_rank, reranked_rank, 10),
                "native_top5_ids": [candidate["candidate_id"] for candidate in row["candidates"][:5]],
                "reranked_top5_ids": ranked[:5],
                "native_top10_ids": [candidate["candidate_id"] for candidate in row["candidates"][:10]],
                "reranked_top10_ids": ranked[:10],
                "reranked_ids": ranked,
            }
        )
    metrics = _metrics(reranked_ranks)
    native_metrics = _metrics(native_ranks)
    transitions = {
        str(cutoff): {
            "rescues": sum(
                row["native_rank"] > cutoff and row["reranked_rank"] <= cutoff
                for row in trace_rows
            ),
            "damages": sum(
                row["native_rank"] <= cutoff and row["reranked_rank"] > cutoff
                for row in trace_rows
            ),
        }
        for cutoff in (5, 10)
    }
    for values in transitions.values():
        values["net_rescue"] = values["rescues"] - values["damages"]
    rank_deltas = [row["rank_delta_native_minus_reranked"] for row in trace_rows]
    metrics.update(
        {
            "mean_gold_rank_delta": round(statistics.mean(rank_deltas), 6),
            "median_gold_rank_delta": round(statistics.median(rank_deltas), 6),
            "mean_native_top5_overlap": round(
                statistics.mean(
                    len(set(row["native_top5_ids"]) & set(row["reranked_top5_ids"])) / 5
                    for row in trace_rows
                ),
                6,
            ),
            "mean_native_top10_overlap": round(
                statistics.mean(
                    len(set(row["native_top10_ids"]) & set(row["reranked_top10_ids"])) / 10
                    for row in trace_rows
                ),
                6,
            ),
        }
    )
    return {
        "native_metrics": native_metrics,
        "reranked_metrics": metrics,
        "deltas": {
            key: round(metrics[key] - native_metrics[key], 6)
            for key in ("recall_at_5", "recall_at_10", "recall_at_50", "mrr_at_50", "ndcg_at_10")
        },
            "transitions": transitions,
            "subgroups": {
                "fold": _subgroup_metrics(trace_rows, "fold"),
                "band": _subgroup_metrics(trace_rows, "band"),
            },
            "trace_rows": trace_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    substrate = json.loads(Path(args.substrate).read_text(encoding="utf-8"))
    FREEZE.validate_frozen(substrate)
    model_id = QWEN_MODEL_ID if args.model == "qwen" else JINA_MODEL_ID
    model_path, revision = _resolve_snapshot(Path(args.model_cache), model_id)
    started = time.perf_counter()
    model = _load_qwen_reranker(model_path) if args.model == "qwen" else _load_jina_reranker(model_path)
    condition_results: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    selected_conditions = tuple(args.conditions) if args.conditions else CONDITIONS
    for condition in selected_conditions:
        rows = substrate["conditions"][condition]["queries"]
        if args.model == "qwen":
            rankings = _qwen_rank(model, rows, args.batch_size)
        else:
            rankings, usage[condition] = _jina_rank(model, rows)
        condition_results[condition] = _evaluate(rows, rankings)
        print(f"completed condition={condition}", flush=True)
    return {
        "schema_version": "qwen-top50-frozen-reranker-v1",
        "status": "completed",
        "substrate": str(Path(args.substrate)),
        "substrate_sha256": __import__("hashlib").sha256(Path(args.substrate).read_bytes()).hexdigest(),
        "retriever": substrate["retriever"],
        "reranker": {
            "name": args.model,
            "model_id": model_id,
            "snapshot_revision": revision,
            "resolved_model": model_path,
            "backend": "mlx",
            "batch_size": args.batch_size if args.model == "qwen" else None,
            "listwise": args.model == "jina",
        },
        "runtime": {
            "python": sys.executable,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "candidate_count_per_query": POOL_SIZE,
        },
        "conditions": condition_results,
        "listwise_usage": usage if args.model == "jina" else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", default=str(DEFAULT_SUBSTRATE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("qwen", "jina"), required=True)
    parser.add_argument("--model-cache", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=None,
        help="conditions to rerank; defaults to all frozen conditions",
    )
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_seconds": result["runtime"]["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
