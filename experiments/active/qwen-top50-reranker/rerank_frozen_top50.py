"""Rerank frozen Qwen top-50 pools for the B0/C1/J1 episode conditions.

This diagnostic never calls Jina and never changes the archived chain-selection
decision. Recall@50 is fixed by the native Qwen pool; the reranker can only
change the ordering inside that pool.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data/generated/instagram-jina-candidate-selection-v1/jina-selection-results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/qwen-top50-reranker-v1"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models/weights/huggingface/hub"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MLX_MODEL_ID = "mku64/Qwen3-Reranker-0.6B-mlx-8Bit"
CONDITIONS = ("baseline", "c1_full", "jina_selected")
POOL_SIZE = 50


def _retrieval_text(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        str(by_id[message_id].get("content", ""))
        for message_id in row["primary"].get("self_message_ids", [])
    ).strip()


def _round(value: float) -> float:
    return round(value, 6)


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _metrics(ranks: list[int], *, pool_size: int = POOL_SIZE) -> dict[str, Any]:
    count = len(ranks)
    return {
        "query_count": count,
        "mrr_pool": _round(
            sum(1.0 / rank if rank <= pool_size else 0.0 for rank in ranks) / count
            if count
            else 0.0
        ),
        "recall_at_10": _round(sum(rank <= 10 for rank in ranks) / count if count else 0.0),
        "ndcg_at_10": _round(
            sum(1.0 / __import__("math").log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks)
            / count
            if count
            else 0.0
        ),
        "recall_at_50": _round(sum(rank <= pool_size for rank in ranks) / count if count else 0.0),
    }


def _group_metrics(rows: list[dict[str, Any]], key: str, rank_key: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(int(row[rank_key]))
    return {group: _metrics(ranks) for group, ranks in sorted(groups.items())}


def _validate_input(result: dict[str, Any]) -> None:
    qwen = result.get("qwen_retrieval", {})
    if result.get("structural", {}).get("passed") is not True:
        raise ValueError("frozen structural checks did not pass")
    if qwen.get("backend", {}).get("embedding_backend") != "qwen3":
        raise ValueError("frozen ranking artifact is not Qwen retrieval")
    if qwen.get("query_count") != 60:
        raise ValueError("expected exactly 60 queries")
    if set(qwen.get("conditions", {})) != set(CONDITIONS):
        raise ValueError("expected exactly B0, C1, and J1 conditions")
    for condition in CONDITIONS:
        rows = qwen["conditions"][condition].get("rows", [])
        if len(rows) != 60:
            raise ValueError(f"{condition} does not contain 60 native rankings")


def _resolve_model(model_cache: str, repo_id: str) -> tuple[str, str]:
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=model_cache,
        local_files_only=True,
    )
    revision = Path(path).name
    return path, revision


class MLXReranker:
    def __init__(self, model_path: str, *, max_length: int) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model, tokenizer = load(model_path)
        self.tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        self.max_length = max_length
        self.instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            "based on the Query and the Instruct provided. Note that the answer can "
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

    def _pair_ids(self, query: str, document: str) -> list[int]:
        content = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)
        available = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        if available <= 0:
            raise ValueError("max_length is smaller than the MLX reranker prompt")
        return self.prefix_ids + content_ids[:available] + self.suffix_ids

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        del show_progress_bar, convert_to_numpy
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            encoded = [
                self._pair_ids(query, document)
                for query, document in pairs[start : start + batch_size]
            ]
            max_len = max(len(ids) for ids in encoded)
            padded = [
                ids + [self.pad_id] * (max_len - len(ids))
                for ids in encoded
            ]
            tokens = self.mx.array(padded)
            logits = self.model(tokens)
            self.mx.eval(logits)
            positions = self.mx.array([len(ids) - 1 for ids in encoded])
            last_logits = logits[self.mx.arange(len(encoded)), positions, :]
            pair_logits = self.mx.stack(
                [last_logits[:, self.no_id], last_logits[:, self.yes_id]],
                axis=1,
            )
            batch_scores = self.mx.softmax(pair_logits, axis=1)[:, 1]
            self.mx.eval(batch_scores)
            scores.extend(float(score) for score in batch_scores.tolist())
        return scores

def run(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    _validate_input(result)
    messages = json.loads(Path(result["scope"]["messages_path"]).read_text(encoding="utf-8"))
    by_id = {str(message["message_id"]): message for message in messages}
    queries = result["queries"]
    native_conditions = result["qwen_retrieval"]["conditions"]
    source_rows = result["rows"]
    document_texts = {
        condition: {
            str(row["anchor_message_id"]): _retrieval_text(row, by_id)
            for row in source_rows[condition]
        }
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        if not all(document_texts[condition].values()):
            raise ValueError(f"{condition} contains an empty Qwen retrieval document")


    # Reproduce the per-query Qwen ranking so the reranker sees a real fixed
    # top-50 pool. The archived artifact stores only top-10 IDs, not top-50 IDs.
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from embed_chunks import create_backend

    qwen_cache = Path(args.model_cache)
    if qwen_cache.name == "hub":
        qwen_cache = qwen_cache.parent
    qwen_backend, qwen_metadata = create_backend(
        "qwen3",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device=args.qwen_embedding_device,
        model_cache=qwen_cache,
    )
    query_ids = sorted(queries)
    query_vectors = qwen_backend.encode_queries(
        [queries[anchor_id] for anchor_id in query_ids],
        batch_size=args.qwen_embedding_batch_size,
        policy="qwen_card_query",
    )
    unique_doc_ids = sorted(
        f"{condition}:{candidate_id}"
        for condition in CONDITIONS
        for candidate_id in document_texts[condition]
    )
    document_vectors = qwen_backend.encode_documents(
        [
            document_texts[condition][candidate_id]
            for condition, candidate_id in (
                item.split(":", 1) for item in unique_doc_ids
            )
        ],
        batch_size=args.qwen_embedding_batch_size,
    )
    vector_by_id = dict(zip(unique_doc_ids, document_vectors))
    qwen_rankings: dict[str, dict[str, list[str]]] = {}
    frozen_rank_matches: dict[str, int] = {}
    frozen_top10_matches: dict[str, int] = {}
    for condition in CONDITIONS:
        candidate_ids = sorted(document_texts[condition])
        native_rows = {
            str(row["anchor_message_id"]): row
            for row in native_conditions[condition]["rows"]
        }
        rankings: dict[str, list[str]] = {}
        matches = 0
        top10_matches = 0
        for query_index, anchor_id in enumerate(query_ids):
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
            rankings[anchor_id] = ranked
            if ranked.index(anchor_id) + 1 == int(native_rows[anchor_id]["rank"]):
                matches += 1
            if ranked[:10] == native_rows[anchor_id].get("top10_anchor_message_ids", []):
                top10_matches += 1
        qwen_rankings[condition] = rankings
        frozen_rank_matches[condition] = matches
        frozen_top10_matches[condition] = top10_matches

    if args.backend == "mlx":
        model_id = MLX_MODEL_ID
        model_path, model_revision = _resolve_model(
            args.model_cache,
            model_id,
        )
        model = MLXReranker(model_path, max_length=args.max_length)
        instruction = model.instruction
    else:
        model_id = MODEL_ID
        model_path, model_revision = _resolve_model(
            args.model_cache,
            model_id,
        )
        model = CrossEncoder(
            model_path,
            device=args.device,
            max_length=args.max_length,
        )
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    condition_results: dict[str, Any] = {}
    total_pairs = 0
    started = time.perf_counter()
    cpu_info = {
        "available_logical_cpus": os.cpu_count(),
        "configured_intraop_threads": None,
        "configured_interop_threads": None,
    }
    if args.device == "cpu":
        import torch

        configured_threads = args.cpu_threads or os.cpu_count() or 1
        torch.set_num_threads(configured_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, configured_threads)))
        except RuntimeError:
            pass
        cpu_info["configured_intraop_threads"] = torch.get_num_threads()
        cpu_info["configured_interop_threads"] = torch.get_num_interop_threads()
    print(
        f"scoring {len(CONDITIONS)} conditions x {len(queries)} queries "
        f"x {POOL_SIZE} candidates; device={args.device}; "
        f"cpu_threads={cpu_info['configured_intraop_threads']}",
        flush=True,
    )

    for condition in CONDITIONS:
        native_rows = {
            str(row["anchor_message_id"]): row
            for row in native_conditions[condition]["rows"]
        }
        pair_keys: list[tuple[str, str]] = []
        pairs: list[tuple[str, str]] = []
        for anchor_id in query_ids:
            pool_ids = qwen_rankings[condition][anchor_id][:POOL_SIZE]
            pairs.extend(
                (queries[anchor_id], document_texts[condition][candidate_id])
                for candidate_id in pool_ids
            )
            pair_keys.extend((anchor_id, candidate_id) for candidate_id in pool_ids)
        print(
            f"prepared condition={condition} pairs={len(pairs)}",
            flush=True,
        )

        scores = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        total_pairs += len(pairs)
        print(f"scored condition={condition}", flush=True)
        scores_by_anchor: dict[str, list[tuple[str, float]]] = {
            anchor_id: [] for anchor_id in sorted(queries)
        }
        for (anchor_id, candidate_id), score in zip(pair_keys, scores):
            scores_by_anchor[anchor_id].append((candidate_id, float(score)))

        trace_rows: list[dict[str, Any]] = []
        for anchor_id in sorted(queries):
            ranked = sorted(
                scores_by_anchor[anchor_id],
                key=lambda item: (-item[1], item[0]),
            )
            reranked_ids = [candidate_id for candidate_id, _ in ranked]
            native_rank = qwen_rankings[condition][anchor_id].index(anchor_id) + 1
            reranked_rank = (
                reranked_ids.index(anchor_id) + 1
                if anchor_id in reranked_ids
                else POOL_SIZE + 1
            )
            trace_rows.append(
                {
                    "anchor_message_id": anchor_id,
                    "fold": native_rows[anchor_id]["fold"],
                    "band": native_rows[anchor_id]["band"],
                    "native_rank": native_rank,
                    "frozen_native_rank": int(native_rows[anchor_id]["rank"]),
                    "in_native_top50": native_rank <= POOL_SIZE,
                    "reranked_rank": reranked_rank,
                    "reranker_score": next(
                        score
                        for candidate_id, score in ranked
                        if candidate_id == anchor_id
                    )
                    if anchor_id in reranked_ids
                    else None,
                    "reranked_top10_anchor_message_ids": reranked_ids[:10],
                }
            )
        native_pool_ranks = [
            row["native_rank"] if row["native_rank"] <= POOL_SIZE else POOL_SIZE + 1
            for row in trace_rows
        ]
        reranked_ranks = [int(row["reranked_rank"]) for row in trace_rows]
        condition_results[condition] = {
            "native_frozen": native_conditions[condition],
            "native_top50_pool": _metrics(native_pool_ranks),
            "reranked_top50_pool": _metrics(reranked_ranks),
            "native_top50_pool_by_fold": _group_metrics(trace_rows, "fold", "native_rank"),
            "reranked_top50_pool_by_fold": _group_metrics(trace_rows, "fold", "reranked_rank"),
            "native_top50_pool_by_band": _group_metrics(trace_rows, "band", "native_rank"),
            "reranked_top50_pool_by_band": _group_metrics(trace_rows, "band", "reranked_rank"),
            "rows": trace_rows,
        }

    elapsed = time.perf_counter() - started
    for condition in CONDITIONS:
        native = condition_results[condition]["native_top50_pool"]
        reranked = condition_results[condition]["reranked_top50_pool"]
        condition_results[condition]["delta_reranked_vs_native_pool"] = {
            field: _round(reranked[field] - native[field])
            for field in ("mrr_pool", "recall_at_10", "ndcg_at_10", "recall_at_50")
        }
    return {
        "schema_version": "qwen-top50-reranker-diagnostic-v1",
        "status": "post_hoc_top50_reranking_diagnostic",
        "jina_used": False,
        "input_artifact": str(Path(args.input)),
        "candidate_pool": {
            "retriever": "Qwen/Qwen3-Embedding-0.6B",
            "native_pool_size": POOL_SIZE,
            "recall_at_50_fixed_by_construction": True,
            "oracle_ceiling_definition": "native Qwen relevant-anchor rank <= 50",
        },

        "qwen_retrieval_reproduction": {
            "backend": qwen_metadata,
            "query_policy": "qwen_card_query",
            "per_query_pool": True,
            "frozen_rank_matches": frozen_rank_matches,
            "frozen_top10_matches": frozen_top10_matches,
            "frozen_rank_total": len(query_ids),
            "frozen_top50_ids_available": False,
            "rank_drift_recorded_not_gated": True,
        },
        "reranker": {
            "backend": args.backend,
            "model": model_id,
            "resolved_model": model_path,
            "snapshot_revision": model_revision,
            "instruction": instruction,
            "device": "mlx-gpu" if args.backend == "mlx" else str(args.device),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "fallback_used": False,
        },
        "runtime": {
            "python": __import__("sys").executable,
            "pair_count": total_pairs,
            "elapsed_seconds": _round(elapsed),
            "cpu_info": cpu_info,
        },
        "conditions": condition_results,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen top-50 reranker diagnostic",
        "",
        "Jina was not used. Each condition uses native Qwen top-50 candidates, and the reranker only changes their order.",
        "Recall@50 is therefore fixed by construction and is the oracle ceiling for that condition.",
        "Native Qwen rankings were reproduced with the current Qwen implementation; the frozen input stored top-10 IDs but not top-50 IDs. Frozen rank/top-10 drift is recorded in the machine-readable output.",
        "",
        "`mrr_pool` is reciprocal rank inside the fixed top-50 pool; a relevant item outside that pool contributes zero.",
        "",
        "| Condition | Native MRR@50 pool | Reranked MRR@50 pool | Native Recall@10 | Reranked Recall@10 | Native nDCG@10 | Reranked nDCG@10 | Oracle Recall@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = result["conditions"][condition]
        native = values["native_top50_pool"]
        reranked = values["reranked_top50_pool"]
        lines.append(
            f"| {condition} | {native['mrr_pool']:.6f} | {reranked['mrr_pool']:.6f} | "
            f"{native['recall_at_10']:.6f} | {reranked['recall_at_10']:.6f} | "
            f"{native['ndcg_at_10']:.6f} | {reranked['ndcg_at_10']:.6f} | "
            f"{native['recall_at_50']:.6f} |"
        )
    lines.extend(["", "## Deltas from reranking", ""])
    lines.extend(
        f"- `{condition}`: "
        f"MRR `{values['delta_reranked_vs_native_pool']['mrr_pool']:+.6f}`, "
        f"Recall@10 `{values['delta_reranked_vs_native_pool']['recall_at_10']:+.6f}`, "
        f"nDCG@10 `{values['delta_reranked_vs_native_pool']['ndcg_at_10']:+.6f}`, "
        f"Recall@50 `{values['delta_reranked_vs_native_pool']['recall_at_50']:+.6f}`"
        for condition in CONDITIONS
        for values in [result["conditions"][condition]]
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The oracle Recall@50 is not a reranker result; it is the maximum recall available after Qwen retrieval.",
            "This diagnostic does not alter the archived Jina chain-selection decision or promote a production reranker.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-cache", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--backend", choices=("mlx", "pytorch"), default="mlx")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--qwen-embedding-device",
        default="cpu",
        help="Device used to reproduce the frozen Qwen retrieval ranks",
    )
    parser.add_argument(
        "--qwen-embedding-batch-size",
        type=int,
        default=8,
        help="Batch size used to reproduce the frozen Qwen retrieval ranks",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU intra-op threads; 0 uses all logical CPUs",
    )
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reranker-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
