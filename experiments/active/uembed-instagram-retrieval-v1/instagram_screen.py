"""Stage 1: native UEmbed channel validation over the exact zembed screen scope.

Builds cases with the zembed screen's own builder (frozen 60-anchor episode
artifact, self-authored documents, other-speaker-neighbourhood queries),
encodes everything once through UEmbedBackend, and evaluates:

    uembed_dense   cosine over last.normal
    uembed_sparse  SPLADE dot product
    uembed_native  RRF(dense_rank, sparse_rank), k=60

qwen_baseline metrics are carried from the recorded zembed screen artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"), str(EXPERIMENT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ranking import grouped_metrics, metrics, rrf
from uembed_backend import UEmbedBackend

ZEMBED_SCREEN_MODULE = (
    PROJECT_ROOT / "experiments/active/zembed-retrieval-validity-v1/instagram_screen.py"
)
GATED_RESULTS = (
    PROJECT_ROOT
    / "data/generated/instagram-conversation-episodes-v2-gated-qwen-rerun/gated-results.json"
)
MESSAGES = PROJECT_ROOT / "data/normalized/messages.json"
RECORDED_SCREEN = (
    PROJECT_ROOT / "data/generated/zembed-instagram-screen-v1/instagram-screen.json"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def build_cases(limit: int | None):
    screen = load_module(ZEMBED_SCREEN_MODULE, "zembed_ig_screen")
    episode_result = json.loads(GATED_RESULTS.read_text())
    messages = json.loads(MESSAGES.read_text())
    conditions, queries, documents = screen._build_cases(episode_result, messages)
    anchor_ids = sorted(queries)
    if limit:
        anchor_ids = anchor_ids[:limit]
    keep = set(anchor_ids)
    queries = {aid: queries[aid] for aid in anchor_ids}
    documents = {
        key: text for key, text in documents.items() if key.split(":", 1)[1] in keep
    }
    rows = {
        condition: {aid: rows[aid] for aid in anchor_ids}
        for condition, rows in conditions.items()
    }
    return rows, queries, documents


def rank_all(rows, queries, documents, query_vecs, doc_vecs):
    import torch

    per_anchor_channels: dict[str, dict[str, Any]] = {}
    policies = {"uembed_dense": {}, "uembed_sparse": {}, "uembed_native_rrf": {}}
    document_keys = sorted(
        key for key in documents if key.startswith("baseline:")
    )
    dense_matrix = torch.stack([torch.as_tensor(doc_vecs[key]["dense"]) for key in document_keys]).float()
    sparse_matrix = torch.stack([torch.as_tensor(doc_vecs[key]["sparse"]) for key in document_keys]).float()
    doc_anchor_by_row = [key.split(":", 1)[1] for key in document_keys]
    for anchor_id in sorted(queries):
        candidate_ids = sorted(
            key.split(":", 1)[1] for key in documents if key.startswith("baseline:")
        )
        q_dense = torch.as_tensor(query_vecs[anchor_id]["dense"]).float()
        q_sparse = torch.as_tensor(query_vecs[anchor_id]["sparse"]).float()
        dense_vector = dense_matrix @ q_dense
        sparse_vector = sparse_matrix @ q_sparse
        dense_lookup = {
            anchor: float(dense_vector[row])
            for row, anchor in enumerate(doc_anchor_by_row)
        }
        sparse_lookup = {
            anchor: float(sparse_vector[row])
            for row, anchor in enumerate(doc_anchor_by_row)
        }
        dense_scores = {cid: dense_lookup[cid] for cid in candidate_ids}
        sparse_scores = {cid: sparse_lookup[cid] for cid in candidate_ids}
        dense_rank = sorted(candidate_ids, key=lambda c: (-dense_scores[c], c))
        sparse_rank = sorted(candidate_ids, key=lambda c: (-sparse_scores[c], c))
        native_rank = rrf([dense_rank, sparse_rank])
        policies["uembed_dense"][anchor_id] = dense_rank
        policies["uembed_sparse"][anchor_id] = sparse_rank
        policies["uembed_native_rrf"][anchor_id] = native_rank
        per_anchor_channels[anchor_id] = {
            "fold": rows["baseline"][anchor_id]["fold"],
            "band": rows["baseline"][anchor_id]["band"],
            "candidate_ids": candidate_ids,
            "dense_scores": dense_scores,
            "sparse_scores": sparse_scores,
        }

    output: dict[str, Any] = {}
    for policy, mapping in policies.items():
        result_rows = []
        ranks = []
        for anchor_id in sorted(mapping):
            ranked = mapping[anchor_id]
            rank = ranked.index(anchor_id) + 1
            ranks.append(rank)
            base = per_anchor_channels[anchor_id]
            result_rows.append(
                {
                    "anchor_message_id": anchor_id,
                    "fold": base["fold"],
                    "band": base["band"],
                    "rank": rank,
                }
            )
        aggregate = metrics(ranks)
        output[policy] = {
            "aggregate": aggregate,
            "rows": result_rows,
            "by_fold": grouped_metrics(result_rows, "fold"),
            "by_band": grouped_metrics(result_rows, "band"),
        }
    output["_channels"] = per_anchor_channels
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None, help="sanity slice size")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    rows, queries, documents = build_cases(args.limit)
    backend = UEmbedBackend(device=args.device)
    card_check = backend.model_card_check()

    started = time.perf_counter()
    sorted_anchors = sorted(queries)
    query_vec_list = backend.encode_queries([queries[a] for a in sorted_anchors])
    query_vecs = {
        aid: {"dense": d, "sparse": sp}
        for aid, d, sp in zip(sorted_anchors, query_vec_list["dense"], query_vec_list["sparse"])
    }
    document_keys = sorted(documents)
    doc_vec_list = backend.encode_documents([documents[k] for k in document_keys])
    doc_vecs = {
        key: {"dense": d, "sparse": sp}
        for key, d, sp in zip(document_keys, doc_vec_list["dense"], doc_vec_list["sparse"])
    }
    encode_seconds = round(time.perf_counter() - started, 1)

    scored = rank_all(rows, queries, documents, query_vecs, doc_vecs)
    channels = scored.pop("_channels")

    # determinism: identical ranks on an immediate rescoring pass
    rescored = rank_all(rows, queries, documents, query_vecs, doc_vecs)
    deterministic = True
    for policy in ("uembed_dense", "uembed_sparse", "uembed_native_rrf"):
        first_rows = {r["anchor_message_id"]: r["rank"] for r in scored[policy]["rows"]}
        second_rows = {r["anchor_message_id"]: r["rank"] for r in rescored[policy]["rows"]}
        deterministic &= first_rows == second_rows

    recorded = json.loads(RECORDED_SCREEN.read_text())
    qwen_baseline = {}
    for condition, block in recorded["conditions"].items():
        if isinstance(block, dict) and "qwen" in block:
            qwen_baseline[condition] = {
                k: v
                for k, v in block["qwen"].items()
                if k in ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_50", "query_count")
            }

    report = {
        "schema": "ablation-run-v1",
        "experiment": "uembed-instagram-retrieval-v1/stage1",
        "stage": 1,
        "scope": {"anchors": len(queries), "documents": len(documents), "slice_limit": args.limit},
        "model_card_check": card_check,
        "encode_seconds": encode_seconds,
        "repeat_determinism": bool(deterministic),
        "policies": scored,
        "qwen_baseline_recorded": qwen_baseline,
        "channels": channels,
        "runs": {
            f"uembed_stage1{args.suffix}": {
                "accepted": None,
                "word_count": 0,
                "tokens_prompt": 0,
                "tokens_output": 0,
                "claim_verifier_failures": 0,
                "qualifier_failures": 0,
                "evidence_coverage": None,
                "holdout_status": "not_run",
                "repeat_determinism": bool(deterministic),
                "call_count": len(queries) + len(documents),
                "repair_count": 0,
                "failure_class_totals": {},
                "qwen_only": False,
                "jina_used": False,
                "production_index_changed": False,
                "generator_model": "Alibaba-NLP/UEmbed-2B (embedding screen)",
                "max_token_budget": 8192,
            }
        },
    }
    out_dir = EXPERIMENT_DIR / "artifacts" / "stage-1"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"screen{args.suffix}.json"
    (out_dir / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({p: scored[p]["aggregate"] for p in ("uembed_dense", "uembed_sparse", "uembed_native_rrf")}, indent=1))
    print("deterministic:", deterministic, "| encoded in", encode_seconds, "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
