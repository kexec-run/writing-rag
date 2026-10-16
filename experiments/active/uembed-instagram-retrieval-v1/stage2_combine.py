"""Stage 2: combine production-Qwen channel with UEmbed dense/sparse (RRF k=60).

Baseline channel = recorded zembed-screen Qwen rows (identical universe and
query policy): top-10 exact, tail in lexicographic order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from ranking import grouped_metrics, metrics, rrf

PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
STAGE1 = EXPERIMENT_DIR / "artifacts/stage-1/screen.json"
RECORDED = PROJECT_ROOT / "data/generated/zembed-instagram-screen-v1/instagram-screen.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    stage1 = json.loads(STAGE1.read_text())
    channels = stage1["channels"]
    anchor_ids = sorted(channels)
    recorded = json.loads(RECORDED.read_text())
    qwen_rows = {
        row["anchor_message_id"]: row
        for row in recorded["conditions"]["baseline"]["qwen"]["rows"]
    }

    def rank_channel(policy: str) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for row in stage1["policies"][policy]["rows"]:
            aid = row["anchor_message_id"]
            ch = channels[aid]
            cids = ch["candidate_ids"]
            if policy == "uembed_dense":
                mapping[aid] = sorted(cids, key=lambda c: (-ch["dense_scores"][c], c))
            elif policy == "uembed_sparse":
                mapping[aid] = sorted(cids, key=lambda c: (-ch["sparse_scores"][c], c))
            else:
                d = sorted(cids, key=lambda c: (-ch["dense_scores"][c], c))
                sp = sorted(cids, key=lambda c: (-ch["sparse_scores"][c], c))
                from ranking import rrf

                mapping[aid] = rrf([d, sp])
        return mapping

    policies: dict[str, dict[str, list[str]]] = {
        "uembed_dense": rank_channel("uembed_dense"),
        "uembed_sparse": rank_channel("uembed_sparse"),
        "uembed_native_rrf": rank_channel("uembed_native_rrf"),
    }

    qwen_rank_full: dict[str, list[str]] = {}
    for aid in anchor_ids:
        top10 = list(qwen_rows[aid]["top10_anchor_message_ids"])
        rest = [c for c in sorted(channels[aid]["candidate_ids"]) if c not in top10]
        qwen_rank_full[aid] = top10 + rest
    policies["qwen_uembed_rrf"] = {
        aid: rrf(
            [
                qwen_rank_full[aid],
                policies["uembed_dense"][aid],
                policies["uembed_sparse"][aid],
            ]
        )
        for aid in anchor_ids
    }

    summary: dict[str, Any] = {}
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for name, mapping in policies.items():
        rows = []
        for aid in anchor_ids:
            rows.append(
                {
                    "anchor_message_id": aid,
                    "fold": channels[aid]["fold"],
                    "band": channels[aid]["band"],
                    "rank": mapping[aid].index(aid) + 1,
                }
            )
        rows_by_policy[name] = rows
        summary[name] = {
            "aggregate": metrics([r["rank"] for r in rows]),
            "by_fold": grouped_metrics(rows, "fold"),
            "by_band": grouped_metrics(rows, "band"),
        }

    def rank_of(mapping: dict[str, list[str]], aid: str) -> int:
        return mapping[aid].index(aid) + 1

    rescues = sum(
        rank_of(policies["qwen_uembed_rrf"], a) < rank_of(qwen_rank_full, a)
        for a in anchor_ids
    )
    damages = sum(
        rank_of(policies["qwen_uembed_rrf"], a) > rank_of(qwen_rank_full, a)
        for a in anchor_ids
    )
    fold_deltas: dict[str, float] = {}
    folds = {channels[a]["fold"] for a in anchor_ids}
    for fold in folds:
        ids = [a for a in anchor_ids if channels[a]["fold"] == fold]
        q = metrics([rank_of(qwen_rank_full, a) for a in ids])["mrr"]
        n = metrics([rank_of(policies["qwen_uembed_rrf"], a) for a in ids])["mrr"]
        fold_deltas[str(fold)] = round(float(n) - float(q), 4)

    report = {
        "schema": "ablation-run-v1",
        "experiment": "uembed-instagram-retrieval-v1/stage2",
        "stage": 2,
        "baseline_source": (
            "recorded zembed screen qwen rows (identical universe/policy); "
            "top-10 exact, tail lexicographic"
        ),
        "policies": summary,
        "failure_audit": {
            "rescues_vs_qwen": rescues,
            "damages_vs_qwen": damages,
            "net": rescues - damages,
            "fold_mrr_deltas_qwen_uembed": fold_deltas,
            "dense_vs_sparse_top1_disagreements": sum(
                policies["uembed_dense"][a][0] != policies["uembed_sparse"][a][0]
                for a in anchor_ids
            ),
        },
        "audit_rows": [
            {
                "anchor_message_id": aid,
                "fold": channels[aid]["fold"],
                "band": channels[aid]["band"],
                "rank_qwen": rank_of(qwen_rank_full, aid),
                "rank_uembed_dense": rank_of(policies["uembed_dense"], aid),
                "rank_uembed_sparse": rank_of(policies["uembed_sparse"], aid),
                "rank_qwen_uembed": rank_of(policies["qwen_uembed_rrf"], aid),
            }
            for aid in anchor_ids
        ],
        "runs": {
            "uembed_stage2_combine": {
                "accepted": None,
                "word_count": 0,
                "tokens_prompt": 0,
                "tokens_output": 0,
                "claim_verifier_failures": 0,
                "qualifier_failures": 0,
                "evidence_coverage": None,
                "holdout_status": "not_run",
                "repeat_determinism": True,
                "call_count": len(anchor_ids),
                "repair_count": 0,
                "failure_class_totals": {},
                "qwen_only": False,
                "jina_used": False,
                "production_index_changed": False,
                "generator_model": "Qwen3-Embedding-0.6B + Alibaba-NLP/UEmbed-2B",
                "max_token_budget": 8192,
            }
        },
    }
    out = EXPERIMENT_DIR / "artifacts/stage-2/combine.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({n: v["aggregate"] for n, v in summary.items()}, indent=1))
    print("audit:", json.dumps(report["failure_audit"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
