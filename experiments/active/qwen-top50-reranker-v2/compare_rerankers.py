"""Compare native Qwen, Qwen MLX, and Jina MLX frozen-pool results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay_frozen_rankings import load_validated_result

CONDITIONS = ("baseline", "c1_full", "jina_selected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--jina", required=True)
    parser.add_argument("--substrate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    substrate_path = Path(args.substrate) if args.substrate else None
    qwen = load_validated_result(Path(args.qwen), substrate_path)
    jina = load_validated_result(Path(args.jina), substrate_path)
    qwen_mode = qwen.get("execution_mode", "model_execution")
    jina_mode = jina.get("execution_mode", "model_execution")
    if qwen_mode != jina_mode:
        raise ValueError("cannot compare synthetic replay with model-execution results")
    if qwen["substrate_sha256"] != jina["substrate_sha256"]:
        raise ValueError("Qwen and Jina results do not use the same frozen substrate")
    rows = []
    for condition in CONDITIONS:
        qwen_values = qwen["conditions"][condition]
        jina_values = jina["conditions"][condition]
        rows.append(
            {
                "condition": condition,
                "native": qwen_values["native_metrics"],
                "qwen": qwen_values["reranked_metrics"],
                "jina": jina_values["reranked_metrics"],
                "qwen_transitions": qwen_values["transitions"],
                "jina_transitions": jina_values["transitions"],
                "qwen_subgroups": qwen_values["subgroups"],
                "jina_subgroups": jina_values["subgroups"],
            }
        )
    result = {
        "schema_version": "qwen-top50-frozen-reranker-comparison-v1",
        "qwen_artifact": str(Path(args.qwen)),
        "jina_artifact": str(Path(args.jina)),
        "same_substrate_required": True,
        "execution_mode": qwen_mode,
        "model_executed": qwen_mode != "synthetic_replay",
        "substrate_sha256": qwen["substrate_sha256"],
        "ranking_labels": {
            "qwen": qwen.get("ranking_label", qwen.get("reranker", {}).get("name", "Qwen MLX")),
            "jina": jina.get("ranking_label", jina.get("reranker", {}).get("name", "Jina MLX")),
        },
        "conditions": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    qwen_label = result["ranking_labels"]["qwen"]
    jina_label = result["ranking_labels"]["jina"]
    if qwen_mode == "synthetic_replay":
        description = "Synthetic ranking replay; no model inference was executed."
        qwen_label = f"{qwen_label} fixture"
        jina_label = f"{jina_label} fixture"
    else:
        description = "Qwen remains the retriever. Rerankers only reorder the same frozen 50 candidates per query."
    lines = [
        "# Frozen top-50 reranker comparison",
        "",
        description,
        "",
        f"| Condition | Native R@5 | {qwen_label} R@5 | {jina_label} R@5 | Native R@10 | {qwen_label} R@10 | {jina_label} R@10 | Native MRR | {qwen_label} MRR | {jina_label} MRR | Native nDCG@10 | {qwen_label} nDCG@10 | {jina_label} nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        native = row["native"]
        qwen_values = row["qwen"]
        jina_values = row["jina"]
        lines.append(
            f"| {row['condition']} | {native['recall_at_5']:.6f} | {qwen_values['recall_at_5']:.6f} | {jina_values['recall_at_5']:.6f} | "
            f"{native['recall_at_10']:.6f} | {qwen_values['recall_at_10']:.6f} | {jina_values['recall_at_10']:.6f} | "
            f"{native['mrr_at_50']:.6f} | {qwen_values['mrr_at_50']:.6f} | {jina_values['mrr_at_50']:.6f} | "
            f"{native['ndcg_at_10']:.6f} | {qwen_values['ndcg_at_10']:.6f} | {jina_values['ndcg_at_10']:.6f} |"
        )
    if qwen_mode == "synthetic_replay":
        lines.extend([
            "",
            "Synthetic fixture results are contract checks only; they are not model measurements and cannot change a model or production decision.",
        ])
    else:
        lines.extend(
        [
            "",
            "## Decision gates",
            "",
            "Promising requires Jina ΔR@10 ≥ +0.05, net rescue@10 ≥ +3, non-negative R@5, MRR, and nDCG@10, with no catastrophic fold or band regression.",
            "A result below those thresholds remains diagnostic and does not promote Jina or change the Qwen retrieval path.",
        ]
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(output), "report": str(output.with_suffix('.md'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
