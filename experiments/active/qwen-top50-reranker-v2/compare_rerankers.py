"""Compare native Qwen, Qwen MLX, and Jina MLX frozen-pool results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ("baseline", "c1_full", "jina_selected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--jina", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    qwen = json.loads(Path(args.qwen).read_text(encoding="utf-8"))
    jina = json.loads(Path(args.jina).read_text(encoding="utf-8"))
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
        "substrate_sha256": qwen["substrate_sha256"],
        "conditions": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Frozen top-50 reranker comparison",
        "",
        "Qwen remains the retriever. Qwen MLX and Jina MLX only reorder the same frozen 50 candidates per query.",
        "",
        "| Condition | Native R@5 | Qwen R@5 | Jina R@5 | Native R@10 | Qwen R@10 | Jina R@10 | Native MRR | Qwen MRR | Jina MRR | Native nDCG@10 | Qwen nDCG@10 | Jina nDCG@10 |",
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
