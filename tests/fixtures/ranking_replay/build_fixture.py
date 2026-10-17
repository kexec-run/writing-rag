"""Materialize deterministic synthetic top-50 and ranking JSON fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CONDITIONS = ("baseline", "c1_full", "jina_selected")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pool(query: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    query_id = query["query_id"]
    gold_id = f"{query_id}-gold"
    candidate_ids = [f"{query_id}-candidate-{index:02d}" for index in range(50)]
    native_rank = query["native_rank"]
    if native_rank <= 50:
        native_ids = candidate_ids.copy()
        native_ids.insert(native_rank - 1, gold_id)
        native_ids = native_ids[:50]
    else:
        native_ids = candidate_ids.copy()
    candidates = []
    for rank, candidate_id in enumerate(native_ids, 1):
        text = f"Synthetic passage {rank:02d} belonging to {query_id}."
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_text": text,
                "candidate_sha256": _hash(text),
                "native_score": round(1.0 - rank / 100.0, 8),
                "native_rank": rank,
            }
        )
    if native_rank <= 50:
        gold_text = f"Synthetic relevant passage for {query_id}."
        candidates[native_rank - 1]["candidate_id"] = gold_id
        candidates[native_rank - 1]["candidate_text"] = gold_text
        candidates[native_rank - 1]["candidate_sha256"] = _hash(gold_text)
    row = {
        "query_id": query_id,
        "query_text": f"Synthetic query {query_id}.",
        "query_sha256": _hash(f"Synthetic query {query_id}."),
        "fold": query["fold"],
        "band": query["band"],
        "gold_candidate_id": gold_id,
        "gold_native_rank": native_rank,
        "native_top5": native_rank <= 5,
        "native_top10": native_rank <= 10,
        "candidates": candidates,
    }
    return row, native_ids


def _ranking_order(candidate_ids: list[str], gold_id: str, rank: int) -> list[str]:
    ordered = [candidate_id for candidate_id in candidate_ids if candidate_id != gold_id]
    if rank <= 50:
        ordered.insert(rank - 1, gold_id)
        return ordered[:50]
    return ordered


def build(spec_path: Path, output_dir: Path) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to write into a nonempty directory: {output_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    queries = spec.get("queries")
    if spec.get("schema_version") != "ranking-replay-fixture-spec-v1" or not isinstance(queries, list) or len(queries) != 6:
        raise ValueError("fixture specification must declare exactly six synthetic queries")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_rows: dict[str, dict[str, Any]] = {}
    native_orders: dict[str, list[str]] = {}
    for query in queries:
        row, native_orders[query["query_id"]] = _pool(query)
        frozen_rows[row["query_id"]] = row

    substrate = {
        "schema_version": "qwen-top50-frozen-substrate-v1",
        "status": "frozen",
        "scope": {"query_count": len(queries), "pool_size": 50, "condition_order": list(CONDITIONS)},
        "retriever": {"embedding_backend": "not_executed", "query_policy": "synthetic_fixture_only"},
        "invariants": {"candidate_text_frozen": True, "query_text_frozen": True, "gold_labels_frozen": True, "rerankers_may_only_change_order": True, "jina_is_not_retriever": True},
        "conditions": {
            condition: {"query_count": len(queries), "pool_size": 50, "queries": [frozen_rows[query["query_id"]] for query in queries]}
            for condition in CONDITIONS
        },
    }
    substrate_path = output_dir / "frozen-top50.json"
    _write(substrate_path, substrate)
    substrate_hash = hashlib.sha256(substrate_path.read_bytes()).hexdigest()
    outputs = {"substrate": substrate_path}
    for label_key, label, file_name in (
        ("qwen_rank", "Synthetic Qwen-order", "qwen-rankings.json"),
        ("jina_rank", "Synthetic Jina-order", "jina-rankings.json"),
    ):
        ranking_artifact = {
            "schema_version": "supplied-ranking-fixture-v1",
            "execution_mode": "synthetic_replay",
            "ranking_origin": "supplied_fixture",
            "label": label,
            "query_count": len(queries),
            "substrate_sha256": substrate_hash,
            "conditions": {
                condition: {
                    "query_count": len(queries),
                    "queries": [
                        {
                            "query_id": query["query_id"],
                            "reranked_ids": _ranking_order(
                                native_orders[query["query_id"]],
                                frozen_rows[query["query_id"]]["gold_candidate_id"],
                                query[label_key],
                            ),
                        }
                        for query in queries
                    ]
                }
                for condition in CONDITIONS
            },
        }
        path = output_dir / file_name
        _write(path, ranking_artifact)
        outputs[label_key] = path
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        outputs = build(Path(args.spec), Path(args.output))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
