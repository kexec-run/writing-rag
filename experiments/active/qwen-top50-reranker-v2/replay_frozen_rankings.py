"""Replay supplied rankings against a frozen top-50 substrate without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from freeze_qwen_top50 import CONDITIONS, validate_frozen
from rerank_frozen_pools import _evaluate

RANKING_SCHEMA = "supplied-ranking-fixture-v1"
RESULT_SCHEMA = "qwen-top50-frozen-reranker-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(substrate_path: Path, rankings_path: Path) -> dict[str, Any]:
    substrate = _load(substrate_path)
    rankings = _load(rankings_path)
    validate_frozen(substrate)
    substrate_hash = _sha256(substrate_path)
    if rankings.get("schema_version") != RANKING_SCHEMA:
        raise ValueError("unsupported supplied-ranking schema")
    if rankings.get("execution_mode") != "synthetic_replay":
        raise ValueError("ranking input must be marked synthetic_replay")
    if rankings.get("ranking_origin") != "supplied_fixture":
        raise ValueError("ranking input must identify supplied_fixture origin")
    if rankings.get("substrate_sha256") != substrate_hash:
        raise ValueError("ranking input substrate hash does not match frozen substrate")
    label = rankings.get("label")
    if not isinstance(label, str) or not label.strip() or "\n" in label:
        raise ValueError("ranking label must be a nonempty single line")
    if set(rankings.get("conditions", {})) != set(CONDITIONS):
        raise ValueError("ranking conditions are missing or unexpected")
    query_count = substrate["scope"]["query_count"]
    if rankings.get("query_count") != query_count:
        raise ValueError("ranking declared query count does not match frozen substrate")

    results: dict[str, Any] = {}
    for condition in CONDITIONS:
        frozen_rows = substrate["conditions"][condition]["queries"]
        expected_ids = {row["query_id"] for row in frozen_rows}
        supplied_rows = rankings["conditions"][condition].get("queries")
        if rankings["conditions"][condition].get("query_count") != query_count:
            raise ValueError(f"ranking condition declared query count mismatch: {condition}")
        if not isinstance(supplied_rows, list) or len(supplied_rows) != len(frozen_rows):
            raise ValueError(f"ranking query count mismatch: {condition}")
        supplied_ids = [row.get("query_id") for row in supplied_rows]
        if any(not isinstance(value, str) or not value for value in supplied_ids):
            raise ValueError(f"invalid ranking query ID: {condition}")
        if len(set(supplied_ids)) != len(supplied_ids):
            raise ValueError(f"duplicate ranking query ID: {condition}")
        if set(supplied_ids) != expected_ids:
            raise ValueError(f"ranking query IDs do not match substrate: {condition}")
        supplied_by_id = {row["query_id"]: row for row in supplied_rows}
        ranking_by_query: dict[str, list[str]] = {}
        for frozen_row in frozen_rows:
            query_id = frozen_row["query_id"]
            candidate_ids = [item["candidate_id"] for item in frozen_row["candidates"]]
            ordered_ids = supplied_by_id[query_id].get("reranked_ids")
            if not isinstance(ordered_ids, list) or len(ordered_ids) != 50:
                raise ValueError(f"ranking must contain exactly 50 IDs: {condition}/{query_id}")
            if any(not isinstance(value, str) or not value for value in ordered_ids):
                raise ValueError(f"invalid candidate ID in ranking: {condition}/{query_id}")
            if len(set(ordered_ids)) != 50 or set(ordered_ids) != set(candidate_ids):
                raise ValueError(f"ranking is not a permutation of the frozen pool: {condition}/{query_id}")
            ranking_by_query[query_id] = ordered_ids
        results[condition] = _evaluate(frozen_rows, ranking_by_query)

    return {
        "schema_version": RESULT_SCHEMA,
        "status": "completed",
        "execution_mode": "synthetic_replay",
        "model_executed": False,
        "ranking_label": label.strip(),
        "ranking_origin": "supplied_fixture",
        "substrate_sha256": substrate_hash,
        "retriever": {"embedding_backend": "not_executed", "query_policy": "synthetic_fixture_only"},
        "reranker": {"name": label.strip(), "backend": "supplied_rankings", "model_executed": False},
        "runtime": {"candidate_count_per_query": 50},
        "conditions": results,
    }


def validate_result(result: dict[str, Any], substrate: dict[str, Any], substrate_hash: str) -> None:
    """Fail if a saved result is not the deterministic evaluation of its trace."""
    validate_frozen(substrate)
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unexpected ranking-result schema")
    execution_mode = result.get("execution_mode", "model_execution")
    if execution_mode not in {"synthetic_replay", "model_execution"}:
        raise ValueError("unknown result execution mode")
    if execution_mode == "synthetic_replay" and result.get("model_executed") is not False:
        raise ValueError("synthetic replay result claims model execution")
    if execution_mode == "model_execution" and result.get("model_executed") is False:
        raise ValueError("model-execution result claims no model was executed")
    if result.get("substrate_sha256") != substrate_hash:
        raise ValueError("result substrate hash does not match frozen substrate")
    if set(result.get("conditions", {})) != set(CONDITIONS):
        raise ValueError("result conditions are missing or unexpected")
    for condition in CONDITIONS:
        rows = substrate["conditions"][condition]["queries"]
        condition_result = result["conditions"][condition]
        trace_rows = condition_result.get("trace_rows")
        if not isinstance(trace_rows, list):
            raise TypeError(f"result trace rows are missing: {condition}")
        trace_ids = [row.get("query_id") for row in trace_rows]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError(f"duplicate result query ID: {condition}")
        if set(trace_ids) != {row["query_id"] for row in rows}:
            raise ValueError(f"result query IDs do not match substrate: {condition}")
        by_id = {row["query_id"]: row for row in trace_rows}
        rankings: dict[str, list[str]] = {}
        for frozen_row in rows:
            query_id = frozen_row["query_id"]
            ordered_ids = by_id[query_id].get("reranked_ids")
            candidate_ids = [item["candidate_id"] for item in frozen_row["candidates"]]
            if not isinstance(ordered_ids, list) or len(ordered_ids) != 50:
                raise ValueError(f"complete ranking missing: {condition}/{query_id}")
            if len(set(ordered_ids)) != 50 or set(ordered_ids) != set(candidate_ids):
                raise ValueError(f"result ranking differs from frozen pool: {condition}/{query_id}")
            rankings[query_id] = ordered_ids
        expected = _evaluate(rows, rankings)
        if condition_result != expected:
            raise ValueError(f"saved metrics or trace disagree with rankings: {condition}")


def load_validated_result(path: Path, substrate_path: Path | None) -> dict[str, Any]:
    result = _load(path)
    digest = result.get("substrate_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"missing or invalid substrate hash: {path}")
    mode = result.get("execution_mode", "model_execution")
    if mode not in {"synthetic_replay", "model_execution"}:
        raise ValueError(f"unknown execution mode: {path}")
    if mode == "synthetic_replay" and substrate_path is None:
        raise ValueError("synthetic replay comparison requires --substrate")
    if substrate_path is not None:
        substrate = _load(substrate_path)
        actual_hash = _sha256(substrate_path)
        if digest != actual_hash:
            raise ValueError(f"result substrate hash does not match --substrate: {path}")
        validate_result(result, substrate, actual_hash)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", required=True)
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    try:
        result = replay(Path(args.substrate), Path(args.rankings))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ranking replay failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "execution_mode": result["execution_mode"], "model_executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
