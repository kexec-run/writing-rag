"""Create one fresh, contamination-checked Instagram confirmation set.

This creates episode cases only.  Model execution is performed separately by
``instagram_validation.py`` so the selected substrate is inspectable before
any Qwen or zembed ranking is run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
GATED_RUNNER = ROOT / "experiments/active/instagram-conversation-episodes/gated_episode_experiment.py"
DEFAULT_MESSAGES = ROOT / "data/normalized/messages.json"
DEFAULT_OUTPUT = ROOT / "data/generated/zembed-instagram-confirmation-v1"
DEFAULT_PRIOR = [
    ROOT / "data/generated/instagram-conversation-episodes-v1-screening-qwen",
    ROOT / "data/generated/instagram-conversation-episodes-v1-confirmation-qwen",
    ROOT / "data/generated/instagram-conversation-episodes-v2-gated-qwen-rerun/gated-results.json",
    ROOT / "data/generated/instagram-jina-candidate-selection-v1/jina-selection-results.json",
]


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [
        candidate
        for candidate in (
            path / "episode-results.json",
            path / "gated-results.json",
            path / "jina-selection-results.json",
        )
        if candidate.is_file()
    ]


def _collect_ids(value: Any, ids: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {
                "anchor_message_id", "message_id", "self_message_ids",
                "context_message_ids", "message_ids", "window_message_ids",
            }:
                if isinstance(child, str):
                    ids.add(child)
                elif isinstance(child, list):
                    ids.update(str(item) for item in child if isinstance(item, str))
            _collect_ids(child, ids)
    elif isinstance(value, list):
        for child in value:
            _collect_ids(child, ids)


def _excluded_messages(
    prior_paths: list[Path],
    ordered: list[dict[str, Any]],
    query_radius: int,
) -> tuple[set[str], dict[str, str]]:
    excluded: set[str] = set()
    hashes: dict[str, str] = {}
    by_id = {str(message["message_id"]): index for index, message in enumerate(ordered)}
    prior_anchor_ids: set[str] = set()
    for path in prior_paths:
        for result_path in _result_files(path):
            result = _load(result_path)
            hashes[str(result_path)] = _sha256_file(result_path)
            local_ids: set[str] = set()
            _collect_ids(result, local_ids)
            excluded.update(local_ids)
            if isinstance(result, dict):
                prior_anchor_ids.update(str(anchor_id) for anchor_id in result.get("queries", {}) if anchor_id)
                prior_anchor_ids.update(
                    str(row.get("anchor_message_id"))
                    for row in result.get("baseline_rows", [])
                    if row.get("anchor_message_id")
                )
                for anchors in result.get("cohorts", {}).values():
                    prior_anchor_ids.update(
                        str(anchor.get("message_id"))
                        for anchor in anchors
                        if anchor.get("message_id")
                    )
    for anchor_id in prior_anchor_ids:
        index = by_id.get(anchor_id)
        if index is None:
            continue
        start = max(0, index - query_radius)
        end = min(len(ordered), index + query_radius + 1)
        excluded.update(str(message["message_id"]) for message in ordered[start:end])
    return excluded, hashes


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    gate = _load_module(GATED_RUNNER, "instagram_confirmation_gate")
    messages = _load(Path(args.messages))
    instagram = [message for message in messages if message.get("source") == "instagram"]
    ordered = sorted(
        instagram,
        key=lambda message: (int(message.get("timestamp_ms", 0)), str(message["message_id"])),
    )
    by_id = {str(message["message_id"]): message for message in ordered}
    thresholds = gate.percentile_thresholds(instagram)
    prior_paths = [Path(value) for value in args.prior]
    excluded_ids, prior_hashes = _excluded_messages(
        prior_paths, ordered, args.query_context_radius
    )
    selected, selection = gate.select_five_fold_anchors(
        ordered,
        thresholds=thresholds,
        excluded_ids=excluded_ids,
        fold_count=5,
        per_band_fold=4,
        query_radius=args.query_context_radius,
        minimum_separation=args.minimum_separation,
    )
    anchors = [anchor for items in selected.values() for anchor in items]
    baseline_rows: list[dict[str, Any]] = []
    c1_rows: list[dict[str, Any]] = []
    queries: dict[str, str] = {}
    for anchor in anchors:
        anchor_index = next(
            index for index, message in enumerate(ordered)
            if str(message["message_id"]) == anchor["message_id"]
        )
        baseline_chain = gate.build_anchor_chain(
            ordered,
            anchor_index,
            token_counter=gate.count_tokens,
            max_gap_minutes=args.baseline_gap_minutes,
            max_tokens=args.baseline_max_tokens,
            short_message_tokens=args.baseline_short_message_tokens,
        )
        baseline_rows.append(gate._baseline_row(baseline_chain, anchor, by_id))
        c1_rows.append(
            gate._episode_row(
                gate.build_composite_episode(
                    ordered,
                    anchor_index,
                    widths=tuple(args.widths),
                    max_tokens=args.max_tokens,
                ),
                anchor,
                "c1_full",
            )
        )
        queries[anchor["message_id"]] = gate._other_query(
            ordered, anchor_index, args.query_context_radius
        )
    rows = {
        "baseline": baseline_rows,
        "c1_full": c1_rows,
        "gated": c1_rows,
    }
    structural = gate._structural_checks(
        ordered,
        selected,
        rows,
        by_id,
        thresholds=thresholds,
        excluded_ids=excluded_ids,
        selection_metadata=selection,
    )
    return {
        "schema_version": "zembed-instagram-confirmation-v1",
        "status": "unranked_confirmation_substrate",
        "method": {
            "baseline": "B0_frozen_baseline",
            "primary": "C1_overlap_context_causal_v1",
            "qwen_retrieval": "Qwen/Qwen3-Embedding-0.6B",
            "zembed_retrieval": "lexrivera/zembed-1-embedding-mlx-6Bit",
            "no_jina": True,
            "no_reranker": True,
        },
        "scope": {
            "messages_path": str(Path(args.messages)),
            "source": "instagram",
            "canonical_message_count": len(instagram),
            "anchor_count": len(anchors),
            "fold_count": 5,
            "freshness_policy": "exclude_prior_anchor_and_context_message_windows",
        },
        "thresholds": thresholds,
        "selection": selection,
        "prior_artifacts": prior_hashes,
        "excluded_message_count": len(excluded_ids),
        "cohorts": selected,
        "queries": queries,
        "rows": rows,
        "structural": structural,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prior", action="append", default=[str(path) for path in DEFAULT_PRIOR])
    parser.add_argument("--query-context-radius", type=int, default=8)
    parser.add_argument("--minimum-separation", type=int, default=40)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--baseline-gap-minutes", type=int, default=15)
    parser.add_argument("--baseline-max-tokens", type=int, default=700)
    parser.add_argument("--baseline-short-message-tokens", type=int, default=8)
    args = parser.parse_args()
    result = _build_result(args)
    output = Path(args.output)
    _write(output / "confirmation-cases.json", result)
    print(json.dumps({
        "output": str(output / "confirmation-cases.json"),
        "anchor_count": result["scope"]["anchor_count"],
        "structural_passed": result["structural"]["passed"],
        "selection_errors": result["selection"]["selection_errors"],
    }, indent=2))
    return 0 if result["structural"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
