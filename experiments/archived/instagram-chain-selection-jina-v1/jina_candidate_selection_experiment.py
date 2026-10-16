"""Compare Qwen retrieval after local Jina text-matching candidate selection.

Jina is used only to select among locally generated episode candidates. It is
never used to encode retrieval queries/documents or to replace the Qwen index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "scripts"
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(MODULE_DIR))

from episode_graph import build_candidate_windows, build_episode, build_composite_episode  # noqa: E402
from gated_episode_experiment import (  # noqa: E402
    _baseline_row,
    _cosine,
    _episode_row,
    _evaluate_rankings,
    _other_query,
    collect_excluded_ids,
    corpus_profile,
    percentile_thresholds,
    select_five_fold_anchors,
)
from jina_text_matching_mlx import JinaTextMatchingMLX  # noqa: E402
from pipeline_common import count_tokens, read_json, write_json  # noqa: E402
from validators import (  # noqa: E402
    run_fixture_checks,
    run_perturbation_checks,
    validate_episode,
)
from instagram_anchor_chain_experiment import build_anchor_chain  # noqa: E402


DEFAULT_MESSAGES = PROJECT_ROOT / "data/normalized/messages.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/instagram-jina-candidate-selection-v1"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models/weights/huggingface"
DEFAULT_EXCLUSIONS = (
    PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v1-screening-qwen",
    PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v1-confirmation-qwen",
    PROJECT_ROOT / "data/generated/instagram-conversation-episodes-v2-gated-qwen-rerun",
)


def _candidate_texts(
    episode: dict[str, Any], by_id: dict[str, dict[str, Any]]
) -> tuple[str, str]:
    context = "\n".join(
        str(by_id[message_id].get("content", ""))
        for message_id in episode.get("context_message_ids", [])
    ).strip()
    self_text = "\n".join(
        str(by_id[message_id].get("content", ""))
        for message_id in episode.get("self_message_ids", [])
    ).strip()
    return context, self_text


def _candidate_record(
    candidate_id: str,
    episode: dict[str, Any],
    source: str,
    anchor: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    context, self_text = _candidate_texts(episode, by_id)
    return {
        "candidate_id": candidate_id,
        "source": source,
        "fold": anchor["fold"],
        "band": anchor["band"],
        "message_ids": list(episode.get("message_ids", [])),
        "episode": episode,
        "context_text": context,
        "self_text": self_text,
    }


def _candidate_pool(
    ordered: list[dict[str, Any]],
    anchor: dict[str, Any],
    anchor_index: int,
    by_id: dict[str, dict[str, Any]],
    *,
    widths: tuple[int, ...],
    max_tokens: int,
    baseline_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    observed_gaps = [
        max(0.0, (int(right.get("timestamp_ms", 0)) - int(left.get("timestamp_ms", 0))) / 60_000)
        for left, right in zip(ordered, ordered[1:])
    ]
    baseline_episode = baseline_row["primary"]
    records = [
        _candidate_record("B0", baseline_episode, "baseline", anchor, by_id)
    ]
    candidates = build_candidate_windows(
        ordered, anchor_index, widths=widths, max_tokens=max_tokens
    )
    episodes = [
        build_episode(candidate, ordered, observed_gaps=observed_gaps)
        for candidate in candidates
    ]
    c1_primary = build_composite_episode(
        ordered, anchor_index, widths=widths, max_tokens=max_tokens
    )["primary"]["candidate"]["candidate_id"]
    records.extend(
        _candidate_record(
            str(episode["candidate"]["candidate_id"]),
            episode,
            "c1_candidate",
            anchor,
            by_id,
        )
        for episode in episodes
    )
    return records, c1_primary


def _score_candidates(
    jina: JinaTextMatchingMLX,
    candidate_pools: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    pairs: list[tuple[str, str, str]] = []
    for anchor_id, candidates in sorted(candidate_pools.items()):
        for candidate in candidates:
            if candidate["context_text"] and candidate["self_text"]:
                pairs.append(
                    (
                        f"{anchor_id}:{candidate['candidate_id']}:context",
                        candidate["context_text"],
                        candidate["self_text"],
                    )
                )
    texts = [text for _, context, self_text in pairs for text in (context, self_text)]
    vectors = jina.encode(texts)
    vector_map = {
        pair_id: vectors[index * 2 : index * 2 + 2]
        for index, (pair_id, _, _) in enumerate(pairs)
    }
    reverse_vectors = jina.encode(list(reversed(texts)))
    symmetry_differences = []
    for index, (pair_id, _, _) in enumerate(pairs):
        original = vector_map[pair_id]
        reverse = [
            reverse_vectors[len(reverse_vectors) - 1 - index * 2],
            reverse_vectors[len(reverse_vectors) - 2 - index * 2],
        ]
        symmetry_differences.extend(
            abs(left - right)
            for left_row, right_row in zip(original, reverse)
            for left, right in zip(left_row, right_row)
        )
    score_map: dict[str, dict[str, float]] = {
        anchor_id: {} for anchor_id in candidate_pools
    }
    for pair_id, _, _ in pairs:
        anchor_id, candidate_id, _ = pair_id.split(":", 2)
        pair_vectors = vector_map[pair_id]
        score_map[anchor_id][candidate_id] = _cosine(pair_vectors[0], pair_vectors[1])
    for anchor_id, candidates in candidate_pools.items():
        for candidate in candidates:
            score_map[anchor_id].setdefault(candidate["candidate_id"], -1.0)
    return score_map, {
        "encoded_pair_count": len(pairs),
        "encoded_text_count": len(texts),
        "reverse_order_max_abs_difference": max(symmetry_differences, default=0.0),
    }


def _select_jina_rows(
    candidate_pools: dict[str, list[dict[str, Any]]],
    score_map: dict[str, dict[str, float]],
    anchors_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for anchor_id, candidates in sorted(candidate_pools.items()):
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -score_map[anchor_id][candidate["candidate_id"]],
                candidate["candidate_id"],
            ),
        )
        selected = ranked[0]
        selected_score = score_map[anchor_id][selected["candidate_id"]]
        second_score = (
            score_map[anchor_id][ranked[1]["candidate_id"]] if len(ranked) > 1 else -1.0
        )
        anchor = anchors_by_id[anchor_id]
        rows.append(
            {
                "anchor_message_id": anchor_id,
                "fold": anchor["fold"],
                "band": anchor["band"],
                "method": "jina_selected",
                "candidate_id": selected["candidate_id"],
                "selection_score": selected_score,
                "selection_margin": selected_score - second_score,
                "primary": selected["episode"],
            }
        )
        selections[anchor_id] = {
            "ranked_candidate_ids": [candidate["candidate_id"] for candidate in ranked],
            "selected_candidate_id": selected["candidate_id"],
            "selected_score": selected_score,
            "selection_margin": selected_score - second_score,
            "inputs": "local_context_and_self_messages_only",
        }
    return rows, selections


def _structural_checks(
    result: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = result["rows"]
    anchors = result["anchors"]
    positive = {
        "exactly_60_anchors": len(anchors) == 60,
        "same_anchor_ids": len({row["anchor_message_id"] for row in rows["baseline"]}) == 60
        and {
            row["anchor_message_id"] for row in rows["baseline"]
        }
        == {row["anchor_message_id"] for row in rows["jina_selected"]},
        "five_folds": {str(anchor["fold"]) for anchor in anchors} == {"0", "1", "2", "3", "4"},
        "queries_available": all(result["queries"].values()),
        "fixture_checks_passed": run_fixture_checks()["passed"],
        "perturbation_checks_passed": run_perturbation_checks(build_composite_episode)["passed"],
        "jina_is_local_only": result["selection_contract"]["qwen_query_used"] is False,
    }
    validation_errors: dict[str, list[str]] = {}
    for row in rows["jina_selected"]:
        errors = validate_episode(row["primary"], by_id)
        if errors:
            validation_errors[row["anchor_message_id"]] = errors
    fold_sets = {
        fold: {
            message_id
            for row in rows["jina_selected"]
            if row["fold"] == fold
            for message_id in row["primary"].get("message_ids", [])
        }
        for fold in range(5)
    }
    negative = {
        "no_validation_errors": not validation_errors,
        "no_fold_message_overlap": all(
            not (fold_sets[left] & fold_sets[right])
            for left in range(5)
            for right in range(left + 1, 5)
        ),
        "no_style_contamination": not any(
            by_id[message_id].get("speaker") != "self"
            for row in rows["jina_selected"]
            for message_id in row["primary"].get("self_message_ids", [])
        ),
        "jina_reverse_order_symmetry": result["jina_symmetry"][
            "reverse_order_max_abs_difference"
        ] == 0.0,
    }
    return {
        "positive": positive,
        "negative": negative,
        "validation_errors": validation_errors,
        "passed": all(positive.values()) and all(negative.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_messages = read_json(args.messages)
    instagram = [message for message in all_messages if message.get("source") == "instagram"]
    ordered = sorted(
        instagram,
        key=lambda message: (int(message.get("timestamp_ms", 0)), str(message["message_id"])),
    )
    by_id = {str(message["message_id"]): message for message in ordered}
    thresholds = percentile_thresholds(instagram)
    excluded_ids = collect_excluded_ids([Path(value) for value in args.exclude])
    anchors_by_fold, selection_metadata = select_five_fold_anchors(
        ordered,
        thresholds=thresholds,
        excluded_ids=excluded_ids,
        fold_count=5,
        per_band_fold=args.anchors_per_band_fold,
        query_radius=args.query_context_radius,
        minimum_separation=args.minimum_separation,
    )
    anchors = [anchor for items in anchors_by_fold.values() for anchor in items]
    anchors_by_id = {anchor["message_id"]: anchor for anchor in anchors}
    baseline_rows: list[dict[str, Any]] = []
    c1_rows: list[dict[str, Any]] = []
    candidate_pools: dict[str, list[dict[str, Any]]] = {}
    queries: dict[str, str] = {}
    for anchor in anchors:
        anchor_index = next(
            index
            for index, message in enumerate(ordered)
            if str(message["message_id"]) == anchor["message_id"]
        )
        baseline_chain = build_anchor_chain(
            ordered,
            anchor_index,
            token_counter=count_tokens,
            max_gap_minutes=args.baseline_gap_minutes,
            max_tokens=args.baseline_max_tokens,
            short_message_tokens=args.baseline_short_message_tokens,
        )
        baseline = _baseline_row(baseline_chain, anchor, by_id)
        c1 = build_composite_episode(
            ordered, anchor_index, widths=tuple(args.widths), max_tokens=args.max_tokens
        )
        baseline_rows.append(baseline)
        c1_rows.append(_episode_row(c1, anchor, "c1_full"))
        candidate_pools[anchor["message_id"]], _ = _candidate_pool(
            ordered,
            anchor,
            anchor_index,
            by_id,
            widths=tuple(args.widths),
            max_tokens=args.max_tokens,
            baseline_row=baseline,
        )
        queries[anchor["message_id"]] = _other_query(
            ordered, anchor_index, args.query_context_radius
        )
    jina = JinaTextMatchingMLX(model_cache=args.model_cache, batch_size=1)
    score_map, jina_symmetry = _score_candidates(jina, candidate_pools)
    jina_rows, selections = _select_jina_rows(candidate_pools, score_map, anchors_by_id)
    result: dict[str, Any] = {
        "schema_version": "instagram-jina-candidate-selection-v1",
        "method": {
            "baseline": "B0_frozen_baseline",
            "diagnostic": "C1_overlap_context_causal_v1",
            "candidate": "J1_jina_local_text_matching_selection",
            "jina_model": "jinaai/jina-embeddings-v5-text-small-mlx",
            "jina_task": "text-matching",
            "qwen_retrieval_model": "Qwen/Qwen3-Embedding-0.6B",
        },
        "selection_contract": {
            "qwen_query_used": False,
            "article_prompt_used": False,
            "jina_vector_space": "jina-v5-text-matching-mlx",
            "jina_batch_size": 1,
            "candidate_pool": ["B0", *[f"w{width}" for width in args.widths]],
            "selection_inputs": "candidate_local_context_and_self_messages",
        },
        "scope": {
            "messages_path": str(Path(args.messages)),
            "source": "instagram",
            "canonical_message_count": len(instagram),
            "anchor_count": len(anchors),
            "fold_count": 5,
            "query_context_radius": args.query_context_radius,
        },
        "thresholds": thresholds,
        "selection": selection_metadata,
        "excluded_anchor_ids": sorted(excluded_ids),
        "anchors": anchors,
        "queries": queries,
        "rows": {
            "baseline": baseline_rows,
            "c1_full": c1_rows,
            "jina_selected": jina_rows,
        },
        "candidate_selections": selections,
        "jina_contract": jina.contract(),
        "jina_symmetry": jina_symmetry,
        "corpus_profile": corpus_profile(instagram),
    }
    result["structural"] = _structural_checks(result, by_id)
    if result["structural"]["passed"]:
        result["qwen_retrieval"] = _evaluate_rankings(
            {
                "baseline": baseline_rows,
                "c1_full": c1_rows,
                "jina_selected": jina_rows,
            },
            queries,
            by_id,
            model_cache=args.model_cache,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
    return result


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Instagram Jina candidate-selection experiment",
        "",
        "Jina was used only for local symmetric text matching. Qwen remained the retrieval model.",
        "",
        f"Structural verification passed: `{result['structural']['passed']}`.",
        f"Jina reverse-order maximum difference: `{result['jina_symmetry']['reverse_order_max_abs_difference']}`.",
        "",
        "| Condition | MRR | Recall@10 | nDCG@10 | Δ MRR | Δ Recall@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    retrieval = result.get("qwen_retrieval")
    if retrieval:
        for condition in ("baseline", "c1_full", "jina_selected"):
            metrics = retrieval["conditions"][condition]
            lines.append(
                f"| {condition} | {metrics['mrr']:.6f} | {metrics['recall_at_10']:.6f} | "
                f"{metrics['ndcg_at_10']:.6f} | {metrics.get('delta_mrr', 0.0):+.6f} | "
                f"{metrics.get('delta_recall_at_10', 0.0):+.6f} |"
            )
    else:
        lines.append("| retrieval | not run | not run | not run | — | — |")
    lines.extend(
        [
            "",
            "The Jina selector did not receive the Qwen evaluation query, article prompt, or retrieval score.",
            "All Jina vectors remain in the separate text-matching vector space.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--exclude", action="append", default=[str(path) for path in DEFAULT_EXCLUSIONS])
    parser.add_argument("--anchors-per-band-fold", type=int, default=4)
    parser.add_argument("--query-context-radius", type=int, default=8)
    parser.add_argument("--minimum-separation", type=int, default=40)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--baseline-gap-minutes", type=int, default=15)
    parser.add_argument("--baseline-max-tokens", type=int, default=700)
    parser.add_argument("--baseline-short-message-tokens", type=int, default=8)
    parser.add_argument("--model-cache", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "jina-selection-results.json", result)
    (output / "report.md").write_text(_render(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "structural": result["structural"]}, indent=2))
    return 0 if result["structural"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
