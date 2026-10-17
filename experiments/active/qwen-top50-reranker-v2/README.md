# Qwen top-50 reranker v2

Status: historical frozen-pool protocol for native Qwen versus Jina Reranker v3.5. The public synthetic replay described below is a separate test capability, not a reranker experiment result.

Qwen owns retrieval and the exact top-50 candidate pool. Jina v3.5 may only
listwise-reorder candidates already in that pool. It cannot add documents,
change Recall@50, or retrieve style evidence.

## Entrypoints

- `freeze_qwen_top50.py` — freeze Qwen pools and metadata
- `rerank_frozen_pools.py` — Jina v3.5 listwise reorder of frozen pools
- `compare_rerankers.py` — paired comparison helpers
- `replay_frozen_rankings.py` — compute metrics from supplied synthetic ranking permutations without loading a model

## Related

- Confirmation experiment: [../j1-qwen-jina-confirmation-v1/README.md](../j1-qwen-jina-confirmation-v1/README.md)
- Architecture boundary: Qwen owns retrieval and the frozen candidate pool;
  Jina can only reorder that pool. The architecture diagram and local
  per-query comparison artifact are omitted from this public snapshot.

The 110-anchor confirmation remained `not_confirmed`. Do not promote Jina v3.5
or authorize article holdout from this folder alone.

## Public research record

Disposition: `results-only`. A reranker cannot recover a relevant item missing from top-50. Aggregate gains cannot override subgroup regression or the predeclared gate.

See the [sanitized study summary](../../../docs/experiments/studies/study-028.md) and [complete experiment catalog](../../../docs/experiments/README.md).

## Public availability

- Catalog disposition: `results-only` ([study-028](../../../docs/experiments/studies/study-028.md)).
- Present in this folder: `README.md`, `compare_rerankers.py`, `freeze_qwen_top50.py`, `replay_frozen_rankings.py`, and `rerank_frozen_pools.py`.
- Omitted: private query/candidate pools, gold labels, model cache, and per-query outputs; the linked architecture diagram and local comparison artifact are also omitted.
- Reproduction limit: the historical model confirmation requires private frozen pools and model weights. The synthetic replay below validates ranking and metric handling only; it makes no model-quality claim.

## Public synthetic replay

The catalog registers this CPU-only capability separately from `study-028`.
It creates six invented queries with 50 neutral candidate passages each, then
replays fixed rank permutations through the existing metric functions. No
embedding, reranker, private data, or network access is used. The duplicated
condition payloads exercise the historical format; they are not independent
evaluation cases.

Run the commands from the repository root. They create files only inside a
fresh temporary directory:

```bash
set -eu
OUT="$(mktemp -d)"
python3 tests/fixtures/ranking_replay/build_fixture.py --spec tests/fixtures/ranking_replay/fixture_spec.json --output "$OUT/inputs"
python3 experiments/active/qwen-top50-reranker-v2/replay_frozen_rankings.py --substrate "$OUT/inputs/frozen-top50.json" --rankings "$OUT/inputs/qwen-rankings.json" --output "$OUT/qwen-1.json"
python3 experiments/active/qwen-top50-reranker-v2/replay_frozen_rankings.py --substrate "$OUT/inputs/frozen-top50.json" --rankings "$OUT/inputs/qwen-rankings.json" --output "$OUT/qwen-2.json"
python3 experiments/active/qwen-top50-reranker-v2/replay_frozen_rankings.py --substrate "$OUT/inputs/frozen-top50.json" --rankings "$OUT/inputs/jina-rankings.json" --output "$OUT/jina-1.json"
python3 experiments/active/qwen-top50-reranker-v2/replay_frozen_rankings.py --substrate "$OUT/inputs/frozen-top50.json" --rankings "$OUT/inputs/jina-rankings.json" --output "$OUT/jina-2.json"
python3 experiments/active/qwen-top50-reranker-v2/compare_rerankers.py --qwen "$OUT/qwen-1.json" --jina "$OUT/jina-1.json" --substrate "$OUT/inputs/frozen-top50.json" --output "$OUT/comparison.json"
python3 experiments/active/j1-qwen-jina-confirmation-v1/verify_repeatability.py --left "$OUT/qwen-1.json" --right "$OUT/qwen-2.json" --substrate "$OUT/inputs/frozen-top50.json"
python3 experiments/active/j1-qwen-jina-confirmation-v1/verify_repeatability.py --left "$OUT/jina-1.json" --right "$OUT/jina-2.json" --substrate "$OUT/inputs/frozen-top50.json"
```

The comparison writes JSON plus a Markdown report. Expected Recall@10 is 3/6
for native and synthetic Qwen ordering, and 4/6 for synthetic Jina ordering.
All three have Recall@50 of 5/6. Qwen ordering has one top-10 rescue and one
damage; Jina ordering has one rescue and no damage. These are fixture checks,
not reranker measurements. Complete ranking equality is required for both
repeatability checks.

The fixture’s independent expected values are in
[`expected_metrics.json`](../../../tests/fixtures/ranking_replay/expected_metrics.json).
