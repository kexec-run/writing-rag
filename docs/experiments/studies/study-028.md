# Frozen Qwen top-50 reranking

Study ID: `study-028`

Disposition: `results-only`
Decision: reranking is a bounded ranking hypothesis; confirmation was not met.

## Question

Can a reranker move relevant items upward without changing the Qwen candidate set?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **3 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The top-50 pool fixes candidate recall. Jina can only reorder those Qwen candidates. The larger confirmation result remained `not_confirmed`, including a p99 regression, despite aggregate Recall@5 and Recall@10 gains.

## Evidence and limits

A reranker cannot recover a relevant item missing from top-50. Aggregate gains cannot override subgroup regression or the predeclared gate. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`experiments/active/qwen-top50-reranker-v2/freeze_qwen_top50.py`](../../../experiments/active/qwen-top50-reranker-v2/freeze_qwen_top50.py)
- [`experiments/active/qwen-top50-reranker-v2/rerank_frozen_pools.py`](../../../experiments/active/qwen-top50-reranker-v2/rerank_frozen_pools.py)

The v2 folder also has a verified synthetic ranking replay. It checks metric
calculation and ranking integrity using invented fixture data; it does not
reproduce this study’s private corpus or model results. See the [replay
instructions](../../../experiments/active/qwen-top50-reranker-v2/README.md#public-synthetic-replay).

## Revisit condition

Use a new frozen confirmation set and unchanged subgroup, uncertainty and damage criteria.
