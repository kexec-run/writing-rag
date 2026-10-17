# Qwen top-50 with Jina reranker confirmation

Study ID: `study-011`

Disposition: `results-only`
Decision: not confirmed; no production or article-holdout promotion.

## Question

Can Jina Reranker v3.5 improve ordering inside an unchanged Qwen top-50 pool?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **1 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

On 110 unseen anchors, Recall@10 moved from 0.7364 to 0.8182 and Recall@5 from 0.6273 to 0.7182, with 15 rescues and 6 damages. The predeclared gate still failed: the p99 band regressed by 10 percentage points and uncertainty bounds were inadequate.

## Evidence and limits

The reranker cannot recover candidates absent from Qwen's pool. Corpus cases and run artifacts are private; figures are transcribed from the sanitized decision ledger, not recomputed here. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`experiments/active/j1-qwen-jina-confirmation-v1/evaluate_confirmation.py`](../../../experiments/active/j1-qwen-jina-confirmation-v1/evaluate_confirmation.py)
- [`experiments/active/qwen-top50-reranker-v2/rerank_frozen_pools.py`](../../../experiments/active/qwen-top50-reranker-v2/rerank_frozen_pools.py)

## Revisit condition

Only on a separately frozen confirmation set with sufficient p99 coverage and the original gate unchanged.
