# Whole-article versus per-slot retrieval

Study ID: `study-032`

Disposition: `results-only`
Decision: not promoted; global Qwen retrieval remains the current path.

## Question

Does per-slot Qwen retrieval, with or without reranking inside the same pool, improve grounded article generation?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **3 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

On duality, all arms were rejected; S0 had five claim failures and S1 had none. On the corrected exploratory near-black run, all arms were rejected; S0 had zero and S1 had two claim failures. Neither target produced an accepted article.

## Evidence and limits

The near-black comparison was resource-safe and exploratory. Two targets are insufficient for general promotion. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/generate.py`](../../../scripts/generate.py)

## Revisit condition

Use a new held-out target pair, fixed reader-facing scaffolds and claim-unit metrics.
