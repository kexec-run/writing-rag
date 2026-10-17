# Jina retrieval replacement attempts

Study ID: `study-013`

Disposition: `results-only`
Decision: remove Jina retrieval replacement; Qwen remains the corpus retriever.

## Question

Can Jina replace Qwen as the corpus retrieval embedding model?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **3 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The recorded Q2 Jina retrieval MRR was 0.7750 versus the Qwen Q0 reference of 0.8000 on the frozen pilot. This did not show an advantage for replacement.

## Evidence and limits

A small frozen pilot cannot establish universal superiority or infer behavior on other corpora. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

No corresponding public experiment implementation is included in this seed.

## Revisit condition

Only as a separately named model-role experiment with a fresh compatible index and the same frozen Q0/Q1 metrics.
