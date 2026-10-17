# Qwen query-encoding policy comparison

Study ID: `study-027`

Disposition: `results-only`
Decision: Q0 remains the measured baseline; no Q1 improvement claimed.

## Question

Does the model-card query encoding policy improve Qwen retrieval on the project's cases?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **1 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

On the frozen pilot, Q0 MRR was 0.8000 and card-style Q1 query encoding was 0.7750. The difference is not proof that query instructions are generally harmful; the tested policy did not improve these cases.

## Evidence and limits

Small case suite; Q0 and Q1 remain experiment labels, not universal model rankings. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

No corresponding public experiment implementation is included in this seed.

## Revisit condition

Only with more independent cases and identical Q0/Q1 metrics.
