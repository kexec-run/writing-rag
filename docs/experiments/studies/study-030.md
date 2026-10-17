# Scaffold with Qwen and Jina selection

Study ID: `study-030`

Disposition: `results-only`
Decision: no scaffold minimum or Jina default promoted.

## Question

Which parts of a user-provided narrative scaffold are sufficient for guided generation?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **2 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The corrected v2 comparison rejected all article variants. B2-lite had the fewest issues (3), compared with 6 for full B2 with Jina and 17 for the load-bearing semantic condition.

## Evidence and limits

Issue counts are not acceptance rates and are target-specific. The scaffold may carry author intent not present in retrieved text. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

No corresponding public experiment implementation is included in this seed.

## Revisit condition

Use partial ablations on held-out targets and retain user-supplied narrative intent.
