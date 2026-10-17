# Automatic conversation-role and label directions

Study ID: `study-009`

Disposition: `results-only`
Decision: archive the human-label-gated direction; automatic labels remain hypotheses.

## Question

Can conversation roles be inferred automatically from speaker turns, timing and response behavior?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **1 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Human labels were unavailable. Pseudo-labels derived from heuristics are not independent ground truth and cannot validate the same heuristics that produced them.

## Evidence and limits

No manual chat labeling was performed. Do not describe pseudo-labels as human judgments or semantic truth. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

No corresponding public experiment implementation is included in this seed.

## Revisit condition

Use independent observable checks, overlap-aware episode evaluation and perturbation tests; keep label provenance explicit.
