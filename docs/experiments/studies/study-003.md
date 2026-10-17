# Generation engine and retry ablations

Study ID: `study-003`

Disposition: `results-only`
Decision: duplicate retry variants were removed from the reusable path.

## Question

Can alternate generation and retry contracts improve grounded article completion?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **11 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Retry and engine variants did not establish a reliable acceptance improvement. The archived decision record says retry output remained rejected.

## Evidence and limits

Private prompt contracts, generated prose and full traces are not included. The published conclusion is limited to the recorded decision. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/generate.py`](../../../scripts/generate.py)
- [`scripts/paragraph_generation.py`](../../../scripts/paragraph_generation.py)

## Revisit condition

Revisit only after a measured change to retrieval, evidence contracts or verifier behavior.
