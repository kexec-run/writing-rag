# Jina selection for Instagram chains

Study ID: `study-008`

Disposition: `results-only`
Decision: structural_only; not promoted for chain selection.

## Question

Does symmetric Jina text matching select more useful candidate chains from a fixed set?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **3 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The J1 candidate-selection result was positive in only 3 of 5 folds and regressed in the p95 band. The archived decision is not to use Jina for Instagram chain selection.

## Evidence and limits

This is distinct from corpus retrieval and from Jina Reranker v3.5. The private anchor set and candidate payloads are not published. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`experiments/archived/instagram-chain-selection-jina-v1/jina_candidate_selection_experiment.py`](../../../experiments/archived/instagram-chain-selection-jina-v1/jina_candidate_selection_experiment.py)

## Revisit condition

The ledger says not to retry thresholds, modes, models or rerankers for this objective.
