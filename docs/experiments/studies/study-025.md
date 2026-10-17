# Qwen embedding characterization

Study ID: `study-025`

Disposition: `results-only`
Decision: corrected 22-case pilot retained as exploratory characterization.

## Question

How stable are Qwen embedding similarities and ranks under a controlled case suite?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **16 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The corrected pilot formalized numeric, ranking and repeatability checks. Its results are exploratory; it does not establish a universal embedding quality ranking or authorize a Jina retrieval switch.

## Evidence and limits

Cases are private and small. The characterization metrics are not equivalent to downstream article acceptance. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/compare_qwen_characterization_runs.py`](../../../scripts/compare_qwen_characterization_runs.py)
- [`scripts/embed_chunks.py`](../../../scripts/embed_chunks.py)

## Revisit condition

Add independent cases only under a separately reviewed confirmation protocol.
