# Jina model and task-mode characterization

Study ID: `study-012`

Disposition: `results-only`
Decision: retain Jina only in explicitly bounded auxiliary roles.

## Question

How do Jina task modes behave on the project's small frozen semantic cases?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **13 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The characterization work tested retrieval, text matching and other modes. Task-specific behavior did not establish a reason to replace Qwen's production retrieval path.

## Evidence and limits

Small-pilot measurements are not broad retrieval evidence. Vectors from different task modes are not interchangeable. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`experiments/archived/jina-model-characterization/compare_jina_mode_runs.py`](../../../experiments/archived/jina-model-characterization/compare_jina_mode_runs.py)

## Revisit condition

Use separate task contracts and independent evaluation cases for any future mode study.
