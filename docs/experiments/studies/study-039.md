# Embedding retrieval validity and source-homogeneous checks

Study ID: `study-039`

Disposition: `results-only`
Decision: Qwen baseline retained; no production backend change from this family.

## Question

Do alternative embedding or batching policies preserve retrieval validity across source and length strata?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **11 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The family contains multiple validation, batching, padding and source-homogeneous artifacts. Their directory count is not an independent sample count, and the underlying cases are not included in the release.

## Evidence and limits

Without the private frozen cases and full sanitized reports, aggregate outcomes are not re-computed here. No sanitized source result was available for a quantitative claim in this pass.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`experiments/active/zembed-retrieval-validity-v1/README.md`](../../../experiments/active/zembed-retrieval-validity-v1/README.md)
- [`experiments/active/zembed-retrieval-validity-v1/validate_artifacts.py`](../../../experiments/active/zembed-retrieval-validity-v1/validate_artifacts.py)

## Revisit condition

Use the existing validity contract and publish a sanitized summary with exact sample units and decisions.
