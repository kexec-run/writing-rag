# Retrieval-quality diagnostics

Study ID: `study-029`

Disposition: `results-only`
Decision: diagnostic series; later frozen studies govern current conclusions.

## Question

How do changes to retrieval candidates affect ranking and evidence coverage?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **10 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Multiple numbered retrieval-quality bundles exist. The directory count is a versioned artifact history, not independent confirmation evidence.

## Evidence and limits

Earlier metrics were superseded by corrected characterization and confirmation protocols. This inventory does not recompute historical scores. No sanitized source result was available for a quantitative claim in this pass.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/evaluate_retrieval.py`](../../../scripts/evaluate_retrieval.py)
- [`scripts/diagnose_retrieval_coverage.py`](../../../scripts/diagnose_retrieval_coverage.py)

## Revisit condition

Use the current canonical Qwen index and a frozen, versioned case suite.
