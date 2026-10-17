# Paragraph-level retrieval and drafting

Study ID: `study-022`

Disposition: `results-only`
Decision: not promoted as a replacement for the canonical global retrieval path.

## Question

Does retrieving evidence per paragraph improve coverage without increasing unsupported claims?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **10 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The work explored per-paragraph evidence selection and generation. The public decision record does not establish a repeatable accepted-article advantage.

## Evidence and limits

The different numbered directories are not independent studies; many are diagnostic or repeated runs. No sanitized source result was available for a quantitative claim in this pass.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/paragraph_generation.py`](../../../scripts/paragraph_generation.py)
- [`writing_rag/generate/paragraph_generation.py`](../../../writing_rag/generate/paragraph_generation.py)

## Revisit condition

Use held-out targets, identical style evidence and claim-unit accounting before changing defaults.
