# Evidence locality and backfill diagnostics

Study ID: `study-016`

Disposition: `results-only`
Decision: diagnostic artifacts; no canonical-path change.

## Question

Does retrieving nearby source context help preserve the evidence needed for a requested narrative unit?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **4 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Several locality and backfill artifact sets exist. Their per-case payloads were not independently re-audited in this pass, so no numerical or acceptance result is asserted.

## Evidence and limits

Artifacts may share targets and are not independent confirmation runs by directory count. No sanitized source result was available for a quantitative claim in this pass.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`writing_rag/plan/source_hydration.py`](../../../writing_rag/plan/source_hydration.py)
- [`writing_rag/plan/narrative_reconstruction.py`](../../../writing_rag/plan/narrative_reconstruction.py)

## Revisit condition

Publish a result only when a sanitized report declares controls, sample unit and the run-to-study relationship.
