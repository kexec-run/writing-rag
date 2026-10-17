# Shared pipeline and experiment-support scripts

Study ID: `study-034`

Disposition: `pending sanitization`
Decision: not an independent experiment family.

## Question

Which excluded Python components supported one or more historical experiments?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **0 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Some excluded Python files are shared import, validation or data-preparation tools rather than standalone experiments. They are inventoried so their disposition is visible, but they are not counted as separate studies.

## Evidence and limits

A support-file association is administrative and does not imply that the file is safe or complete for public execution. No sanitized source result was available for a quantitative claim in this pass.

## Reproduction

Status: `blocked_pending_sanitization`.

Public code related to this study:

- [`scripts/run_pipeline.py`](../../../scripts/run_pipeline.py)
- [`scripts/validate_outputs.py`](../../../scripts/validate_outputs.py)

## Revisit condition

Review dependencies and synthetic tests before restoring any support file.
