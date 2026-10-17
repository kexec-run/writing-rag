# Interpretation and claim-boundary audits

Study ID: `study-010`

Disposition: `pending sanitization`
Decision: diagnostic tooling only; no general semantic-quality claim.

## Question

Can generated interpretations be distinguished from directly supported observations and claims?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **4 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The held artifacts include interpretation traces and audits. Their payloads were not used to make public claims because they contain source-linked article material.

## Evidence and limits

The trace content and source spans are private; no public payload was reviewed or copied. No sanitized source result was available for a quantitative claim in this pass.

## Reproduction

Status: `blocked_pending_sanitization`.

Public code related to this study:

- [`scripts/interpretation_audit.py`](../../../scripts/interpretation_audit.py)
- [`writing_rag/verify/claims.py`](../../../writing_rag/verify/claims.py)

## Revisit condition

Publish only a source-free schema description and aggregate validation result after a separate privacy review.
