# Evidence trace and provenance diagnostics

Study ID: `study-035`

Disposition: `pending sanitization`
Decision: private diagnostics retained; no trace payload published.

## Question

Can each generated claim be traced to the exact source and relation evidence that supports it?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **4 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Trace artifacts exist for direct article runs and boundary tests. Their source-linked payloads were inventoried without copying them into the release.

## Evidence and limits

Message IDs, source spans, private text and hashes are intentionally absent from the public register. No sanitized source result was available for a quantitative claim in this pass.

## Reproduction

Status: `blocked_pending_sanitization`.

Public code related to this study:

- [`scripts/evidence_trace.py`](../../../scripts/evidence_trace.py)
- [`writing_rag/verify/evidence_trace.py`](../../../writing_rag/verify/evidence_trace.py)

## Revisit condition

Publish only schema-level examples built from synthetic text and synthetic identifiers.
