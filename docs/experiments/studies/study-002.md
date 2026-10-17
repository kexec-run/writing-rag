# Embedding runtime and Apple Neural Engine probes

Study ID: `study-002`

Disposition: `results-only`
Decision: remove the conversion path; no ANE backend or fallback.

## Question

Can alternate runtimes or larger embedding models improve throughput or retrieval quality on the available hardware?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **6 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Core ML numerical comparisons ran, but runtime evidence did not establish ANE execution. A separate 4B wordwise run was about 5.25 times slower, without retrieval-quality evidence.

## Evidence and limits

Numerical equivalence is not proof of device placement. Runtime measurements are machine-specific. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/embed_chunks.py`](../../../scripts/embed_chunks.py)

## Revisit condition

Require execution-level device evidence and a retrieval benchmark before reconsidering.
