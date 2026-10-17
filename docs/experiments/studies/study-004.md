# Graph-aware article retrieval

Study ID: `study-004`

Disposition: `results-only`
Decision: removed from production retrieval.

## Question

Does graph expansion add relevant evidence beyond the existing retrieval candidates?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **9 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The recorded article run produced zero graph expansions and no article change. This did not justify adding graph expansion to the canonical path.

## Evidence and limits

The result is from a bounded article experiment; it does not show that graphs are never useful. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/graph_retrieve.py`](../../../scripts/graph_retrieve.py)
- [`writing_rag/retrieve/graph_retrieve.py`](../../../writing_rag/retrieve/graph_retrieve.py)

## Revisit condition

Require a held-out retrieval and grounded-generation gain before reconsidering.
