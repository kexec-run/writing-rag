# Narrative reconstruction and source hydration

Study ID: `study-019`

Disposition: `results-only`
Decision: research direction only; canonical generation remains separate.

## Question

Can a provenance-bearing narrative representation preserve chronology and causal relations through article generation?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **1 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The codebase contains source-hydration and narrative-map building blocks, but retrieved snippets plus author scaffolds are not yet a single canonical map-and-verify path. The map must not invent unsupported causal edges.

## Evidence and limits

Conversation payloads and source-span maps are private. Current public code does not establish that every production generation path consumes a narrative map. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`writing_rag/plan/narrative_reconstruction.py`](../../../writing_rag/plan/narrative_reconstruction.py)
- [`writing_rag/plan/source_hydration.py`](../../../writing_rag/plan/source_hydration.py)
- [`writing_rag/verify/claims.py`](../../../writing_rag/verify/claims.py)

## Revisit condition

Evaluate explicit versus inferred relations against exact source spans before connecting them to claims.
