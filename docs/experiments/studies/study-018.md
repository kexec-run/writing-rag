# Narrative-map smoke and packet experiments

Study ID: `study-018`

Disposition: `results-only`
Decision: experiment hook only; not a canonical production stage.

## Question

Can source-backed narrative nodes and relations be represented before article drafting?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **3 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The experiments explored mapping hydrated messages to narrative units and packets. The production article entry point does not establish this as a required, fully verified stage for every run.

## Evidence and limits

Some directories are empty or smoke-only; an empty directory is inventoried but contains no run result. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`writing_rag/plan/narrative_reconstruction.py`](../../../writing_rag/plan/narrative_reconstruction.py)
- [`writing_rag/plan/source_hydration.py`](../../../writing_rag/plan/source_hydration.py)

## Revisit condition

Require complete source-span provenance and independent validation of explicit and inferred relation handling.
