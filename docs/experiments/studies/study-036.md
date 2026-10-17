# Typed article distillation

Study ID: `study-036`

Disposition: `results-only`
Decision: validated section fallback works in the recorded run; continuity pass is not accepted.

## Question

Can a two-pass process convert evidence-bearing sections into a coherent article without adding claims?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **6 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The O0 typed v6 run passed 10 of 10 section checks. The continuity rewrite truncated at its recorded budget and failed validation, so the workflow used `fallback_to_section_draft`.

## Evidence and limits

A validated section draft is not equivalent to an accepted continuous article. The target draft and trace remain private. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`writing_rag/verify/claims.py`](../../../writing_rag/verify/claims.py)

## Revisit condition

Separate output truncation from rewrite infidelity with a declared continuity budget and unchanged evidence constraints.
