# Qwen retrieval with bounded Jina text selection

Study ID: `study-026`

Disposition: `results-only`
Decision: Qwen remains retrieval; A0 remains default; Jina is an auxiliary experiment.

## Question

Can Jina matching select a better subset from a fixed Qwen candidate pool for article generation?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **1 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The frozen-pool pilot reported MRR 0.8000 to 0.8500 and nDCG@10 0.8586 to 0.8884 after Jina selection. Both generated articles were rejected, so retrieval ranking gains did not establish article acceptance.

## Evidence and limits

Jina did not retrieve from the corpus. A ranking gain and an accepted article are separate outcomes. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/jina_text_selection.py`](../../../scripts/jina_text_selection.py)

## Revisit condition

Use held-out targets with identical Qwen candidate IDs, style evidence, generation budget and verifier.
