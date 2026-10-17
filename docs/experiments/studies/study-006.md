# Instagram conversation-chain construction

Study ID: `study-006`

Disposition: `pending sanitization`
Decision: keep the filtered chain pilot exploratory; production ingestion is unchanged.

## Question

Can short messages be grouped into useful conversation chains using time, reply structure and causal cues?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **1 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

Time grouping and causal heuristics formed candidate chains, but some chains overmerged and proxy retrieval did not beat the baseline.

## Evidence and limits

The public runner has unresolved local imports, and the underlying conversation corpus is private. Heuristic structure is not human-verified ground truth. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `blocked_pending_sanitization`.

Public code related to this study:

- [`experiments/active/instagram-chain-retrieval/instagram_mdx_filtered_ablation.py`](../../../experiments/active/instagram-chain-retrieval/instagram_mdx_filtered_ablation.py)

## Revisit condition

Restore only after dependencies are complete and overlap, perturbation and Qwen retrieval checks are reproducible on sanitized fixtures.
