# Jina symmetric text matching

Study ID: `study-014`

Disposition: `results-only`
Decision: direction supported on the frozen pilot; auxiliary use only.

## Question

Can symmetric Jina matching distinguish related text pairs better than the Qwen characterization baseline?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **6 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The Q3 text-matching pilot reported MRR 0.8500, average precision 0.8458, 14 of 20 rank-1 positives and five hard-negative failures. This supports further text-matching evaluation, not corpus retrieval or production promotion.

## Evidence and limits

The evaluation is small and its labels and payloads are private. It does not establish chain causality or article-quality gains. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

Public code related to this study:

- [`scripts/jina_text_matching_mlx.py`](../../../scripts/jina_text_matching_mlx.py)
- [`writing_rag/select/jina_text_matching_mlx.py`](../../../writing_rag/select/jina_text_matching_mlx.py)

## Revisit condition

Use a larger independent matching set and preserve the retrieval/matching task boundary.
