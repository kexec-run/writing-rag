# UEmbed screening and reranking

Study ID: `study-037`

Disposition: `pending sanitization`
Decision: screening was invalid; later fused retrieval remained not confirmed.

## Question

Does UEmbed provide complementary dense or sparse candidates for Instagram retrieval?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **1 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The initial screen was invalid because it used the wrong pooled token and omitted the chat template. A later frozen stage showed directionally positive Qwen–UEmbed fusion but missed the declared confirmation gates; production remained unchanged.

## Evidence and limits

The public Instagram screen imports an absent local backend, and the corpus cases remain private. The initial invalid screen is not usable evidence. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `blocked_pending_sanitization`.

Public code related to this study:

- [`experiments/active/uembed-instagram-retrieval-v1/instagram_screen.py`](../../../experiments/active/uembed-instagram-retrieval-v1/instagram_screen.py)
- [`experiments/active/uembed-instagram-retrieval-v1/ranking.py`](../../../experiments/active/uembed-instagram-retrieval-v1/ranking.py)

## Revisit condition

Do not rerun the invalid screen; revisit only under the later recorded conditions and a fresh confirmation substrate.
