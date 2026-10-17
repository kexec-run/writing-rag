# Wordwise corpus embeddings

Study ID: `study-038`

Disposition: `results-only`
Decision: 4B path removed from the wordwise experiment; no retrieval claim.

## Question

Do larger embedding models produce more useful semantic neighborhoods for words from the writing corpus?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **2 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The Qwen3-Embedding-4B wordwise run took about 5.25 times longer. The experiment measured vector neighborhoods, not downstream retrieval quality.

## Evidence and limits

A richer neighborhood is not proof of better retrieval or writing. Source words and vectors are not public. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `results_only`.

No corresponding public experiment implementation is included in this seed.

## Revisit condition

Compare actual held-out retrieval and downstream article evidence before spending on a larger model.
