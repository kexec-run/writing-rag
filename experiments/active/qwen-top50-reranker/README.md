# Qwen top-50 reranker (v1 diagnostic)

Status: active diagnostic helper for frozen Qwen top-50 pools.

This path reranks already-frozen Qwen candidate pools for B0/C1/J1 episode
conditions. It does not call Jina, does not change Recall@50, and does not
reopen the archived Instagram Jina chain-selection decision.

Entrypoint: `rerank_frozen_top50.py`.

Prefer the v2 frozen-pool protocol for current confirmation work:

- [../qwen-top50-reranker-v2/README.md](../qwen-top50-reranker-v2/README.md)
- [../j1-qwen-jina-confirmation-v1/README.md](../j1-qwen-jina-confirmation-v1/README.md)

The v2 protocol and its current decision are summarized in the
[Qwen top-50 reranker v2 README](../qwen-top50-reranker-v2/README.md).

## Public research record

Disposition: `results-only`. A reranker cannot recover a relevant item missing from top-50. Aggregate gains cannot override subgroup regression or the predeclared gate.

See the [sanitized study summary](../../../docs/experiments/studies/study-028.md) and [complete experiment catalog](../../../docs/experiments/README.md).

## Public availability

- Catalog disposition: `results-only` ([study-028](../../../docs/experiments/studies/study-028.md)).
- Present in this folder: `README.md` and `rerank_frozen_top50.py`.
- Omitted: frozen candidate pools, private relevance labels, model weights/cache, and run outputs.
- Reproduction limit: the script is a historical diagnostic entrypoint; the private pools required for the recorded study are unavailable here.
