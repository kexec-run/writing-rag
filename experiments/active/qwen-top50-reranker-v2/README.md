# Qwen top-50 reranker v2

Status: active frozen-pool protocol for native Qwen versus Jina Reranker v3.5.

Qwen owns retrieval and the exact top-50 candidate pool. Jina v3.5 may only
listwise-reorder candidates already in that pool. It cannot add documents,
change Recall@50, or retrieve style evidence.

## Entrypoints

- `freeze_qwen_top50.py` — freeze Qwen pools and metadata
- `rerank_frozen_pools.py` — Jina v3.5 listwise reorder of frozen pools
- `compare_rerankers.py` — paired comparison helpers

## Related

- Confirmation experiment: [../j1-qwen-jina-confirmation-v1/README.md](../j1-qwen-jina-confirmation-v1/README.md)
- Architecture boundary: Qwen owns retrieval and the frozen candidate pool;
  Jina can only reorder that pool. The architecture diagram and local
  per-query comparison artifact are omitted from this public snapshot.

The 110-anchor confirmation remained `not_confirmed`. Do not promote Jina v3.5
or authorize article holdout from this folder alone.
