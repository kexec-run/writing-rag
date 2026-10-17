# Qwen retrieval and Jina auxiliary paths

Qwen locates evidence in the corpus. Jina experiments operate on text or candidates after Qwen has produced a bounded set.

```text
source documents and conversation windows
                  │
          Qwen document embeddings
                  │
article query ─ Qwen query embedding
                  │
                  ▼
         Qwen corpus retrieval
                  │
          fixed candidate pool
          ┌───────┼────────┐
          │       │        │
          │       │        └── native Qwen ranking
          │       │                 │
          │       │                 └── baseline top-k
          │       │
          │       ├── Jina Reranker v3.5 experiment
          │       │       └── reorder this exact Qwen pool
          │       │           (cannot change Recall@pool-size)
          │       │
          │       └── Jina v5 text-matching experiment
          │               └── symmetric comparison/selection
          │                   among supplied texts only
          │
          └── selected content evidence
                     │
self-only Qwen style retrieval ──┤
                     │
                     ▼
             article generation
                     │
                     ▼
             claim verification
```

The branches are distinct experiments. Jina v5 text matching does not encode the corpus index and does not perform retrieval. Jina Reranker v3.5 receives a frozen Qwen candidate pool and can only change its order.

## Current status

- **Canonical:** Qwen3-Embedding-0.6B corpus retrieval.
- **Auxiliary experiment:** Jina v5 symmetric text matching on supplied texts.
- **Unconfirmed ranking hypothesis:** Jina Reranker v3.5 over frozen Qwen top-50 pools.
- **Not used for retrieval:** Jina retrieval replacement and Jina vectors in the Qwen index.

The 110-anchor reranker confirmation improved aggregate Recall@5 and Recall@10 but failed its predeclared gate, including a p99 regression. It remains `not_confirmed`. See the [decision ledger](experiment-ledger.md) and the [full experiment catalog](../experiments/README.md).
