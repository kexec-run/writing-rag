# Cursor rules index

Glob-scoped rules load when matching files are in play. Read this index
instead of opening every `.mdc` file.

| Rule | Globs | Prevents |
|---|---|---|
| `agent-edit-surface.mdc` | always on | Full-file reads of hub scripts; cross-boundary edits |
| `qwen-retrieval.mdc` | retrieve, embed, pipeline | Jina retrieval; mixed vector spaces |
| `jina-auxiliary.mdc` | jina*, reranker experiments | Jina as retriever; batch>1; archived chain-selection retries |
| `generation-verify.mdc` | generate, paragraph, `writing_rag/plan/`, article_plan shims | Allowlist escapes; cross-slot evidence |
| `experiment-isolation.mdc` | experiments/**, writing_rag/**, *experiment* scripts | Production changes from ablations |
| `mlx-runtime.mdc` | generate, embed, jina mlx, model_paths | Silent Metal fallback; sandbox MLX runs |
| `no-promote.mdc` | docs/**, experiment READMEs | Rewriting ledger decisions; promoting pilots |

The detailed experiment ledger and runtime instructions are maintained in the
private working checkout and are intentionally omitted from this public seed.
