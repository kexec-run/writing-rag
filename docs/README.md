# Documentation

Start with the system path, then the import guide and experiment record.

| Document | What it covers |
|---|---|
| [Project overview](../README.md) | Implemented capabilities, local setup and validation limits |
| [Importing chats](importing-chats.md) | Supported source shapes, speaker/time handling and privacy |
| [Canonical pipeline](architecture/single-path.md) | Current ingestion, retrieval, generation and verification flow |
| [Qwen and Jina roles](architecture/qwen-jina-reranker-tree.md) | Retrieval, text matching and frozen-pool reranking boundaries |
| [Experiment decisions](architecture/experiment-ledger.md) | Key outcomes, failures, decisions and revisit conditions |
| [Article-generation problem](architecture/article-generation-problem-brief.md) | Why retrieval success has not solved narrative and acceptance failures |
| [Future composition architecture](architecture/future-shape-beat-architecture.md) | Proposed author-intent and evidence flow; not the canonical runtime |
| [Experiment catalog](experiments/README.md) | Sanitized study summaries and complete anonymous artifact register |

## Model and evidence boundaries

- Qwen3-Embedding-0.6B owns corpus retrieval in the current path.
- Jina v5 text matching is symmetric and auxiliary; it does not search the corpus.
- Jina Reranker v3.5 can only reorder candidates already returned by Qwen.
- Content evidence and self-authored style evidence are separate collections.
- An experiment result does not change production policy unless its stated gate passes.

The experiment catalog distinguishes inventory records from runs and independent evaluation cases. Corpus-dependent reproduction requires local data that is not included here.
