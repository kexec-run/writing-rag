# Current pipeline

This page describes the public implementation. Experimental plans and historical results are linked separately.

## Ingestion and indexing

```text
selected local exports
        │
        ▼
source-specific extraction and normalization
        │
        ▼
canonical messages with source, speaker, thread and time metadata
        │
        ▼
conversation windows and retrieval documents
        │
        ├── content_memory: conversation context and documents
        └── style_memory: self-authored messages only
                 │
                 ▼
          Qwen embeddings and separate indices
```

The import and indexing entry points include `scripts/run_pipeline.py`, `scripts/create_canonical_messages.py`, `scripts/create_candidate_windows.py`, `scripts/build_retrieval_documents.py` and `scripts/embed_chunks.py`. The top-level pipeline requires real local inputs; the synthetic demo is a separate plumbing check.

## Retrieval and generation

```text
article query
     │
     ├── Qwen retrieval from content_memory
     └── Qwen retrieval from self-only style_memory
                  │
                  ▼
        generation context and article plan
                  │
                  ▼
       draft generation and claim verification
```

Whole-query content retrieval is the default generation path. Paragraph-mode intent retrieval is an explicit optional path. The public code also contains planning, source-hydration and narrative-reconstruction components, but a narrative map is not a mandatory stage in every canonical generation call.

The [generation contracts and diagnostics](generation-contracts-and-diagnostics.md)
guide explains the implemented optional paths: author-supplied assertion IDs,
hash-checked source hydration, per-intent draft packets, paragraph contracts
and local repair. It also covers retrieval traces, embedding identity and
runtime token accounting. These mechanisms make the pipeline inspectable;
they are separate from experimental claims about article quality.

## Evidence and style rules

- Other speakers contribute conversational context; they are not style examples.
- Self-authored material supplies the style collection.
- Retrieved evidence supports factual claims; a scaffold expresses author intent and composition.
- Shared source IDs alone do not prove a causal relation.
- Causal claims require relation evidence under the applicable verification contract.

## Current limits

The [conversation-context and style record](conversation-context-and-style.md)
documents the executed Instagram experiments: role-labeled context excerpts,
preserved self text, overlapping windows and heuristic links. It separates
those structural and retrieval results from the still-unproven goal of
preserving causal style in generated articles.

The synthetic demo validates import shapes, canonicalization, window construction, content/style separation and retrieval plumbing. It uses hashing and invented examples, so it does not validate semantic retrieval quality or article acceptance. Public experiment scripts may also depend on excluded local modules or private corpora; see the [experiment catalog](../experiments/README.md).

The current result and architecture boundaries are summarized in the [decision ledger](experiment-ledger.md).
