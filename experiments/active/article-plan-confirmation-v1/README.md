# Article-plan confirmation v1

This is a bounded confirmation pilot for the opt-in article-plan path. It
compares the existing whole-article Qwen paragraph pipeline (`W0`) with the
same pipeline using a source-hydrated, authority-aware article plan (`A1`).

```text
held-out target
    ↓ exclude target documents/messages from every retrieval collection
Qwen content retrieval
    ↓ frozen query, corpus, vectors, and ranking policy
fixed outline + paragraph-mode generation
    ├── W0: no article plan
    └── A1: article plan from the same Qwen-selected evidence
```

Qwen remains the only corpus retriever. Jina text matching and the Jina
reranker are not used here. Style evidence is retrieved once with Qwen and is
copied to both arms.

## Fixed pilot contract

- Four predeclared targets: two message-backed and two MDX-backed.
- One W0 run and one A1 run per target: exactly eight arm runs.
- Target source IDs, target hashes, Qwen index metadata, query policy, frozen
  outline, style query, generator model, token limit, temperature, evidence
  budget, group size, repair budget, and verifier budget are recorded in the
  manifest and run metadata.
- Message targets are removed by canonical `message_ids`; MDX targets are
  removed by `mdx_article` and exact source-document hashes.
- The runner is resumable. A completed arm artifact is reused; a failed arm is
  recorded and the remaining arms continue.
- MLX is fail-closed. The host Metal preflight must pass before the model is
  loaded. No prompt or hashing fallback is allowed.

The runner uses the existing `generate_paragraph_article` implementation with
a frozen outline. In paragraph mode the CLI's global outline/draft branch is
not entered; this is recorded as `paragraph_mode_fixed_outline` rather than
being presented as a full CLI two-pass run. W0 and A1 still use identical
paragraph, verifier, context, and repair settings.

## Positive checks

- Every target has a nonempty frozen outline and a distinct holdout set.
- Target message/document IDs and exact target hashes are absent from all
  documents and vectors passed to W0/A1.
- W0 and A1 have identical initial Qwen content IDs and order, identical
  per-intent evidence IDs, identical style IDs and style text hashes, and the
  same model/runtime/budget contract.
- A1's final refreshed article plan is `ready`, has 100% hashed source-span
  coverage, and validates against `validate_article_plan`.
- Every article-plan claim has only its slot's source, edge, or author
  assertion IDs.
- Causal claims cite explicit allowlisted narrative edges. MDX targets do not
  require message narrative edges.
- The result records claim counts, supported-claim rate, protocol failures,
  unsupported/contradicted/insufficient-evidence failures, qualifier failures,
  article-plan failures, token usage, and generation calls.

## Negative checks

The harness must fail closed for each of these twelve mutations:

1. Add a held-out document ID to the retrieval corpus.
2. Add a held-out message ID to a window's metadata.
3. Add a held-out target text hash to the retrieval corpus.
4. Change one Qwen candidate ID or its order between W0 and A1.
5. Change a style ID between arms.
6. Change a style text hash between arms.
7. Add a source ID from another intent to an article-plan claim.
8. Add a mismatched source-span hash to a packet.
9. Add an unknown source span to a claim.
10. Add an unknown causal edge ID to a causal claim.
11. Add an unknown author assertion ID to an author-asserted claim.
12. Mark a failed Metal/model run as completed or enable implicit fallback.

## Decision gate

`direction_supported` requires all contract checks and negative mutations to
pass, zero holdout leakage, zero Qwen/style drift, zero A1 source/style/cross-
slot escapes, no new unsupported/contradicted/insufficient-evidence or
qualifier failures versus W0, and at least two of four targets with a lower
semantic-failure count in A1. Any target regression, incomplete arm, failed
negative check, or missing MLX proof yields `not_confirmed` or
`inconclusive`; it does not authorize a production switch.

Stop after the decision report. Do not tune the article plan, change Qwen,
start Jina experiments, or redesign the graph in this pilot.
