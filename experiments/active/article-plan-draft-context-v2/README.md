# Article-plan draft-context v2

This experiment tests whether bounded, source-text-bearing narrative packets
change drafting under a common strict verifier.

```text
C0: Qwen evidence -> current paragraph draft -> strict common verifier
C1: Qwen evidence + scoped plan packets -> paragraph draft -> same verifier
```

Qwen remains the only corpus retriever. Jina, reranking, graph expansion, and
production defaults are outside this experiment.

## Goal

Support the direction only when the full four-target run shows:

- at least two targets with fewer semantic failures in C1;
- no new unsupported/contradicted failures;
- no new qualifier failures;
- complete claim-to-span trace coverage;
- at least one changed draft.

The two-target smoke is a wiring gate, not a promotion result.

## Required invariants

- C0 and C1 have identical Qwen IDs/order, per-intent evidence IDs,
  source-span IDs, style IDs/text hashes, article plan, verifier contract,
  verifier configuration, model, budgets, and holdout corpus.
- C1 draft packets contain actual source span text, not only identifiers.
- Every packet span, edge, and assertion is inside its intent allowlist.
- MDX packets do not impose message causal-edge requirements.
- Draft plan context is capped at 6,000 measured tokens per group. This is a
  fixed feasibility bound: the first run found a single source span requiring
  5,623 tokens after bounded group rendering, while the full model context is
  12,000 tokens. A cap overflow is invalid, not a generation fallback.
- Prompt overflow, missing provenance, fallback generation, and runtime drift
  fail closed.

## Phases

1. Run the negative checks.
2. Run `mdx-animal` and `message-causal` as the smoke phase.
3. Run all four targets only when smoke passes.
4. Write one decision report and stop.
