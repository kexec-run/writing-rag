# Article-plan draft-context C2 focused retest

This experiment tests one bounded revision after the C1 message confirmation
regressed three of four fresh message targets.

```text
C0: Qwen paragraph retrieval -> legacy draft packet -> common verifier
C1: same retrieval/plan/style/generator/verifier
    -> focused_v1 draft packet -> common verifier
```

The full source allowlist, canonical source spans, article plan, and verifier
remain the same. Only the text presented to the draft model changes.

## Focused packet policy

- derive conservative terms from the intent question, intended claim, role,
  opening, and closing;
- keep canonical spans with at least one intent-term overlap;
- rank by term overlap, retrieval evidence score, exact excerpt match, then
  source order;
- keep at most six spans per intent;
- retain speaker diversity when eligible self and other spans exist;
- show only the selected spans to drafting while retaining every allowlisted
  span in the machine-readable trace;
- show only explicit causal edges in the focused draft prompt;
- abstain explicitly when no candidate meets the threshold; never fall back to
  the complete source packet.

The threshold and cap are fixed in `manifest.json` and cannot be tuned during
the run.

## Verification

Positive checks cover manifest integrity, focused selection, bounded prompt
content, full trace retention, abstention behavior, Metal preflight, identical
Qwen/style/plan/verifier inputs, and 100% C1 plan-span coverage.

Negative checks mutate the manifest hash, policy, span cap, source-plan hash,
packet provenance, and the existing context contract. They must fail closed.

## Scope and interpretation

The same four message targets from the prior development confirmation are
reused intentionally to test the packet mechanism. This is a development
retest, not a new holdout confirmation. Even a positive result requires a new,
time-disjoint confirmation set before production promotion.

Jina, reranking, retrieval changes, graph redesign, and production-default
changes are out of scope.

## Public research record

Disposition: `results-only`. Private outlines, source plans, assertions and per-target payloads are excluded. Do not infer a general win from an individual accepted or rejected draft.

See the [sanitized study summary](../../../docs/experiments/studies/study-001.md) and [complete experiment catalog](../../../docs/experiments/README.md).

## Public availability

- Catalog disposition: `results-only` ([study-001](../../../docs/experiments/studies/study-001.md)).
- Present in this folder: `README.md` only.
- Omitted: private message targets, source plans, model cache, experiment runner, and generated comparison artifacts.
- Reproduction limit: the public record describes a historical context-packet comparison; it cannot reproduce the corpus result.
