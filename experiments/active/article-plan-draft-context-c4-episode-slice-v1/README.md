# Article-plan draft-context C4: episode slice

C4 is a same-target development experiment for the representation hypothesis
described in `docs/architecture/article-generation-problem-brief.md`. It keeps
the C3 retrieval, anchor candidates, style evidence, plan, verifier, generator
configuration, and four message targets fixed. The only C1-prime delta is the
draft packet selection surface:

```text
C0: frozen C3 C0
C1: frozen C3 C1 exploratory packets
C1-prime: frozen C3 anchor + at most two same-document/thread/window neighbors
```

## Selection contract

`episode_slice_v1` first applies the existing C3 match and overlap eligibility
gate without reranking. It then validates a required frozen anchor contract:

- anchor span and document IDs;
- canonical text hash and rank;
- source thread and window bounds;
- C3 selection-policy hash.

Neighbors are restricted to the anchor's exact document, source thread, and
window. Explicit reply links are preferred; timestamp distance is the fallback
heuristic. At most two neighbors are selected, no slot is force-filled, and
the display is chronological by timestamp and canonical span ID. Causal edges
are retained in the machine trace but enter the draft packet only when every
endpoint is displayed. Monologues are not converted into a self/other pattern.

## Artifact authority and isolation

`anchors.json` is selector input extracted from frozen C3 C1 `repeat-01`
`draft_plan_context_records`. The actual C3 schema stores these records in each
C1 run artifact, so extraction validates that structure directly.

`evaluation/pairs.json` is separately authored evaluation metadata. Selector
code does not import it, and the confirmation runner does not accept an
evaluation-file argument. Pair labels are applied only after packet selection
for audit classification.

## Scope and decision

- four C3 message targets;
- three repeats in the registered contract;
- fixed C1-only episode-slice configuration;
- no C0/C1 regeneration;
- the packets-only harness remains available;
- the full path ran with `--phase full --host-generation`;
- 12/12 C1-prime repeats completed and were scored against frozen C3 C0;
- decision: `not_supported`; the contract passed, but the direction gate failed;
- no production promotion;
- no confirmed status.

Contract failure is `inconclusive`. `direction_supported` is subject to the
full development gate, while `confirmed` is permanently unavailable for this
known-target experiment.

## How to read this run

```text
direction gate: not_supported (1/4 improved, 3/4 semantic regressions, 4/4 grounding regressions)
mechanism: inconclusive (0/2 complete required pairs; interface abstained)
confirmed: false
```

This is not a clean falsification of episode-slice versus independent excerpts:
partial or abstained neighborhoods cannot test the intent-authority hypothesis.
It does not authorize follow-on work on Authorial-target removal, group size,
retrieval, or Jina.

## Artifacts

The private study used a registration manifest, frozen anchors and evaluation
pairs, two experiment scripts, and generated run/compliance/deviance reports.
Those inputs, scripts, and outputs are intentionally omitted from this public
snapshot; their historical names are listed below for provenance only and are
not available reproduction instructions:

- `manifest.json`, `anchors.json`, and `evaluation/pairs.json` (private study
  inputs; not included here);
- `scripts/article_plan_draft_context_c4_episode_slice_artifacts.py` and
  `scripts/article_plan_draft_context_c4_episode_slice_confirmation.py`
  (private scripts; not included here);
- `data/generated/article-plan-draft-context-c4-episode-slice-v1/run-report.json`,
  `plan-compliance.json`, and `plan-deviance.md` (private generated outputs;
  not included here).

## Public research record

Disposition: `results-only`. Private outlines, source plans, assertions and per-target payloads are excluded. Do not infer a general win from an individual accepted or rejected draft.

See the [sanitized study summary](../../../docs/experiments/studies/study-001.md) and [complete experiment catalog](../../../docs/experiments/README.md).

## Public availability

- Catalog disposition: `results-only` ([study-001](../../../docs/experiments/studies/study-001.md)).
- Present in this folder: `README.md` only.
- Omitted: private manifest inputs, anchors, evaluation pairs, runner scripts, generated reports, corpus, and model cache; the preceding artifact names are historical references, not available files.
- Reproduction limit: the public page preserves the selector contract and decision context, not a runnable C4 study.
