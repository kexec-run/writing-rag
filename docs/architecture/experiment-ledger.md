# Experiment decision ledger

This is a compact record of the results that determine current model and architecture boundaries. The [experiment catalog](../experiments/README.md) lists all 25 named experiment directories, 96 excluded Python support/experiment files and 222 held artifact directories. Artifact-directory counts are not run counts or independent sample sizes.

The aggregate values below are transcribed from the private sanitized decision ledger. The documentation pass did not recompute corpus metrics; case payloads and raw traces remain private.

| Study | Recorded result | Decision |
|---|---|---|
| Qwen Q0 vs card-style Q1 query encoding | MRR `0.8000` vs `0.7750` on the frozen pilot | Q0 remains the measured baseline; no general claim about query instructions |
| Jina Q2 retrieval | MRR `0.7750` | Do not replace Qwen retrieval |
| Jina Q3 text matching | MRR `0.8500`, AP `0.8458`, 14/20 rank-1 positives, five hard-negative failures | Direction supported on the pilot; auxiliary use only |
| Qwen retrieval plus Jina text selection | MRR `0.8000 → 0.8500`; nDCG@10 `0.8586 → 0.8884`; both articles rejected | Keep Qwen retrieval and A0 default; selection is experimental |
| Instagram J1 chain selection | Positive vs C1 in 3/5 folds; p95 MRR change `-0.050244` | `structural_only`; not promoted |
| Qwen top-50 plus Jina reranker confirmation | 110 unseen anchors; R@10 `0.7364 → 0.8182`, R@5 `0.6273 → 0.7182`, 15 rescues / 6 damages; p99 regressed 10 pp | `not_confirmed`; no production change |
| Scaffold ablation | B2-lite had 3 issues; B2 full with Jina had 6; all corrected variants were rejected | Do not promote a minimum scaffold from issue counts |
| Whole-query vs per-slot retrieval | Both tested targets remained rejected; claim failures differed by arm | Keep global Qwen retrieval as current path |
| Graph expansion | Zero expansions in the article run; no output change | Removed from production retrieval |
| MDX source ablation | Added prose ranked below usable cutoff; six outputs were byte-identical | Ingestion alone did not change retrieval or generation in this test |
| Instagram chain heuristics | Some overmerging; proxy retrieval did not beat baseline | Filtered chain work remains exploratory |
| Human-label direction screen | Human labels unavailable; pseudo-labels are not independent ground truth | Do not claim human-validated labels |
| Wordwise Qwen 4B | About 5.25× slower in the wordwise run; no retrieval-quality test | No case for 4B on this evidence |
| Core ML/ANE conversion | Numerical comparison passed; ANE placement not demonstrated | No ANE backend claim |
| Near-black two-pass distillation | Section pass succeeded; continuity failed and fell back | Keep fallback behavior; no accepted continuous article result |
| Shape–Beat Axis 0/1A | Axis 0 checks passed; Axis 1A pilots exposed verifier/generator limits | Pilot only; no general promotion |
| UEmbed retrieval | Directional gains did not meet confirmation gates | Bounded auxiliary only; production unchanged |
| Qwythos marker formatting | Exact marker/figure syntax passed its narrow held-out test | Narrow syntax capability only; not claim grounding |

## Architecture decision

```text
Qwen owns corpus retrieval
Jina text matching compares supplied texts only
Jina reranking may reorder a frozen Qwen pool only
article generation remains gated by evidence and claim verification
```

No retrieval, chain-construction or generator experiment becomes canonical merely because one aggregate metric improves. Preserve subgroup regressions, failed checks, uncertainty and invalid-harness findings in the study summaries.

## Revisit policy

Reopen a decision only under the condition recorded in its [study summary](../experiments/README.md). Do not use repeated directories as independent cases or substitute proxy labels for human truth. The catalog's complete anonymous artifact register preserves coverage without publishing corpus payloads, source identifiers or private paths.
