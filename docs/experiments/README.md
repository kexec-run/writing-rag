# Experiment history

This catalog preserves the research paths tested in the private project while keeping conversations, corpus payloads, source-span maps, vectors and raw traces local.

## How to read the inventory

The catalog lists **25 named experiment directories**, **96 excluded Python support/experiment files**, and **222 held generated-artifact directories**, grouped into **38 research lines plus one shared-support group**. No historical corpus/model study is classified as runnable from this snapshot. Two isolated synthetic capabilities are separately verified: one replays frozen ranking metrics, and one checks Shape–Beat composition contracts. Neither executes models or reproduces a study result. The 222 are directory containers: they may hold reports, repeatability summaries, intermediate files, vectors, empty placeholders or multiple executions. They are not 222 runs or independent examples.

The `active/` and `archived/` directory names preserve the historical
organization of the research; they do not indicate public execution support
or current project status. Use each catalog disposition and its local
experiment README to determine what is actually present. Code retained for a
`results-only` study is historical or illustrative unless a complete public
bundle is later verified. `pending sanitization` means the folder must not be
treated as a reproducible public experiment.

`results-only` means a sanitized historical result can be discussed, but the public checkout does not promise a reproducible corpus experiment. `pending sanitization` means dependencies, evidence or source material still prevent a safe and dependable public execution path. No historical experiment is classified `runnable` by this inventory.

Validity and decision are separate. A result can be `invalid`, `superseded`, `interrupted`, `structural_only`, `not_confirmed` or `not_promoted`; none of these states is converted into a positive finding by inclusion in the catalog.

## Research lines

| ID | Research line or support group | Disposition | Held artifact directories | Decision |
|---|---|---|---:|---|
| [`study-001`](studies/study-001.md) | Article planning and scoped draft context | `results-only` | 19 | experimental; no article-plan path is promoted as the default |
| [`study-002`](studies/study-002.md) | Embedding runtime and Apple Neural Engine probes | `results-only` | 6 | remove the conversion path; no ANE backend or fallback |
| [`study-003`](studies/study-003.md) | Generation engine and retry ablations | `results-only` | 11 | duplicate retry variants were removed from the reusable path |
| [`study-004`](studies/study-004.md) | Graph-aware article retrieval | `results-only` | 9 | removed from production retrieval |
| [`study-005`](studies/study-005.md) | Holdout and near-black repeat studies | `results-only` | 1 | diagnostic record only; no independent promotion claim |
| [`study-006`](studies/study-006.md) | Instagram conversation-chain construction | `pending sanitization` | 1 | keep the filtered chain pilot exploratory; production ingestion is unchanged |
| [`study-007`](studies/study-007.md) | Instagram conversation episodes | `pending sanitization` | 15 | exploratory; no production ingestion change |
| [`study-008`](studies/study-008.md) | Jina selection for Instagram chains | `results-only` | 3 | structural_only; not promoted for chain selection |
| [`study-009`](studies/study-009.md) | Automatic conversation-role and label directions | `results-only` | 1 | archive the human-label-gated direction; automatic labels remain hypotheses |
| [`study-010`](studies/study-010.md) | Interpretation and claim-boundary audits | `pending sanitization` | 4 | diagnostic tooling only; no general semantic-quality claim |
| [`study-011`](studies/study-011.md) | Qwen top-50 with Jina reranker confirmation | `results-only` | 1 | not confirmed; no production or article-holdout promotion |
| [`study-012`](studies/study-012.md) | Jina model and task-mode characterization | `results-only` | 13 | retain Jina only in explicitly bounded auxiliary roles |
| [`study-013`](studies/study-013.md) | Jina retrieval replacement attempts | `results-only` | 3 | remove Jina retrieval replacement; Qwen remains the corpus retriever |
| [`study-014`](studies/study-014.md) | Jina symmetric text matching | `results-only` | 6 | direction supported on the frozen pilot; auxiliary use only |
| [`study-015`](studies/study-015.md) | Load-bearing language and semantic coherence | `results-only` | 2 | diagnostic; no vocabulary simplification rule promoted |
| [`study-016`](studies/study-016.md) | Evidence locality and backfill diagnostics | `results-only` | 4 | diagnostic artifacts; no canonical-path change |
| [`study-017`](studies/study-017.md) | MDX content ingestion and retrieval ablations | `results-only` | 15 | MDX ingestion is retained; global ranking did not demonstrate an article-output change |
| [`study-018`](studies/study-018.md) | Narrative-map smoke and packet experiments | `results-only` | 3 | experiment hook only; not a canonical production stage |
| [`study-019`](studies/study-019.md) | Narrative reconstruction and source hydration | `results-only` | 1 | research direction only; canonical generation remains separate |
| [`study-020`](studies/study-020.md) | Near-black article scaffolds and writing trials | `results-only` | 12 | retain the evidence-first fallback pattern; article generation remains unresolved |
| [`study-021`](studies/study-021.md) | Local knowledge-base prototype | `pending sanitization` | 1 | no public result asserted |
| [`study-022`](studies/study-022.md) | Paragraph-level retrieval and drafting | `results-only` | 10 | not promoted as a replacement for the canonical global retrieval path |
| [`study-023`](studies/study-023.md) | Narrative-pattern and scaffold ablations | `results-only` | 2 | scaffold-guided generation remains a direction; no minimum scaffold promoted |
| [`study-024`](studies/study-024.md) | Personal knowledge-base retrieval prototype | `pending sanitization` | 1 | no public result asserted |
| [`study-025`](studies/study-025.md) | Qwen embedding characterization | `results-only` | 16 | corrected 22-case pilot retained as exploratory characterization |
| [`study-026`](studies/study-026.md) | Qwen retrieval with bounded Jina text selection | `results-only` | 1 | Qwen remains retrieval; A0 remains default; Jina is an auxiliary experiment |
| [`study-027`](studies/study-027.md) | Qwen query-encoding policy comparison | `results-only` | 1 | Q0 remains the measured baseline; no Q1 improvement claimed |
| [`study-028`](studies/study-028.md) | Frozen Qwen top-50 reranking | `results-only` | 3 | reranking is a bounded ranking hypothesis; confirmation was not met |
| [`study-029`](studies/study-029.md) | Retrieval-quality diagnostics | `results-only` | 10 | diagnostic series; later frozen studies govern current conclusions |
| [`study-030`](studies/study-030.md) | Scaffold with Qwen and Jina selection | `results-only` | 2 | no scaffold minimum or Jina default promoted |
| [`study-031`](studies/study-031.md) | MDX holdout scaffold ablation | `results-only` | 1 | partial/interrupted; no promotion conclusion |
| [`study-032`](studies/study-032.md) | Whole-article versus per-slot retrieval | `results-only` | 3 | not promoted; global Qwen retrieval remains the current path |
| [`study-033`](studies/study-033.md) | Shape–Beat composition architecture | `results-only` | 17 | pilot_complete_no_promotion; production retrieval unchanged |
| [`study-034`](studies/study-034.md) | Shared pipeline and experiment-support scripts | `pending sanitization` | 0 | not an independent experiment family |
| [`study-035`](studies/study-035.md) | Evidence trace and provenance diagnostics | `pending sanitization` | 4 | private diagnostics retained; no trace payload published |
| [`study-036`](studies/study-036.md) | Typed article distillation | `results-only` | 6 | validated section fallback works in the recorded run; continuity pass is not accepted |
| [`study-037`](studies/study-037.md) | UEmbed screening and reranking | `pending sanitization` | 1 | screening was invalid; later fused retrieval remained not confirmed |
| [`study-038`](studies/study-038.md) | Wordwise corpus embeddings | `results-only` | 2 | 4B path removed from the wordwise experiment; no retrieval claim |
| [`study-039`](studies/study-039.md) | Embedding retrieval validity and source-homogeneous checks | `results-only` | 11 | Qwen baseline retained; no production backend change from this family |

## Complete machine-readable register

[`catalog.json`](catalog.json) contains one anonymous record for each named experiment directory, excluded Python support/experiment file and held artifact directory. It includes no original private paths, content hashes, source IDs or payload text. A held-directory record has `execution_count: null` because a directory alone cannot establish how many executions it contains.

## Evidence policy

Quantitative statements in the study pages are transcribed from the sanitized decision ledger and labeled as reported results. This documentation pass did not recompute private-corpus metrics. Where no decision record supports an outcome, the page says so instead of inferring one from filenames or artifact counts.

Reproduction of corpus metrics requires the private corpus and some excluded local dependencies. The synthetic demo validates import and pipeline plumbing; it does not reproduce these semantic results.
