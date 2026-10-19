# Experiments

This is the guide to experiment files retained in the public repository. Start
with the [selected findings](../README.md#selected-research-findings) and
[ablation comparisons](../README.md#ablation-studies) for results, or the
[complete research catalog](../docs/experiments/README.md) for studies whose
implementation is not included here. Reusable runtime code lives in
[`writing_rag/`](../writing_rag/), with command-line entrypoints in
[`scripts/`](../scripts/).

## What can run publicly

Two isolated experiment demonstrations use invented fixtures and no models:

- [Frozen-ranking replay](active/qwen-top50-reranker-v2/README.md#public-synthetic-replay): build a candidate fixture, replay supplied rankings, compare metrics, and verify repeatability.
- [Shape–Beat validation](active/shape-beat-axis0-v1/README.md#public-synthetic-validation): validate two composition cases, their schedules, and negative mutations.

The [synthetic import/retrieval/context demo](../README.md#try-the-synthetic-demo)
exercises the main pipeline. These demonstrations do not reproduce historical
corpus or model results.

## Retained directories and result summaries

All 17 directories are listed below. The four article-plan/context variants
share a consolidated report; separate quantitative outcomes are not available
for each variant in the public record.

| Directory | Public result summary | Public availability |
|---|---|---|
| [Article-plan confirmation](active/article-plan-confirmation-v1/) | [Article planning and context](../docs/experiments/studies/study-001.md) | Results only |
| [C2 focused context](active/article-plan-draft-context-c2-focused-v1/) | [Article planning and context](../docs/experiments/studies/study-001.md) | Results only |
| [C4 episode-slice context](active/article-plan-draft-context-c4-episode-slice-v1/) | [Article planning and context](../docs/experiments/studies/study-001.md) | Results only |
| [Draft context v2](active/article-plan-draft-context-v2/) | [Article planning and context](../docs/experiments/studies/study-001.md) | Results only |
| [Instagram chain retrieval](active/instagram-chain-retrieval/) | [Chain construction](../docs/experiments/studies/study-006.md) | Pending sanitization |
| [Instagram conversation episodes](active/instagram-conversation-episodes/) | [Conversation episodes](../docs/experiments/studies/study-007.md) | Pending sanitization |
| [J1/Qwen/Jina confirmation](active/j1-qwen-jina-confirmation-v1/) | [Reranker confirmation](../docs/experiments/studies/study-011.md) | Results only |
| [Local knowledge-base prototype](active/obsidian-kb-poc-v1/) | [Local knowledge base](../docs/experiments/studies/study-021.md) | Pending sanitization |
| [Qwen/Jina text selection](active/qwen-jina-text-selection/) | [Bounded text selection](../docs/experiments/studies/study-026.md) | Results only |
| [Qwen top-50 reranker](active/qwen-top50-reranker/) | [Frozen-pool reranking](../docs/experiments/studies/study-028.md) | Results only |
| [Qwen top-50 reranker v2](active/qwen-top50-reranker-v2/) | [Frozen-pool reranking](../docs/experiments/studies/study-028.md) | Results only; synthetic replay available |
| [Shape–Beat Axis 0](active/shape-beat-axis0-v1/) | [Composition architecture](../docs/experiments/studies/study-033.md) | Results only; synthetic validator available |
| [UEmbed screening](active/uembed-instagram-retrieval-v1/) | [Screening and reranking](../docs/experiments/studies/study-037.md) | Pending sanitization |
| [Embedding retrieval validity](active/zembed-retrieval-validity-v1/) | [Validity and source checks](../docs/experiments/studies/study-039.md) | Results only |
| [Embedding runtime / ANE](archived/embedding-runtime/) | [Runtime and ANE probes](../docs/experiments/studies/study-002.md) | Results only |
| [Jina Instagram-chain selection](archived/instagram-chain-selection-jina-v1/) | [Chain selection](../docs/experiments/studies/study-008.md) | Results only |
| [Jina model characterization](archived/jina-model-characterization/) | [Task-mode characterization](../docs/experiments/studies/study-012.md) | Results only |

## Reading the historical layout

`active/` and `archived/` preserve the historical organization. Their names do
not establish whether work is current, reproducible, successful, or supported
in this checkout. Use each directory's README and its catalog disposition.

- **Results only:** a sanitized historical finding is documented; retained code may be illustrative or depend on private inputs and excluded modules.
- **Pending sanitization:** the retained files form an incomplete public bundle and must not be treated as a runnable experiment.
- **Synthetic capability:** separately verified commands using invented data; this does not change the historical study's disposition or decision.

Public summaries preserve reported findings and limitations. Original case
payloads, author assertions, source-span maps, model weights, vectors, and raw
run reports remain local. A summary link is not a promise that the private run
can be reproduced from this checkout.
