# Archived Jina model-characterization paths

This page summarizes earlier Jina experiments that tried retrieval
replacement, retrieval-mode characterization, and standalone mode comparisons.
Their experiment-specific runners, tests, and detailed result artifacts remain
in the private checkout and are intentionally omitted from this public seed.

The current boundary is narrower:

```text
Qwen retrieval -> fixed Qwen candidate pool -> optional Jina text matching
```

Jina does not search the corpus, encode retrieval queries/documents, replace
Qwen vectors, or share the Qwen index. The current bounded contract is
[`scripts/jina_text_selection.py`](../../../scripts/jina_text_selection.py),
with its active experiment manifest under
[`experiments/active/qwen-jina-text-selection`](../../active/qwen-jina-text-selection).

## Archived paths

- Jina retrieval characterization and replacement, including MLX variants;
- Jina retrieval/text-matching/mode comparison and validation runners;
- the associated optional tests.

The bullets describe omitted historical work; they are not files available in
this snapshot. The reusable selector interface linked above is included.

The private experiment ledger and detailed run artifacts are not included in
this public snapshot.

## Public research record

Disposition: `results-only`. Small-pilot measurements are not broad retrieval evidence. Vectors from different task modes are not interchangeable.

See the [sanitized study summary](../../../docs/experiments/studies/study-012.md) and [complete experiment catalog](../../../docs/experiments/README.md).

## Public availability

- Catalog disposition: `results-only` ([study-012](../../../docs/experiments/studies/study-012.md)).
- Present in this folder: `README.md`, `compare_jina_mode_runs.py`, `compare_jina_retrieval_runs.py`, `compare_jina_text_matching_runs.py`, and `finalize_jina_mode_report.py`.
- Omitted: the experiment-specific runner/test suite, model weights/cache, private corpus, and detailed mode-comparison artifacts. The selector source linked above is in the repository's `scripts/` folder, not this experiment folder.
- Reproduction limit: the scripts are historical comparison helpers and do not form a complete standalone model-characterization bundle.
