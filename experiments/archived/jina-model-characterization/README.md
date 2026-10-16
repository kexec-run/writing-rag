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
