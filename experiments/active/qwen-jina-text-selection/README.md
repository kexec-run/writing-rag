# Qwen retrieval with Jina text selection

Status: completed; retained as an inconclusive bounded ablation.

Qwen remains the only corpus retriever. Jina is allowed to compare or select
texts only after Qwen has returned a fixed candidate pool.

## Conditions

```text
A0: Qwen retrieval only
A1: Qwen retrieval -> Jina v5 Small MLX text-matching selection
```

Hold fixed:

- canonical messages and candidate windows;
- Qwen document and query vectors;
- Qwen candidate-pool size;
- labels and hard negatives;
- generation model, prompts, and evidence budget;
- provenance and acceptance validators.

## Positive contract

- Every Jina-selected document ID exists in the Qwen candidate pool.
- Jina uses the symmetric `text-matching` task with no query/document prompt.
- Batch size 1 preserves order symmetry.
- Qwen scores and ranks remain available beside Jina scores.
- Disabling Jina reproduces A0 exactly.

## Negative contract

The experiment must reject:

- Jina corpus-wide search;
- Jina retrieval query or passage prompts;
- Jina candidates absent from the Qwen pool;
- mixed Qwen/Jina vector indexes;
- raw cosine subtraction across model spaces;
- similarity being treated as factual or causal support;
- silent fallback to Qwen, hashing, or another model.

## Adoption gate

This remains an experiment until it improves MRR by at least `0.02` or nDCG
by at least `0.03`, without reducing recall, increasing hard-negative failures,
or reducing downstream grounded-generation acceptance. A smaller change is
recorded as marginal/inconclusive.

## Historical private-checkout runner

The command below documents the private-checkout experiment. Its runner and
generated result files are intentionally omitted from this public seed, so it
is not executable from this snapshot. Reproduction requires the private
checkout and its local model cache.

```bash
/bin/zsh -lc 'cd /path/to/writing-rag && /opt/homebrew/bin/python3 scripts/qwen_jina_article_ablation.py --model-cache models/weights/huggingface --output data/generated/qwen-jina-article-ablation-v1'
```

The runner also evaluates the fixed Qwen local pools from the frozen pilot. It
writes A0/A1 article outputs, Jina selection traces, pilot metrics, and a
comparison report. A successful article run alone cannot promote Jina: the
retrieval gate and downstream grounded-generation gate must both pass.

## Observed result

The host-only MLX run completed with the real Metal probe passed. Qwen remained
the only corpus retriever; Jina reordered only IDs already present in the
32-item Qwen pool.

| Condition | MRR | AP | nDCG@10 | Rank-1 positives | Hard-negative failures |
|---|---:|---:|---:|---:|---:|
| Qwen pool order | 0.8000 | 0.8083 | 0.8586 | 12/20 | 8 |
| Qwen -> Jina selection | 0.8500 | 0.8458 | 0.8884 | 14/20 | 5 |

Jina changed 7 of 22 pilot local orderings. The article comparison used the
same generator, prompts, evidence budget, and validators. A0 was rejected with
14 validation issues; A1 was rejected with 2 validation issues. A1 therefore
shows a downstream improvement in this run, but neither article passed the
acceptance gate, so the experiment does not authorize changing the production
path. Keep Qwen retrieval and A0 as the default.

Private result location: `data/generated/qwen-jina-article-ablation-v1/comparison.md`
(not included in this public snapshot).

## Scaffold ablation

The follow-up scaffold test kept the same Qwen pool and bounded Jina selector,
then varied only the narrative scaffold. The first run was retained as a
harness diagnostic because the scaffold prompt allowed claim types to appear as
outline roles and the 900-token draft cap truncated some variants. The corrected
v2 run added the active system role vocabulary and used fixed 500-token outline
and 1,100-token draft budgets.

| Variant | Selector | Status | Words | Validation issues |
|---|---|---|---:|---:|
| B1 roles/questions | Jina | rejected | 658 | 4 |
| B2 full | Qwen | rejected | 746 | 13 |
| B2 full | Jina | rejected | 725 | 6 |
| B2-lite | Jina | rejected | 860 | 3 |
| Load-bearing semantic | Jina | rejected | 745 | 17 |

B2-lite was the closest scaffold condition in this run, but it still failed
validation. The result is not a case against B2: retrieval was still performed
once for the whole article rather than separately for each scaffold slot. The
next test should use Qwen per-slot retrieval, followed by Jina reordering inside
each slot's Qwen pool. No production policy is changed by this ablation.

Historical private-checkout runner: `scripts/scaffold_jina_article_ablation.py`
(not included in this public snapshot).

Private result locations: `data/generated/scaffold-jina-article-ablation-v1/comparison.md`
and `data/generated/scaffold-jina-article-ablation-v2/comparison.md` (not
included in this public snapshot).
