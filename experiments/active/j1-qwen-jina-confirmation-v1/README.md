# J1 + Qwen + Jina confirmation

Status: confirmation run completed; aggregate gains were promising but the
predeclared confirmation gate returned `not_confirmed`.

This path tests only the frozen winning architecture:

```text
J1 construction
  -> Qwen3-Embedding-0.6B retrieval
  -> frozen top-50 candidate pool
  -> Jina Reranker v3.5 MLX listwise ordering
  -> top-10 evaluation
```

Qwen remains the retriever. Jina v3.5 is only the reranker. The Jina v5 text
matching model remains inside the existing J1 construction path; it is not a
retrieval backend.

The freeze is split into three independent contracts:

- `architecture_manifest.json`: model revisions, policies, code hashes, and
  ranking configuration;
- `development_substrate_manifest.json`: exact development queries, candidate
  text/order, gold IDs, and hashes;
- `evaluation_manifest.json`: metric definitions, paired bootstrap procedure,
  and confirmation thresholds.

Positive verification requires exact complete ranking repeatability on the
development substrate. Negative verification must reject architecture,
substrate, or evaluation mutations rather than redirecting or silently
continuing.

The confirmation set contained 110 new anchors, with no overlap with the
original development 60. The primary comparison was native J1/Qwen versus
frozen J1/Qwen plus Jina on the same top-50 pools. The original 60 anchors
remain development data only.

The fresh result was:

| Metric | Native Qwen | Qwen + Jina | Delta |
|---|---:|---:|---:|
| Recall@5 | 0.6273 | 0.7182 | +9.09 pp |
| Recall@10 | 0.7364 | 0.8182 | +8.18 pp |
| MRR@50 | 0.5547 | 0.6178 | +0.0631 |
| nDCG@10 | 0.5879 | 0.6595 | +0.0716 |
| Recall@50 | 0.9818 | 0.9818 | 0 |

The gate failed because the Recall@10 bootstrap lower bound was `0.00`, the
exact McNemar result was `p=0.078`, and the p99 subgroup regressed by 10 pp at
Recall@5 and Recall@10. No production policy or article holdout is authorized.

The result decision is one of:

- `confirmed`;
- `inconclusive`;
- `not_confirmed`.

No result in this folder changes the production retrieval policy or authorizes
article-level generation by itself.

The query-level ranking payloads, manifests, and architecture diagram are not
included in this public snapshot. The aggregate metrics and the
[frozen-pool protocol](../qwen-top50-reranker-v2/README.md) are retained here
so the result can be understood without exposing private evaluation data.
