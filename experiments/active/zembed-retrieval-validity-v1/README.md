# zembed MLX retrieval-validity experiment

This is an isolated exploration of `lexrivera/zembed-1-embedding-mlx-6Bit`.
Only the MLX checkpoint is executed. The official ZeroEntropy model card is a
behavioral reference; its weights are not downloaded or loaded.

The experiment validates the query/document embedding contract, then compares
zembed dense and hybrid rankings with the frozen Q0 Qwen held-out results.
Production Qwen vectors and retrieval code are untouched.

## Fixed contract

- same held-out labels, queries, content filters, document IDs, and lexical scores;
- zembed query and document roles remain separate;
- last non-padding hidden state is normalized to a 2,560-dimensional vector;
- dense and hybrid use the existing `0.65 dense + 0.35 lexical` formula;
- no Jina, reranking, graph expansion, prompt generation, or article evaluation;
- no cross-model cosine subtraction; comparisons are rank- and label-based.

## Decision

`direction_supported` requires the model contract to pass and the primary
dense arm to improve both held-out MRR and nDCG@10 by at least `0.03`, while
Precision@10 and Recall@10/50/100 do not decline and hard-negative intrusion
does not increase. This remains exploratory and does not change production.

`directional_only` records a non-marginal signal that misses a full gate;
`not_supported` records no useful signal; a failed contract is
`invalid_mlx_checkpoint_or_adapter`.
