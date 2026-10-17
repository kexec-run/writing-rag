# writing-rag

A local writing-memory prototype that turns chat and transcript exports into
searchable, provenance-bearing evidence while keeping conversational context
separate from self-authored style examples.

```text
selected local exports
  -> canonical messages and conversation windows
  -> content_memory + self-only style_memory
  -> Qwen retrieval
  -> grounded context for local article generation and verification
```

## What is implemented

- Import adapters for Instagram, WhatsApp, Reddit/Matrix-style chat, Whisper,
  ChatGPT, and Twitter inputs.
- Canonical messages with source, thread, timestamp, speaker, and provenance.
- Structure-aware conversation windows that retain message IDs and time bounds.
- Separate content and style collections; only self-authored messages enter the
  style collection.
- Qwen3-Embedding-0.6B as the corpus retrieval model. Jina is not a corpus
  retriever; its text-matching and reranking roles are isolated experiments over
  Qwen-provided candidates.
- Local generation with evidence and claim checks. Article quality and causal
  narrative reconstruction remain areas for evaluation; the synthetic demo is
  plumbing validation, not a semantic-quality result.

## Try the synthetic demo

Requires Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run_synthetic_demo.py
```

The demo uses invented source-shaped fixtures and explicit hashing. It imports
messages, builds conversation windows and separate content/style documents,
retrieves examples, and assembles a generation context. It does not download or
run a model. This validates data plumbing only; it does not measure semantic
retrieval quality or article quality. It refuses to run if `data/raw/` contains
anything except the exact synthetic fixtures. Generated artifacts stay under
ignored `data/`.

Two additional model-free experiment demonstrations are available:

- [Frozen ranking replay](experiments/active/qwen-top50-reranker-v2/README.md)
  exercises ranking metrics on invented candidate pools; it does not run Qwen
  or Jina and makes no model-quality claim.
- [Shape–Beat validation](experiments/active/shape-beat-axis0-v1/README.md)
  validates two synthetic composition cases and their negative mutations; it
  does not reproduce the private pilot or establish article-generation quality.

## Run with local models

Install the optional model/runtime dependencies and make the Qwen embedding
model available in the Hugging Face cache. This operational path also requires
your own imported corpus and locally generated retrieval documents/vectors; it
is not one of the clean-clone synthetic demonstrations above. Without those
inputs and weights, the model pipeline is not expected to run:

```bash
python -m pip install -r requirements-qwen.txt
export WRITING_RAG_MODEL_CACHE="$PWD/models/weights/huggingface"
python scripts/run_pipeline.py --embedding-backend qwen3
```

The cache path should be a Hugging Face cache root with its normal `hub/` and
snapshot layout. Override it for your machine using
`WRITING_RAG_MODEL_CACHE`. Model weights and all corpus-derived artifacts are
local machine state and must not be committed. The repository's MIT license
does not change the separate licenses that apply to model weights.

## Import formats and privacy

See [`docs/importing-chats.md`](docs/importing-chats.md) for supported input
shapes, identity/time-zone configuration, provenance behavior, and the
synthetic demo. Real exports, embeddings, indexes, and generated traces can
contain private information. Keep them local; the publication checker is only
a guardrail and does not replace file-by-file review.

## Experiments and limits

Experiments are kept separate from the canonical Qwen retrieval path. Published
summaries must distinguish implemented behavior from exploratory results and
must not include corpus payloads or private evaluation traces. Reproducing
corpus-based metrics requires the private source corpus; this public snapshot
contains only synthetic fixtures. The initial public seed's validation counts
(154 passing tests and 181 Ruff findings) are historical. For the current
release candidate, the default suite reported 177 passed, 33 private-data
skips, and 6 subtests passed. The documentation validator passed 21 negative
fixtures. Repository-wide Ruff reported 179 findings, with no new file/rule
diagnostics relative to the reviewed initial commit; the changed D1/D2 Python
files are clean. The D1 ranking replay and both D2 synthetic cases passed, and
repeated D1 rankings and D2 outputs matched exactly. These checks do not make
the repository lint-clean or validate semantic-model or article quality. A
separate optional experiment test collection still has two import errors
because excluded local Qwen characterization modules are unavailable;
private-data skips and that optional collection blocker are not hidden by the
passing default suite.

The public [experiment catalog](docs/experiments/README.md) records 38 research
lines, 25 named experiment directories, 96 excluded Python support or experiment
files, and 222 held artifact directories. These are inventory counts, not
independent runs or samples. Each study separates implementation disposition
from scientific outcome and marks results transcribed from the private decision
ledger.

An optional experiment-test collection produced two errors because excluded
local Qwen characterization modules are unavailable. Static inspection found
11 imports from public experiment/test files that resolve only to excluded local
modules. Some runners also require private corpora or generated artifacts. The
catalog records known blockers; this release step did not restore those paths.

The canonical retrieval path uses Qwen. Jina text matching remains auxiliary,
and the Jina reranker result is unconfirmed. Ranking gains have not yet produced
a repeatable accepted article across the tested scaffold comparisons. See the
[current architecture](docs/architecture/single-path.md), [model roles](docs/architecture/qwen-jina-reranker-tree.md), and [article-generation problem brief](docs/architecture/article-generation-problem-brief.md).

## License

The repository code and documentation are released under the MIT License; see
[`LICENSE`](LICENSE). Third-party model weights remain subject to their own
licenses.
