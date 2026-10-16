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

## Run with local models

Install the optional model/runtime dependencies and make the Qwen embedding
model available in the Hugging Face cache:

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
contains only synthetic fixtures. The validation performed for this release
reported 154 passing tests, 33 private-data-dependent skips, and 181 existing
Ruff findings. Those results are not a claim that all tests passed or that the
repository is lint-clean. The skips require private data, and the synthetic
demo does not provide semantic-model validation.

## License

The repository code and documentation are released under the MIT License; see
[`LICENSE`](LICENSE). Third-party model weights remain subject to their own
licenses.
