# Importing chats and running the synthetic demo

This repository's import pipeline preserves source messages, projects them into
canonical records and conversation windows, then creates separate content and
self-only style collections. Raw exports and all generated artifacts belong in
the ignored local `data/` directory; never commit real exports or derived chat
payloads.

## Prepare exports

Acquire exports through the source service's own export/download flow. Export
availability and privacy settings vary by service. If an export is an archive,
extract it locally and place only the supported source files in the paths below.
The pipeline does not convert arbitrary HTML, PDF, screenshots, or proprietary
database formats. Convert those outside this project to one of the documented
shapes, or expect an explicit unsupported/missing-input error.

| Source | Input location and shape | Current thread behavior |
|---|---|---|
| Instagram | `data/raw/instagram/message_*.json`; each file has a `messages` list with `timestamp_ms`, `sender_name`, and `content` | Current loader assigns every file to `instagram_default`; use one conversation per demo/import run. Separate IDs are not yet supported by this loader. |
| WhatsApp | `data/raw/whatsapp/*.txt`; exported text lines such as `01/01/24, 12:00 - Sender: message`; continuation lines attach to the preceding message | Each file stem becomes a thread ID. Media placeholders are omitted. |
| Reddit/Matrix-style chat | `data/raw/reddit/reddit_chat.json`; array (or supported event list) with `type`, `room_id`, `event_id`, `origin_server_ts`, `sender`, and `content.body` | `room_id` is retained as the thread ID. Deleted/non-message events are skipped. |
| Whisper | `data/raw/whisper/transcript.json`; `transcription` entries with `text` and offsets in milliseconds | One transcript thread; segments are source-native self narration. |
| ChatGPT | Raw export JSON under `data/raw/chatgpt/`; normalization runs only when matching inputs exist | Branches are represented as separate paths by the ChatGPT normalizer. |
| Twitter | `twitter_no_metadata.json` and `twitter_metadata.json` under `data/raw/twitter/` | Both files and `WRITING_RAG_TWITTER_SELF_SCREEN_NAME` are required for optional normalization/graph processing. |

For source-specific author identity, set `WRITING_RAG_SELF_SENDERS` to a JSON
object mapping source names to sender-name arrays. For example:

```bash
export WRITING_RAG_SELF_SENDERS='{"instagram":["Your export display name"],"whatsapp":["Your chat name"],"reddit":["@your-account:server"]}'
```

Names are normalized for matching, but must correspond to the sender values in
the source export. Whisper's `self` identity is source-native and needs no entry.
WhatsApp timestamps use `WRITING_RAG_WHATSAPP_TIMEZONE`, an IANA zone such as
`Asia/Kolkata`; it defaults to `UTC`.



Twitter uses a separate setting because its export normalizer runs before canonical-message projection:

```bash
export WRITING_RAG_TWITTER_SELF_SCREEN_NAME="your_screen_name"
```

The Twitter normalizer stops with an explicit error when this identity is absent; it does not guess which account is yours.
## Run against your local exports

Create an environment and install the base package, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python3 scripts/run_pipeline.py --embedding-backend qwen3
```

The canonical path requires Qwen3-Embedding-0.6B and the optional dependencies
in `requirements-qwen.txt`; install them with `python -m pip install -r
requirements-qwen.txt`. Set `WRITING_RAG_MODEL_CACHE` to the Hugging Face cache
root that contains the normal `hub/` and snapshot layout. To inspect import and
chunking without model weights, explicitly choose the diagnostic hashing backend:

```bash
python3 scripts/run_pipeline.py --embedding-backend hashing
```

Hashing is deterministic plumbing for testing, not semantic retrieval evidence.
Pipeline outputs are written under ignored `data/`, including canonical
messages, candidate windows, retrieval documents, and embeddings.

## Run invented data end to end

The fixtures in `tests/fixtures/synthetic_import/` are invented examples in
source-shaped formats. The preparation script refuses to overwrite existing
raw files. Starting from a checkout with no `data/raw/` inputs:

```bash
python3 scripts/run_synthetic_demo.py
```

This runs the existing importer, chunker, content/style projection, hashing
embeddings, retrieval, and prompt-context assembly. It does not download a
model, perform model inference, or demonstrate Qwen semantic quality. It leaves
the generated synthetic artifacts under ignored `data/` for inspection.

To prepare only source files, without running the pipeline:

```bash
python3 scripts/prepare_synthetic_demo.py
```


The included portable tests run by default. Tests tied to local oracle fixtures are marked `private_data` and skipped; private experiment suites and their corpus-linked runners are omitted from this snapshot:

```bash
python3 -m pytest -q
```

Only opt in after restoring the private corpus and experiment artifacts locally:

```bash
python3 -m pytest --run-private-data-tests -q
```

## Speaker, time, provenance, and content/style rules

Canonical messages retain source, source thread ID, raw message ID, timestamp,
speaker class, sender, content, mode, and source path. Instagram and WhatsApp
timestamps are milliseconds; WhatsApp's local wall time is interpreted using
the configured IANA timezone. Windows preserve message IDs and time boundaries.

`content_memory` contains self messages together with conversational context
from other speakers. `style_memory` is built from self message IDs only; the
builder rejects a style window containing a non-self message. Media payloads are
not converted into textual evidence by these chat loaders.

## Privacy boundary

Raw exports, normalized conversations, embeddings, local databases, and
generated traces may contain private information. Keep them out of version
control and share only invented fixtures. The publication checker can detect
some risky paths, credentials, and symlinks, but it cannot identify every
conversation excerpt or decide whether a statement is private; review every
file selected for release.
