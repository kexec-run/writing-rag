# writing_rag

Importable boundary packages for the writing-memory pipeline.

## Layout

- **Doorbell:** `scripts/*.py` — thin CLIs and compatibility shims for legacy imports.
- **House:** `writing_rag/<boundary>/` — shared logic by function (retrieve, plan, generate, …).
- **Run folder:** `experiments/active|archived/<name>/` — manifest, README, fixtures.

```text
python3 scripts/<experiment>_confirmation.py --manifest experiments/active/<name>/manifest.json
```

## Boundaries

| Package | Role |
|---|---|
| `contracts/` | Intentionally empty; validators live with owning boundaries (`plan`, `verify`, `select`) |
| `ingest/` | Extractors, windows, canonical messages, document builders |
| `retrieve/` | Qwen index, retrieve, embed (no Jina) |
| `select/` | Jina text-matching / rerank over frozen Qwen pools only |
| `plan/` | Article plan, hydration, narrative, draft packets, episode ranking (implementation) |
| `generate/` | Context, outline/draft, paragraph loop, MLX runtime |
| `verify/` | Claims, transitions, allowlists |
| `runtime/` | Model paths, pipeline I/O |

Episode-slice ranking (`_episode_candidate_rank`, draft packets) lives in `plan/`, not `select/`.

Phase 2: `writing_rag/plan/*.py` holds the plan implementation. Matching
`scripts/article_plan.py`, `scripts/source_hydration.py`, and
`scripts/narrative_reconstruction.py` are doorbell shims only.

## Import

```bash
pip install -e .
# or
PYTHONPATH=$PWD python3 -c "from writing_rag.plan import build_article_plan"
```

Tests also add the repo root and `scripts/` to `sys.path` via `tests/conftest.py`.
