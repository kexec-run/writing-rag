"""Generate note titles for the obsidian-kb-poc-v1 slice via MiniCPM5-1B-MLX."""

from __future__ import annotations

import json
import time
from pathlib import Path

from mlx_lm import generate as mlx_generate
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
CHROMA_DOCS = PROJECT_ROOT / "data/chroma_docs/chroma_documents.json"
SLICE = EXPERIMENT_DIR / "slice.json"
SNAPSHOT_DIR = (
    PROJECT_ROOT / "models/weights/huggingface/hub/models--openbmb--MiniCPM5-1B-MLX/snapshots"
)
OUTPUT = PROJECT_ROOT / "data/generated/obsidian-kb-poc-v1/title-bakeoff/titles-minicpm5.jsonl"

PROMPT_TEMPLATE = (
    "Write a 3 to 6 word title describing what this WhatsApp conversation "
    "is about. Reply with the title only.\n\nConversation:\n{text}"
)


def main() -> int:
    docs_by_chunk: dict[str, dict] = {}
    for doc in json.loads(CHROMA_DOCS.read_text(encoding="utf-8")):
        if doc["metadata"]["source"] != "whatsapp":
            continue
        chunk_id = doc["metadata"]["chunk_id"]
        if chunk_id not in docs_by_chunk:
            docs_by_chunk[chunk_id] = doc

    slice_payload = json.loads(SLICE.read_text(encoding="utf-8"))
    windows = slice_payload["windows"]

    snapshots = list(SNAPSHOT_DIR.iterdir())
    if len(snapshots) != 1:
        raise SystemExit(f"expected one snapshot dir, got {snapshots}")
    snapshot_dir = snapshots[0]
    model, tokenizer = load(str(snapshot_dir))
    sampler = make_sampler(temp=0.7, top_p=0.95)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    started = time.perf_counter()
    for index, window in enumerate(windows):
        chunk_id = window["chunk_id"]
        text = docs_by_chunk[chunk_id]["text"].strip().replace("\n", " · ")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text[:800])}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        t0 = time.perf_counter()
        raw = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=48, verbose=False, sampler=sampler)
        elapsed = round(time.perf_counter() - t0, 2)
        title = raw.strip().splitlines()[0].strip().strip('"').rstrip(".")
        # collapse consecutive duplicates and trailing conjunctions
        words = title.split()
        cleaned: list[str] = []
        for wd in words:
            key = wd.strip(",.;:!\"'").casefold()
            if cleaned and cleaned[-1].strip(",.;:!\"'").casefold() == key:
                continue
            cleaned.append(wd)
        while cleaned and cleaned[-1].strip(",.;:!?\"'").casefold() in {
            "vs", "and", "or", "the", "a", "an", "of", "in", "on", "at",
        }:
            cleaned.pop()
        title = " ".join(cleaned[:8]).strip(",;: ")
        lines.append(
            json.dumps({"chunk_id": chunk_id, "title": title, "seconds": elapsed})
        )
        print(f"{index+1}/{len(windows)} {chunk_id}: {title}", flush=True)

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = round(time.perf_counter() - started, 1)
    print(f"wrote {len(lines)} titles in {total}s -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
