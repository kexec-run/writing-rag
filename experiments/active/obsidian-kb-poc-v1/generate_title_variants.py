"""Generate 4 title variants from MiniCPM5-1B-MLX over the frozen whatsapp slice."""

from __future__ import annotations

import json
import time
from pathlib import Path

from mlx_lm import generate as mlx_generate
from mlx_lm import load as mlx_load
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
CHROMA_DOCS = PROJECT_ROOT / "data/chroma_docs/chroma_documents.json"
SLICE = EXPERIMENT_DIR / "slice.json"
SNAPSHOT_DIR = (
    PROJECT_ROOT / "models/weights/huggingface/hub/models--openbmb--MiniCPM5-1B-MLX/snapshots"
)
OUTPUT_ROOT = PROJECT_ROOT / "data/generated/obsidian-kb-poc-v1/title-variants"

IMPROVED_EXTRA = (
    " Reply in English only. Do not repeat words."
    " Each title must be concise and descriptive."
)

CONFIGS = [
    ("v1-t00-orig", 0.0),
    ("v2-t07-orig", 0.7),
    ("v3-t00-improved", 0.0),
    ("v4-t07-improved", 0.7),
]


def main() -> int:
    snapshot_dir = next(SNAPSHOT_DIR.glob("*"))
    model, tokenizer = mlx_load(str(snapshot_dir))
    sampler_cache: dict[float, object] = {}

    docs_by_chunk: dict[str, dict] = {}
    for doc in json.loads(CHROMA_DOCS.read_text(encoding="utf-8")):
        if doc["metadata"]["source"] != "whatsapp":
            continue
        docs_by_chunk.setdefault(doc["metadata"]["chunk_id"], doc)

    slice_payload = json.loads(SLICE.read_text(encoding="utf-8"))
    windows = slice_payload["windows"]
    missing = [w["chunk_id"] for w in windows if w["chunk_id"] not in docs_by_chunk]
    if missing:
        raise SystemExit(f"slice windows missing from corpus: {missing}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    configs = CONFIGS

    for variant_name, temp in configs:
        is_improved = variant_name.endswith("improved")
        key = f"{temp}"
        sampler_obj = sampler_cache.get(key)
        if sampler_obj is None:
            sampler_obj = make_sampler(temp=temp, top_p=0.95)
            sampler_cache[key] = sampler_obj
        out_dir = OUTPUT_ROOT / variant_name
        out_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []
        started = time.perf_counter()

        for w in windows:
            chunk_id = w["chunk_id"]
            doc = docs_by_chunk[chunk_id]
            display_text = doc["text"].strip().replace("\n", " · ")[:800]
            instruction = (
                "Write a 3 to 6 word title describing what this conversation is about."
                " Never use the words 'whatsapp' or 'chat'."
                + (IMPROVED_EXTRA if is_improved else "")
            )
            messages = [{"role": "user", "content": instruction + "\n\nConversation:\n" + display_text}]
            prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
            t0 = time.perf_counter()
            response_text = mlx_generate(
                model, tokenizer,
                prompt=prompt_str,
                max_tokens=48,
                verbose=False,
                sampler=sampler_obj,
            )
            elapsed = round(time.perf_counter() - t0, 2)
            title = response_text.strip().split("\n")[0].strip().strip('"').rstrip(".")
            if title.lower().startswith("title:"):
                title = title[6:].strip()
            words = [
                w for w in title.split()
                if w.strip(",.;:!?").lower() not in {"whatsapp", "whats app", "chat", "chats"}
            ]
            title = " ".join(words).replace("  ", " ").strip(" ,")
            results.append({"chunk_id": chunk_id, "title": title, "seconds": elapsed})

        total_s = round(time.perf_counter() - started, 1)
        (out_dir / "titles.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results) + "\n",
            encoding="utf-8",
        )
        print(f"{variant_name}: {len(results)} titles in {total_s}s")

    print("all variants generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
