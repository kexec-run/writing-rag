"""Freeze the obsidian-kb-poc-v1 whatsapp slice before any arm build."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
CHROMA_DOCS = PROJECT_ROOT / "data/chroma_docs/chroma_documents.json"

QUOTA_PER_THREAD = 60


def load_whatsapp() -> list[dict]:
    docs = json.loads(CHROMA_DOCS.read_text(encoding="utf-8"))
    return [d for d in docs if d["metadata"]["source"] == "whatsapp"]


def month_key(ts_ms: int) -> str:
    if not ts_ms:
        return "unknown"
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m")


def largest_remainder(counts: dict[str, int], quota: int) -> dict[str, int]:
    total = sum(counts.values())
    alloc = {month: quota * n // total for month, n in counts.items()}
    remainder = quota - sum(alloc.values())
    order = sorted(
        counts,
        key=lambda m: (-(counts[m] - quota * counts[m] / total), m),
    )
    for month in order[:remainder]:
        alloc[month] += 1
    return alloc


def select_slice(docs: list[dict]) -> list[dict]:
    by_thread: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for doc in docs:
        chunk_id = doc["metadata"]["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        by_thread[doc["metadata"]["source_thread_id"]].append(doc)
    frozen: list[dict] = []
    for thread in sorted(by_thread):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for doc in by_thread[thread]:
            buckets[month_key(int(doc["metadata"]["start_timestamp_ms"]))].append(doc)
        counts = {m: len(b) for m, b in buckets.items()}
        alloc = largest_remainder(counts, QUOTA_PER_THREAD)
        picked: list[dict] = []
        for month in sorted(alloc):
            ranked = sorted(
                buckets[month],
                key=lambda d: (-len(d["text"]), d["metadata"]["chunk_id"]),
            )
            picked.extend(ranked[: alloc[month]])
        picked.sort(key=lambda d: (int(d["metadata"]["start_timestamp_ms"]), d["metadata"]["chunk_id"]))
        frozen.extend(picked)
    return frozen


def distinctiveness(docs: list[dict]) -> list[tuple[str, float]]:
    stop = {
        "the", "and", "for", "that", "this", "with", "was", "were", "are",
        "you", "your", "have", "has", "had", "not", "but", "from", "they",
        "them", "then", "than", "what", "when", "will", "would", "can",
        "could", "about", "into", "just", "like", "some", "more", "only",
    }
    counts: dict[str, int] = defaultdict(int)
    for doc in docs:
        for term in str(doc["metadata"].get("topic_terms", "")).split(","):
            term = term.strip().casefold()
            if len(term) >= 4 and term not in stop and not term.isdigit():
                counts[term] += 1
    total_docs = max(len(docs), 1)
    scored = [(t, c / total_docs) for t, c in counts.items()]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(EXPERIMENT_DIR / "slice.json"))
    args = parser.parse_args()

    docs = load_whatsapp()
    frozen = select_slice(docs)
    per_thread: dict[str, int] = defaultdict(int)
    for doc in frozen:
        per_thread[doc["metadata"]["source_thread_id"]] += 1

    payload = {
        "schema_version": "obsidian-kb-poc-slice-v1",
        "frozen_at": datetime.now(tz=UTC).isoformat(),
        "quota_per_thread": QUOTA_PER_THREAD,
        "window_count": len(frozen),
        "per_thread": dict(per_thread),
        "windows": [
            {
                "chunk_id": doc["metadata"]["chunk_id"],
                "thread": doc["metadata"]["source_thread_id"],
                "date_start_ms": int(doc["metadata"]["start_timestamp_ms"]),
                "document_ids": [
                    other["document_id"]
                    for other in docs
                    if other["metadata"]["chunk_id"] == doc["metadata"]["chunk_id"]
                ],
                "text_chars": len(doc["text"]),
                "topic_terms": str(doc["metadata"].get("topic_terms", "")),
            }
            for doc in frozen
        ],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"frozen {len(frozen)} windows -> {args.output}")
    print("per-thread:", dict(per_thread))
    for thread in sorted(per_thread):
        subset = [d for d in frozen if d["metadata"]["source_thread_id"] == thread]
        top = distinctiveness(subset)[:12]
        print(f"top terms {thread}: {[t for t, _ in top]}")
        heavy = sorted(subset, key=lambda d: -len(d["text"]))[:5]
        print(f"  heavy windows: {[(d['metadata']['chunk_id'], len(d['text'])) for d in heavy]}")


if __name__ == "__main__":
    main()
