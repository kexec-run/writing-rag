import json
from collections import Counter
from pathlib import Path


INPUT_FILES = [
    Path("data/extracted/instagram_messages.json"),
    Path("data/extracted/whatsapp_messages.json"),
    Path("data/extracted/reddit_messages.json"),
]
OUTPUT_FILE = Path("data/normalized/merged_messages.json")

DEFAULT_MODES = {
    "instagram": "conversational",
    "whatsapp": "long_form_reasoning",
    "reddit": "conversational",
}


def load_json_array(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def source_from_path(path: Path) -> str:
    return path.stem.replace("_messages", "")


def normalize_message(message: dict, source: str) -> dict | None:
    content = str(message.get("content", "")).strip()
    if not content:
        return None

    # This is the unified internal memory format. It keeps the shared fields
    # stable for chunking/RAG while preserving optional platform metadata.
    normalized = {
        "timestamp_ms": message["timestamp_ms"],
        "content": content,
        "source": message.get("source", source),
        "mode": message.get("mode", DEFAULT_MODES.get(source, "conversational")),
    }

    for optional_key in ("sender", "age"):
        if optional_key in message:
            normalized[optional_key] = message[optional_key]

    return normalized


def main() -> None:
    merged = []
    skipped = 0

    for path in INPUT_FILES:
        if not path.exists():
            continue

        source = source_from_path(path)
        for message in load_json_array(path):
            normalized = normalize_message(message, source)
            if normalized is None:
                skipped += 1
                continue
            merged.append(normalized)

    merged.sort(key=lambda message: message["timestamp_ms"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    source_counts = Counter(message["source"] for message in merged)
    print(f"Merged messages: {len(merged)}")
    print(f"Skipped messages: {skipped}")
    print("Messages by source:")
    for source in sorted(source_counts):
        print(f"- {source}: {source_counts[source]}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
