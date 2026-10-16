import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from pipeline_common import count_tokens as pipeline_count_tokens


MERGED_MESSAGES_FILE = Path("data/normalized/merged_messages.json")
FALLBACK_MESSAGE_FILES = [
    Path("data/extracted/instagram_messages.json"),
    Path("data/extracted/whatsapp_messages.json"),
    Path("data/extracted/reddit_messages.json"),
]
CHUNK_FILES = [
    Path("data/chunks/instagram_chunks.json"),
]



def count_tokens(text: str) -> int:
    return pipeline_count_tokens(text)


def percentile(values: list[int], percentile_value: float) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    index = round((percentile_value / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def load_json_array(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def load_messages() -> tuple[list[dict], list[Path]]:
    if MERGED_MESSAGES_FILE.exists():
        return load_json_array(MERGED_MESSAGES_FILE), [MERGED_MESSAGES_FILE]

    messages = []
    loaded_files = []
    for path in FALLBACK_MESSAGE_FILES:
        if not path.exists():
            continue

        loaded_files.append(path)
        source = path.stem.replace("_messages", "")
        for message in load_json_array(path):
            message = dict(message)
            message.setdefault("source", source)
            messages.append(message)

    return messages, loaded_files


def print_token_distribution(label: str, token_counts: list[int]) -> None:
    if not token_counts:
        print(f"{label}: no data")
        return

    total_tokens = sum(token_counts)
    print(label)
    print(f"Total items: {len(token_counts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/item: {total_tokens / len(token_counts):.2f}")
    print(f"Median tokens/item: {median(token_counts):.2f}")
    print(f"P95 tokens/item: {percentile(token_counts, 95)}")
    print(f"P99 tokens/item: {percentile(token_counts, 99)}")
    print(f"Max tokens/item: {max(token_counts)}")


def print_message_stats(messages: list[dict]) -> None:
    token_counts = []
    tokens_by_source = defaultdict(int)
    messages_by_source = defaultdict(int)

    for message in messages:
        content = str(message.get("content", ""))
        token_count = count_tokens(content)
        source = message.get("source", "unknown")

        token_counts.append(token_count)
        tokens_by_source[source] += token_count
        messages_by_source[source] += 1

    print_token_distribution("Message token distribution", token_counts)

    print("Tokens by source")
    for source in sorted(tokens_by_source):
        print(
            f"- {source}: {tokens_by_source[source]} tokens "
            f"across {messages_by_source[source]} messages"
        )


def print_chunk_stats() -> None:
    chunk_token_counts = []
    overlap_tokens = 0
    overlap_messages = 0

    for path in CHUNK_FILES:
        if not path.exists():
            continue

        for chunk in load_json_array(path):
            chunk_token_counts.append(chunk["token_count"])

            overlap_count = chunk.get("overlap_message_count", 0)
            if overlap_count:
                overlap_text = "\n".join(
                    message["content"] for message in chunk["messages"][:overlap_count]
                )
                overlap_tokens += count_tokens(overlap_text)
                overlap_messages += overlap_count

    print_token_distribution("Chunk token distribution", chunk_token_counts)
    if chunk_token_counts:
        print(f"Overlap messages copied into chunks: {overlap_messages}")
        print(f"Approx overlap tokens copied into chunks: {overlap_tokens}")


def main() -> None:
    messages, loaded_files = load_messages()
    messages.sort(key=lambda message: message.get("timestamp_ms", 0))

    print("Loaded message files:")
    for path in loaded_files:
        print(f"- {path}")

    # Real corpus measurement matters because chunking and retrieval quality
    # depend on actual token distribution, not estimated character lengths.
    print_message_stats(messages)

    # Retrieval operates over chunks, so message-level stats are only half the
    # story. Chunk distributions reveal density, fragmentation, and overlap cost.
    print_chunk_stats()


if __name__ == "__main__":
    main()
