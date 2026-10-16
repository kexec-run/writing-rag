import json
import argparse
from pathlib import Path

from writing_rag.runtime.pipeline_common import count_tokens as pipeline_count_tokens


INPUT_FILE = Path("data/extracted/instagram_messages.json")
OUTPUT_FILE = Path("data/chunks/instagram_chunks.json")

MAX_CHUNK_TOKENS = 800
MAX_TIME_GAP_MS = 30 * 60 * 1000
MIN_CHUNK_TOKENS = 300
MIN_CHUNK_MESSAGES = 8
OVERLAP_MESSAGES = 2



def count_tokens(text: str) -> int:
    return pipeline_count_tokens(text)


def combined_text(messages: list[dict]) -> str:
    return "\n".join(message["content"] for message in messages)


def message_token_count(message: dict, is_first_message: bool) -> int:
    if is_first_message:
        return message["token_count"]
    return message["token_count_with_separator"]


def chunk_token_count(messages: list[dict]) -> int:
    total = 0
    for index, message in enumerate(messages):
        total += message_token_count(message, is_first_message=index == 0)
    return total


def build_chunk(chunk_number: int, messages: list[dict], overlap_message_count: int) -> dict:
    text = combined_text(messages)
    cleaned_messages = [
        {
            "timestamp_ms": message["timestamp_ms"],
            "content": message["content"],
        }
        for message in messages
    ]

    return {
        "chunk_id": f"chunk_{chunk_number:03d}",
        "start_timestamp_ms": messages[0]["timestamp_ms"],
        "end_timestamp_ms": messages[-1]["timestamp_ms"],
        "token_count": count_tokens(text),
        "message_count": len(messages),
        "source": messages[0].get("source", "unknown"),
        "contains_overlap": overlap_message_count > 0,
        "overlap_message_count": overlap_message_count,
        "messages": cleaned_messages,
        "combined_text": text,
    }


def load_messages() -> list[dict]:
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        messages = json.load(f)

    sorted_messages = sorted(messages, key=lambda message: message["timestamp_ms"])
    for message in sorted_messages:
        content = message["content"]
        message["token_count"] = count_tokens(content)
        message["token_count_with_separator"] = count_tokens(f"\n{content}")

    return sorted_messages


def chunk_messages(messages: list[dict]) -> list[dict]:
    chunks = []
    current_messages = []
    current_token_count = 0
    current_overlap_count = 0

    for message in messages:
        if not current_messages:
            current_messages.append(message)
            current_token_count = message_token_count(message, is_first_message=True)
            continue

        previous_timestamp_ms = current_messages[-1]["timestamp_ms"]
        time_gap_ms = message["timestamp_ms"] - previous_timestamp_ms
        candidate_message_token_count = message_token_count(message, is_first_message=False)
        candidate_token_count = current_token_count + candidate_message_token_count

        # Token counting matters because embedding and generation models have
        # finite context windows. Keeping chunks bounded makes retrieval more
        # predictable and avoids oversized inputs later in the RAG pipeline.
        exceeds_token_limit = candidate_token_count > MAX_CHUNK_TOKENS

        # Time-window chunking keeps messages from separate conversations or
        # sessions from being blended into one chunk just because they are short.
        # Tiny chunks are allowed to cross time gaps until they have enough
        # density to be useful for retrieval.
        has_minimum_density = (
            current_token_count >= MIN_CHUNK_TOKENS
            or len(current_messages) >= MIN_CHUNK_MESSAGES
        )
        exceeds_time_window = time_gap_ms > MAX_TIME_GAP_MS and has_minimum_density

        if exceeds_time_window or exceeds_token_limit:
            chunks.append(build_chunk(len(chunks) + 1, current_messages, current_overlap_count))

            # Overlap preserves semantic continuity across chunk boundaries, so
            # a retrieved chunk still has a little context from the prior chunk.
            overlap_messages = current_messages[-OVERLAP_MESSAGES:]
            current_messages = overlap_messages + [message]
            current_overlap_count = len(overlap_messages)
            current_token_count = chunk_token_count(current_messages)

            while current_token_count > MAX_CHUNK_TOKENS and len(current_messages) > 1:
                current_messages.pop(0)
                current_overlap_count = max(0, current_overlap_count - 1)
                current_token_count = chunk_token_count(current_messages)
        else:
            current_messages.append(message)
            current_token_count = candidate_token_count

    if current_messages:
        chunks.append(build_chunk(len(chunks) + 1, current_messages, current_overlap_count))

    return chunks


def main() -> None:
    global INPUT_FILE, OUTPUT_FILE, MAX_CHUNK_TOKENS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--max-tokens", type=int, default=MAX_CHUNK_TOKENS)
    args = parser.parse_args()
    INPUT_FILE = Path(args.input)
    OUTPUT_FILE = Path(args.output)
    MAX_CHUNK_TOKENS = args.max_tokens
    messages = load_messages()
    chunks = chunk_messages(messages)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    token_counts = [chunk["token_count"] for chunk in chunks]
    average_token_count = sum(token_counts) / len(token_counts) if token_counts else 0
    largest_token_count = max(token_counts) if token_counts else 0

    print(f"Total messages: {len(messages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk token count: {average_token_count:.2f}")
    print(f"Largest chunk token count: {largest_token_count}")


if __name__ == "__main__":
    main()
