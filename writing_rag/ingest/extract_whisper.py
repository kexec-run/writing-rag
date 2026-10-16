"""Extract timestamped self-authored messages from a Whisper JSON transcript."""

from __future__ import annotations

import argparse

from writing_rag.runtime.pipeline_common import clean_text, read_json, write_json


def extract_segments(data: dict) -> list[dict]:
    messages = []
    for index, segment in enumerate(data.get("transcription", [])):
        text = clean_text(segment.get("text"))
        if not text:
            continue
        offsets = segment.get("offsets") or {}
        messages.append(
            {
                "timestamp_ms": int(offsets.get("from", 0)),
                "end_timestamp_ms": int(offsets.get("to", offsets.get("from", 0))),
                "content": text,
                "source": "whisper",
                "mode": "long_form_reasoning",
                "sender": "self",
                "segment_index": index,
            }
        )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/whisper/transcript.json")
    parser.add_argument("--output", default="data/extracted/whisper_messages.json")
    args = parser.parse_args()
    messages = extract_segments(read_json(args.input))
    write_json(args.output, messages)
    print(f"Whisper segments: {len(messages)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
