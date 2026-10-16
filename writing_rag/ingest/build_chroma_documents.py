"""Project accepted reasoning chains into optional enriched retrieval views."""

from __future__ import annotations

import argparse
import re
from typing import Any

from writing_rag.runtime.pipeline_common import read_json, write_json


CONTENT_VIEWS = {"summary", "reasoning", "knowledge"}
VIEW_CONFIG = {
    "summary": "content_memory",
    "reasoning": "content_memory",
    "knowledge": "content_memory",
    "style": "style_memory",
}


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def reasoning_text(chain: dict[str, Any], messages_by_id: dict[str, dict[str, Any]]) -> str:
    context = [messages_by_id[message_id]["content"] for message_id in chain["context_message_ids"]]
    self_text = [messages_by_id[message_id]["content"] for message_id in chain["self_message_ids"]]
    sections: list[str] = []
    if context:
        sections.append("Context:\n" + "\n".join(context))
    if self_text:
        sections.append("Self:\n" + "\n".join(self_text))
    return "\n\n".join(sections)


def base_metadata(chain: dict[str, Any]) -> dict[str, Any]:
    labels = chain["labels"]
    scores = chain["scores"]
    return {
        "chain_id": chain["chain_id"],
        "source": chain["source"],
        "source_thread_id": chain["source_thread_id"],
        "sequence_type": chain["sequence_type"],
        "quality": "high_signal" if scores["retrieval_value"] >= 0.65 else "standard_signal",
        "retrieval_value": scores["retrieval_value"],
        "topics": ",".join(labels["topics"]),
        "reasoning_type": labels["reasoning_type"],
        "style_mode": labels["style_mode"],
        "message_ids": ",".join(chain["message_ids"]),
        "self_message_ids": ",".join(chain["self_message_ids"]),
        "other_message_ids": ",".join(chain["context_message_ids"]),
        "score_version": chain["score_version"],
        "interpreter": chain["interpreter"],
        "reasoning_status": "accepted",
    }


def build_documents(chains: list[dict[str, Any]], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages_by_id = {message["message_id"]: message for message in messages}
    documents: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    for chain in chains:
        if chain.get("status") != "accepted":
            continue
        view_texts = {
            "summary": chain["chain_summary"],
            "reasoning": reasoning_text(chain, messages_by_id),
            "style": "\n".join(
                messages_by_id[message_id]["content"] for message_id in chain["self_message_ids"]
            ),
            "knowledge": "\n".join(chain["knowledge_claims"]),
        }
        for view, text in view_texts.items():
            if not text.strip():
                continue
            if view == "style" and any(
                messages_by_id[message_id]["speaker"] != "self"
                for message_id in chain["self_message_ids"]
            ):
                raise ValueError(f"Style chain {chain['chain_id']} contains non-self messages")
            dedup_key = (view, normalize_for_dedup(text))
            if dedup_key in seen:
                continue
            document_id = f"{chain['chain_id']}_{view}"
            seen[dedup_key] = document_id
            metadata = base_metadata(chain)
            metadata.update(
                {
                    "view": view,
                    "collection": VIEW_CONFIG[view],
                    "source_scope": "style" if view == "style" else "content",
                    "speaker_filter": "self_only" if view == "style" else "self_and_context",
                    "style_owner": "self" if view == "style" else "none",
                    "style_signal": chain["scores"]["article_usefulness"] if view == "style" else None,
                }
            )
            documents.append(
                {
                    "document_id": document_id,
                    "chain_id": chain["chain_id"],
                    "view": view,
                    "collection": VIEW_CONFIG[view],
                    "text": text,
                    "metadata": metadata,
                }
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default="data/normalized/messages.json")
    parser.add_argument("--chains", default="data/chains/reasoning_chains.json")
    parser.add_argument("--output", default="data/chroma_docs/chroma_documents.json")
    args = parser.parse_args()
    documents = build_documents(read_json(args.chains), read_json(args.messages))
    write_json(args.output, documents)
    print(f"Optional enriched retrieval documents: {len(documents)}")
    print(f"Content documents: {sum(document['collection'] == 'content_memory' for document in documents)}")
    print(f"Style documents: {sum(document['collection'] == 'style_memory' for document in documents)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
