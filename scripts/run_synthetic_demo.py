"""Run a hashing-only import, retrieval, and context-assembly demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_synthetic_demo import INPUTS, prepare_inputs

from writing_rag.generate.generate import (
    build_generation_context,
    retrieve_style_examples,
)
from writing_rag.retrieve.retrieve import EmbeddingSession, retrieve
from writing_rag.runtime.pipeline_common import read_json

SELF_SENDERS = {
    "instagram": ["Demo Author"],
    "whatsapp": ["Demo Author"],
    "reddit": ["@demo-author:example.invalid"],
}


def _self_message_texts(messages: list[dict[str, Any]]) -> set[str]:
    return {str(message["content"]) for message in messages if message.get("speaker") == "self"}


def verify_outputs(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    messages = read_json(project_root / "data/normalized/messages.json")
    documents = read_json(project_root / "data/chroma_docs/chroma_documents.json")
    vectors = read_json(project_root / "data/embeddings/vectors.json")
    metadata = read_json(project_root / "data/embeddings/metadata.json")
    if metadata.get("embedding_backend") != "hashing":
        raise ValueError("synthetic demo must use the explicit hashing diagnostic backend")
    if not messages or not documents or not vectors:
        raise ValueError("pipeline produced an empty messages, documents, or vectors artifact")

    by_id = {document["document_id"]: document for document in documents}
    session = EmbeddingSession(vectors, metadata)
    content_documents = [item for item in documents if item.get("collection") == "content_memory"]
    style_documents = [item for item in documents if item.get("collection") == "style_memory"]
    content_results = retrieve(
        "meeting invitation start time email timezone calendar",
        content_documents,
        vectors,
        top_k=5,
        collection="content_memory",
        embedding_metadata=metadata,
        session=session,
    )
    if not content_results or not any("meeting" in item["document"]["text"].casefold() for item in content_results):
        raise AssertionError("synthetic content query did not retrieve the expected meeting-time evidence")

    style_results = retrieve_style_examples(
        "direct first-person explanation of checking a meeting time zone",
        style_documents,
        vectors,
        metadata,
        top_k=2,
        session=session,
    )
    self_texts = _self_message_texts(messages)
    if not style_results:
        raise AssertionError("synthetic style query returned no self-authored examples")
    for result in style_results:
        document = result["document"]
        item = by_id[document["document_id"]]
        ids = [value for value in item.get("metadata", {}).get("self_message_ids", "").split(",") if value]
        message_by_id = {message["message_id"]: message for message in messages}
        if not ids or any(message_by_id[message_id].get("speaker") != "self" for message_id in ids):
            raise AssertionError("non-self message entered a style example")
        expected = "\n".join(message_by_id[message_id]["content"] for message_id in ids)
        if document["text"] != expected or any(text and text in document["text"] for text in self_texts if text not in expected):
            raise AssertionError("style text differs from its exact self-message source")

    context = build_generation_context(
        "Explain how checking time zones clarified the different meeting times.",
        content_results,
        "Write a concise, evidence-grounded explanation.",
        style_results=style_results,
        style_template="Use a direct, qualified voice.",
    )
    if "Retrieved content evidence" not in context or "Derived style profile" not in context:
        raise AssertionError("generation context did not keep content and style sections distinct")
    if any(message["speaker"] == "other" and message["content"] in context for message in messages):
        # Other-speaker content is allowed in the content section, but not in style evidence.
        style_section = context.split("## Derived style profile", 1)[-1]
        if any(message["speaker"] == "other" and message["content"] in style_section for message in messages):
            raise AssertionError("other-speaker text leaked into the style profile")

    return {
        "message_count": len(messages),
        "content_document_count": len(content_documents),
        "style_document_count": len(style_documents),
        "content_hits": [item["document"]["document_id"] for item in content_results],
        "style_hits": [item["document"]["document_id"] for item in style_results],
        "embedding_backend": "hashing",
        "semantic_model_validation": False,
        "generation_model_run": False,
        "context_assembled": True,
    }


def _existing_demo_inputs(project_root: Path = PROJECT_ROOT) -> list[Path] | None:
    """Accept existing raw inputs only when they exactly match the fixtures."""
    data_root = project_root / "data"
    if data_root.is_symlink():
        raise RuntimeError("refusing to run synthetic demo through a data symlink")
    raw_root = data_root / "raw"
    if raw_root.is_symlink():
        raise RuntimeError("refusing to run synthetic demo through a data/raw symlink")
    if not raw_root.exists():
        return None

    found: list[Path] = []
    for directory, dirnames, filenames in os.walk(raw_root, followlinks=False):
        current = Path(directory)
        for name in list(dirnames):
            path = current / name
            if path.is_symlink():
                raise RuntimeError("refusing to run synthetic demo with symlinked raw input")
        for name in filenames:
            path = current / name
            if path.is_symlink():
                raise RuntimeError("refusing to run synthetic demo with symlinked raw input")
            found.append(path)
    if not found:
        return None

    expected = {
        (Path("data/raw") / relative).as_posix() for relative in INPUTS.values()
    }
    observed = {path.relative_to(project_root).as_posix() for path in found}
    if observed != expected:
        raise RuntimeError(
            "refusing to run synthetic demo unless data/raw contains exactly "
            "the four synthetic fixtures"
        )

    fixture_root = project_root / "tests/fixtures/synthetic_import"
    for source, relative in INPUTS.items():
        destination = project_root / "data/raw" / relative
        if destination.read_bytes() != (fixture_root / source).read_bytes():
            raise RuntimeError(
                "refusing to run synthetic demo because a raw input differs from its fixture"
            )
    return found


def main() -> int:
    copied = _existing_demo_inputs() or prepare_inputs()
    environment = os.environ.copy()
    environment["WRITING_RAG_SELF_SENDERS"] = json.dumps(SELF_SENDERS)
    environment["WRITING_RAG_WHATSAPP_TIMEZONE"] = "UTC"
    subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--embedding-backend", "hashing"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    result = verify_outputs()
    print(json.dumps({"prepared_file_count": len(copied), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
