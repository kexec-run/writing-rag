"""Ingest retrieval documents into separate Chroma collections."""

from __future__ import annotations

import argparse
from typing import Any

from pipeline_common import read_json, write_json


def scalar_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def ingest(
    documents: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    persist_directory: str,
) -> dict[str, int]:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Chroma ingestion requires chromadb. Install requirements-qwen.txt."
        ) from exc

    vector_by_id = {item["document_id"]: item["embedding"] for item in vectors}
    client = chromadb.PersistentClient(path=persist_directory)
    counts: dict[str, int] = {}
    for collection_name in ("content_memory", "style_memory"):
        collection_documents = [
            document
            for document in documents
            if document.get("collection") == collection_name
        ]
        collection = client.get_or_create_collection(collection_name)
        usable = [
            document
            for document in collection_documents
            if document["document_id"] in vector_by_id
        ]
        if usable:
            collection.upsert(
                ids=[document["document_id"] for document in usable],
                documents=[document["text"] for document in usable],
                embeddings=[vector_by_id[document["document_id"]] for document in usable],
                metadatas=[scalar_metadata(document["metadata"]) for document in usable],
            )
        counts[collection_name] = len(usable)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--vectors", default="data/embeddings/vectors.json")
    parser.add_argument("--persist-directory", default="data/chroma")
    parser.add_argument("--manifest", default="data/chroma/manifest.json")
    args = parser.parse_args()

    counts = ingest(read_json(args.documents), read_json(args.vectors), args.persist_directory)
    write_json(
        args.manifest,
        {
            "collections": counts,
            "documents": args.documents,
            "vectors": args.vectors,
            "persist_directory": args.persist_directory,
        },
    )
    print(f"Ingested content_memory: {counts['content_memory']}")
    print(f"Ingested style_memory: {counts['style_memory']}")


if __name__ == "__main__":
    main()
