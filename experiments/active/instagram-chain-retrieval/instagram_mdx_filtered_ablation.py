"""Evaluate MDX retrieval after removing Instagram link and image noise.

This is an isolated, retrieval-only experiment. Instagram messages are filtered
in memory, the Qwen-tokenized windows and anchor chains are rebuilt, and the
three held-out MDX references are used only after retrieval as semantic
relevance proxies. Canonical messages and production retrieval artifacts are
not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from embed_chunks import build_vectors, create_backend  # noqa: E402
from instagram_chain_relevance_eval import (  # noqa: E402
    DEFAULT_MESSAGES,
    DEFAULT_MODEL_CACHE,
    DEFAULT_EMBEDDING_MODEL,
    TARGETS,
    mean,
    reference_units,
)
from instagram_post_chain_ablation import evaluate_results  # noqa: E402
from instagram_post_filtered_ablation import (  # noqa: E402
    build_filtered_conditions,
    filter_instagram_messages,
)
from model_paths import configure_model_cache  # noqa: E402
from pipeline_common import read_json  # noqa: E402
from retrieve import EmbeddingSession, retrieve_many  # noqa: E402
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/generated/instagram-chain-relevance-filtered-v1.json"
)
TARGET_NAMES = ("duality", "reality", "animal")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--model-cache", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="mps")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--review-top-k", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    model_cache = configure_model_cache(resolve_path(args.model_cache))
    original_messages = read_json(resolve_path(args.messages))
    filtered_messages, filter_stats = filter_instagram_messages(original_messages)
    conditions, rebuild_stats, chain_report = build_filtered_conditions(
        filtered_messages,
        model_cache,
    )
    documents = {
        condition: {
            document["document_id"]: document for document in condition_documents
        }
        for condition, condition_documents in conditions.items()
    }
    all_documents = [
        document
        for condition_documents in conditions.values()
        for document in condition_documents
    ]

    backend, embedding_metadata = create_backend(
        "qwen3",
        model_name=args.embedding_model,
        device=args.embedding_device,
        model_cache=model_cache,
    )
    vectors = build_vectors(
        all_documents,
        backend=backend,
        batch_size=args.embedding_batch_size,
    )
    embedding_metadata.update(
        {
            "document_count": len(vectors),
            "model_cache": str(model_cache),
            "source_documents": {
                condition: len(condition_documents)
                for condition, condition_documents in conditions.items()
            },
        }
    )
    session = EmbeddingSession(vectors, embedding_metadata)
    session._backend = backend
    vector_by_id = {item["document_id"]: item["embedding"] for item in vectors}

    document_vectors_by_text: dict[str, list[float]] = {}
    for document in all_documents:
        document_vectors_by_text.setdefault(
            document["text"],
            vector_by_id[document["document_id"]],
        )

    reference_texts = {
        name: reference_units(
            resolve_path(TARGETS[name]["reference"]).read_text(encoding="utf-8")
        )
        for name in TARGET_NAMES
    }
    unique_reference_texts = list(
        dict.fromkeys(unit for units in reference_texts.values() for unit in units)
    )
    missing_reference_texts = [
        unit for unit in unique_reference_texts if unit not in document_vectors_by_text
    ]
    reference_embeddings = backend.encode(
        missing_reference_texts,
        batch_size=args.embedding_batch_size,
    )
    encoded_by_text = {
        **document_vectors_by_text,
        **dict(zip(missing_reference_texts, reference_embeddings)),
    }
    reference_vectors = {
        name: {
            str(index): encoded_by_text[unit]
            for index, unit in enumerate(units)
        }
        for name, units in reference_texts.items()
    }

    target_details: dict[str, Any] = {}
    for target_name in TARGET_NAMES:
        target = TARGETS[target_name]
        condition_results: dict[str, Any] = {}
        for condition, condition_documents in conditions.items():
            view_results: dict[str, Any] = {}
            for view, top_k in (("content", 10), ("style", 5)):
                results = retrieve_many(
                    [target["query"]],
                    condition_documents,
                    vectors,
                    top_k=top_k,
                    collection=f"{view}_memory",
                    embedding_metadata=embedding_metadata,
                    session=session,
                )[0]
                candidate_vectors = {
                    item["document"]["document_id"]: vector_by_id[
                        item["document"]["document_id"]
                    ]
                    for item in results
                }
                view_results[view] = evaluate_results(
                    results,
                    documents[condition],
                    candidate_vectors,
                    reference_vectors[target_name],
                    review_top_k=args.review_top_k,
                )
            condition_results[condition] = view_results
        target_details[target_name] = {
            "query": target["query"],
            "reference_path": str(resolve_path(target["reference"])),
            "reference_unit_count": len(reference_texts[target_name]),
            "conditions": condition_results,
        }

    macro: dict[str, Any] = {}
    for condition in conditions:
        for view in ("content", "style"):
            rows = [
                target_details[target_name]["conditions"][condition][view]
                for target_name in TARGET_NAMES
            ]
            best_values = [
                row["reference_max_cosine"]
                for row in rows
                if row["reference_max_cosine"] is not None
            ]
            top3_values = [
                row["reference_top3_mean_cosine"]
                for row in rows
                if row["reference_top3_mean_cosine"] is not None
            ]
            macro.setdefault(condition, {})[view] = {
                "targets_with_results": len(best_values),
                "mean_best_candidate_reference_cosine": mean(best_values),
                "mean_top3_reference_cosine": mean(top3_values),
            }

    report = {
        "experiment": "instagram-chain-relevance-filtered-v1",
        "scope": {
            "source": "instagram",
            "target_count": len(TARGET_NAMES),
            "canonical_messages_modified": False,
            "retrieval_corpus_modified": False,
            "generation_run": False,
            "filter_applied_before_rechunking": True,
        },
        "filter": {
            "url_only_messages_removed": True,
            "embedded_urls_stripped": True,
            "image_placeholders_removed": True,
            "image_placeholder_definition": (
                "sent an attachment or took a screenshot"
            ),
            "stats": filter_stats,
        },
        "rebuild": rebuild_stats,
        "tokenizer": chain_report["tokenizer"],
        "embedding": {
            **embedding_metadata,
            "reference_unit_counts": {
                name: len(units) for name, units in reference_texts.items()
            },
            "encoded_text_count": len(all_documents) + len(missing_reference_texts),
            "reference_texts_reused_from_documents": (
                len(unique_reference_texts) - len(missing_reference_texts)
            ),
            "reference_texts_encoded_separately": len(missing_reference_texts),
        },
        "targets": {
            name: {
                "query": TARGETS[name]["query"],
                "reference_path": str(resolve_path(TARGETS[name]["reference"])),
                "reference_unit_count": len(reference_texts[name]),
            }
            for name in TARGET_NAMES
        },
        "macro": macro,
        "targets_detail": target_details,
        "limitations": [
            "Reference similarity is a semantic proxy, not a human relevance label or factual-grounding score.",
            "The MDX references were used after retrieval and were not added to the retrieval corpus.",
            "This evaluates only three held-out MDX queries and does not run generation.",
        ],
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "filter": filter_stats,
                "rebuild": rebuild_stats,
                "macro": macro,
                "encoded_text_count": len(all_documents) + len(missing_reference_texts),
            },
            indent=2,
        )
    )
    print(f"Embedded documents: {len(vectors)}")
    print(f"Saved filtered MDX relevance evaluation to {output}")


if __name__ == "__main__":
    main()
