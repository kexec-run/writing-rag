"""Short real-data parity smoke test for the zembed MLX adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BACKEND_PATH = Path(__file__).with_name("zembed_mlx_backend.py")
spec = importlib.util.spec_from_file_location("zembed_mlx_backend", BACKEND_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load zembed adapter")
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)
ZembedMLXBackend = backend_module.ZembedMLXBackend
MODEL_NAME = backend_module.MODEL_NAME

CARD_EXAMPLE = {
    "query": "What is backpropagation?",
    "documents": [
        "Backpropagation is a fundamental algorithm for training neural networks by computing gradients.",
        "Gradient descent is used to optimize model parameters during the training process.",
        "Neural network training relies on efficient computation of derivatives through backpropagation.",
    ],
}
THRESHOLDS = {
    "min_batch_invariance_cosine": 0.999,
    "min_permutation_invariance_cosine": 0.999,
    "max_norm_error": 0.001,
    "max_card_batch_score_delta": 0.01,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_cosine(left: list[float], right: list[float]) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_norm = sum(value * value for value in left_values) ** 0.5
    right_norm = sum(value * value for value in right_values) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("cannot compare a zero vector")
    value = sum(a * b for a, b in zip(left_values, right_values)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def vector_stats(vectors: list[list[float]]) -> dict[str, float]:
    norms = [sum(float(value) * float(value) for value in vector) ** 0.5 for vector in vectors]
    return {"min_norm": min(norms), "max_norm": max(norms), "max_norm_error": max(abs(norm - 1.0) for norm in norms)}


def compare_vectors(first: list[list[float]], second: list[list[float]]) -> dict[str, float]:
    cosines = [normalized_cosine(left, right) for left, right in zip(first, second)]
    return {"count": len(cosines), "min_cosine": min(cosines), "mean_cosine": sum(cosines) / len(cosines)}


def select_documents(documents: list[dict[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    eligible = [
        document
        for document in documents
        if document.get("collection") == "content_memory"
        and document.get("view") == "content"
        and len(str(document.get("text", ""))) <= 8000
        and not (
            document.get("metadata", {}).get("source", "").casefold() == "chatgpt"
            and not str(document.get("metadata", {}).get("self_message_ids", "")).strip()
        )
    ]
    if len(eligible) < count:
        raise ValueError(f"expected at least {count} eligible documents, found {len(eligible)}")
    ordered = sorted(eligible, key=lambda document: len(str(document["text"])))
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    selected = []
    seen: set[str] = set()
    seen_texts: set[str] = set()
    for position in positions:
        document = ordered[position]
        document_id = str(document["document_id"])
        document_text = str(document["text"])
        if document_id not in seen and document_text not in seen_texts:
            selected.append(document)
            seen.add(document_id)
            seen_texts.add(document_text)
    if len(selected) != count:
        raise ValueError("length-stratified sample did not produce unique documents")
    return selected


def card_batch_check(backend: Any) -> dict[str, Any]:
    query_batch = [CARD_EXAMPLE["query"], "How are neural networks optimized?", "How are derivatives computed?"]
    query_single = backend.encode_queries([CARD_EXAMPLE["query"]], batch_size=1)[0]
    query_mixed = backend.encode_queries(query_batch, batch_size=len(query_batch))[0]
    documents_single = backend.encode_documents(CARD_EXAMPLE["documents"], batch_size=1)
    documents_mixed = backend.encode_documents(CARD_EXAMPLE["documents"], batch_size=len(CARD_EXAMPLE["documents"]))
    single_scores = [normalized_cosine(query_single, vector) for vector in documents_single]
    mixed_scores = [normalized_cosine(query_mixed, vector) for vector in documents_mixed]
    single_order = sorted(range(3), key=lambda index: single_scores[index], reverse=True)
    mixed_order = sorted(range(3), key=lambda index: mixed_scores[index], reverse=True)
    score_deltas = [abs(left - right) for left, right in zip(single_scores, mixed_scores)]
    return {
        "single_scores": single_scores,
        "mixed_scores": mixed_scores,
        "max_score_delta": max(score_deltas),
        "single_order": single_order,
        "mixed_order": mixed_order,
        "passed": single_order == mixed_order and max(score_deltas) <= THRESHOLDS["max_card_batch_score_delta"],
    }


def run_smoke(backend: Any, documents: list[dict[str, Any]], queries: list[str], *, batch_size: int) -> dict[str, Any]:
    selected = select_documents(documents)
    document_texts = [str(document["text"]) for document in selected]
    document_single = backend.encode_documents(document_texts, batch_size=1)
    document_batch = backend.encode_documents(document_texts, batch_size=batch_size)
    document_shuffled = list(document_texts)
    random.Random(17).shuffle(document_shuffled)
    shuffled_vectors = backend.encode_documents(document_shuffled, batch_size=batch_size)
    by_text = dict(zip(document_shuffled, shuffled_vectors))
    document_permuted = [by_text[text] for text in document_texts]

    query_single = backend.encode_queries(queries, batch_size=1)
    query_batch = backend.encode_queries(queries, batch_size=batch_size)
    query_shuffled = list(queries)
    random.Random(23).shuffle(query_shuffled)
    shuffled_query_vectors = backend.encode_queries(query_shuffled, batch_size=batch_size)
    query_by_text = dict(zip(query_shuffled, shuffled_query_vectors))
    query_permuted = [query_by_text[text] for text in queries]

    document_batch_stats = compare_vectors(document_single, document_batch)
    query_batch_stats = compare_vectors(query_single, query_batch)
    document_permutation_stats = compare_vectors(document_single, document_permuted)
    query_permutation_stats = compare_vectors(query_single, query_permuted)
    norm_stats = vector_stats(document_single + document_batch + query_single + query_batch)
    card = card_batch_check(backend)
    passed = all(
        (
            document_batch_stats["min_cosine"] >= THRESHOLDS["min_batch_invariance_cosine"],
            query_batch_stats["min_cosine"] >= THRESHOLDS["min_batch_invariance_cosine"],
            document_permutation_stats["min_cosine"] >= THRESHOLDS["min_permutation_invariance_cosine"],
            query_permutation_stats["min_cosine"] >= THRESHOLDS["min_permutation_invariance_cosine"],
            norm_stats["max_norm_error"] <= THRESHOLDS["max_norm_error"],
            card["passed"],
        )
    )
    return {
        "passed": passed,
        "batch_size": batch_size,
        "thresholds": THRESHOLDS,
        "padding_side": backend.padding_side,
        "documents": [{"document_id": str(document["document_id"]), "text_sha256": sha256_text(str(document["text"])), "characters": len(str(document["text"]))} for document in selected],
        "queries": [{"text_sha256": sha256_text(query), "characters": len(query)} for query in queries],
        "document_batch_invariance": document_batch_stats,
        "query_batch_invariance": query_batch_stats,
        "document_permutation_invariance": document_permutation_stats,
        "query_permutation_invariance": query_permutation_stats,
        "norms": norm_stats,
        "model_card_batch_check": card,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model-cache", default="models/weights/huggingface")
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--labels", default="data/evals/retrieval-quality-heldout-v1.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="data/generated/zembed-padding-parity-v1/smoke.json")
    args = parser.parse_args()
    backend = ZembedMLXBackend(args.model, model_cache=ROOT / args.model_cache, batch_size=args.batch_size)
    labels = read_json(ROOT / args.labels)
    queries = [label["query"] for label in labels["queries"] if not label.get("evidence_absent", False)][:8]
    result = {"contract": backend.contract_metadata(), "smoke": run_smoke(backend, read_json(ROOT / args.documents), queries, batch_size=args.batch_size)}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    if not result["smoke"]["passed"]:
        raise SystemExit("zembed padding parity smoke test failed")


if __name__ == "__main__":
    main()
