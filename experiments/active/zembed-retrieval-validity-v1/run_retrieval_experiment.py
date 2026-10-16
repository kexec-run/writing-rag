"""Run the isolated zembed MLX card-contract and retrieval experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")
LEXICAL_STOPWORDS = frozenset(
    """
    a about after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more
    most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with you your yours yourself yourselves truly
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def lexical_terms(text: str) -> set[str]:
    return {term for term in tokenize(text) if term not in LEXICAL_STOPWORDS}


def is_content_evidence_candidate(document: dict[str, Any]) -> bool:
    metadata = document.get("metadata", {})
    return not (
        metadata.get("source", "").casefold() == "chatgpt"
        and not str(metadata.get("self_message_ids", "")).strip()
    )


def lexical_overlap(query: str, text: str) -> float:
    query_terms = lexical_terms(query)
    document_terms = lexical_terms(text)
    if not query_terms:
        return 0.0
    return sum(term in document_terms for term in query_terms) / len(query_terms)

BACKEND_PATH = Path(__file__).with_name("zembed_mlx_backend.py")
spec = importlib.util.spec_from_file_location("zembed_mlx_backend", BACKEND_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load zembed adapter")
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)
ZembedMLXBackend = backend_module.ZembedMLXBackend
MODEL_NAME = backend_module.MODEL_NAME

MODEL_CARD_EXAMPLE = {
    "query": "What is backpropagation?",
    "documents": [
        "Backpropagation is a fundamental algorithm for training neural networks by computing gradients.",
        "Gradient descent is used to optimize model parameters during the training process.",
        "Neural network training relies on efficient computation of derivatives through backpropagation.",
    ],
    "published_similarities": [0.7525, 0.5670, 0.6835],
}
K_VALUES = (10, 50, 100)
MODES = ("dense_only", "hybrid")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def ranking_metrics(ranking: list[dict[str, Any]], relevant_ids: set[str], k: int) -> dict[str, float]:
    top = ranking[:k]
    hits = [row["document_id"] in relevant_ids for row in top]
    dcg = sum(1.0 / math.log2(index + 1) for index, hit in enumerate(hits, 1) if hit)
    ideal = min(k, len(relevant_ids))
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal + 1))
    return {
        f"precision_at_{k}": sum(hits) / k,
        f"recall_at_{k}": sum(hits) / len(relevant_ids) if relevant_ids else 0.0,
        "mrr": next((1.0 / index for index, hit in enumerate(hits, 1) if hit), 0.0),
        f"ndcg_at_{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def aggregate(rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    eligible = [row for row in rows if not row["evidence_absent"]]
    return {key: sum(float(row["metrics"][str(k)][key]) for row in eligible) / len(eligible) for key in (f"precision_at_{k}", f"recall_at_{k}", "mrr", f"ndcg_at_{k}")}


def model_card_check(backend: Any) -> dict[str, Any]:
    example = MODEL_CARD_EXAMPLE
    query = backend.encode_queries([example["query"]], batch_size=1)[0]
    documents = backend.encode_documents(example["documents"], batch_size=1)
    similarities = [cosine(query, vector) for vector in documents]
    errors = [abs(actual - expected) for actual, expected in zip(similarities, example["published_similarities"])]
    order = sorted(range(3), key=lambda index: similarities[index], reverse=True)
    return {
        "published_similarities": example["published_similarities"],
        "observed_similarities": similarities,
        "absolute_errors": errors,
        "same_order": order == [0, 2, 1],
        "max_absolute_error": max(errors),
        "mean_absolute_error": sum(errors) / len(errors),
        "passed": order == [0, 2, 1] and max(errors) <= 0.03 and sum(errors) / len(errors) <= 0.02,
    }


def _eligible_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [document for document in documents if document.get("collection") == "content_memory" and document.get("view") == "content" and is_content_evidence_candidate(document)]


def _text_list_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _document_identity(eligible: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(document["document_id"]) for document in eligible]
    texts = [str(document["text"]) for document in eligible]
    return {
        "count": len(eligible),
        "document_ids_sha256": _text_list_hash(ids),
        "document_texts_sha256": _text_list_hash(texts),
    }


def encode_documents_resumable(
    backend: Any,
    eligible: list[dict[str, Any]],
    *,
    batch_size: int,
    resume_dir: Path,
    length_bucketed: bool = False,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Encode the fixed document substrate with fail-closed batch checkpoints."""
    import numpy as np

    texts = [str(document["text"]) for document in eligible]
    identity = _document_identity(eligible)
    contract = backend.contract_metadata()
    long_text_singleton_threshold = 8000
    batch_ordering = "length_sorted_long_singletons" if length_bucketed else "corpus_order"
    manifest_path = resume_dir / "manifest.json"
    expected = {
        "schema_version": 1,
        "role": "document",
        "batch_size": batch_size,
        "batch_ordering": batch_ordering,
        "long_text_singleton_threshold": long_text_singleton_threshold if length_bucketed else None,
        "dimension": int(contract["dimension"]),
        "backend": contract["backend"],
        "model_name": contract["model_name"],
        "resolved_model": contract["resolved_model"],
        "snapshot_files": contract["snapshot_files"],
        **identity,
    }
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if {key: manifest.get(key) for key in expected} != expected:
            raise RuntimeError("document resume manifest does not match the frozen experiment contract")
    else:
        manifest = {**expected, "batches": {}}
        _atomic_write_json(manifest_path, manifest)

    batches_dir = resume_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_records: dict[str, Any] = manifest.setdefault("batches", {})
    resumed_from_checkpoint = bool(batch_records)
    ordered_indices = list(range(len(texts)))
    if length_bucketed:
        ordered_indices.sort(key=lambda index: (len(texts[index]), index))
        batches: list[list[int]] = []
        current: list[int] = []
        for index in ordered_indices:
            if len(texts[index]) > long_text_singleton_threshold:
                if current:
                    batches.append(current)
                    current = []
                batches.append([index])
            else:
                current.append(index)
                if len(current) == batch_size:
                    batches.append(current)
                    current = []
        if current:
            batches.append(current)
    else:
        batches = [
            ordered_indices[start : min(start + batch_size, len(ordered_indices))]
            for start in range(0, len(ordered_indices), batch_size)
        ]
    batch_offsets: list[tuple[int, int, list[int]]] = []
    cursor = 0
    for indices in batches:
        batch_offsets.append((cursor, cursor + len(indices), indices))
        cursor += len(indices)
    total_batches = len(batch_offsets)
    vectors: list[list[float] | None] = [None] * len(texts)

    for batch_index, (start, end, indices) in enumerate(batch_offsets):
        key = str(batch_index)
        batch_path = batches_dir / f"batch-{batch_index:05d}.npy"
        record = batch_records.get(key)
        if record is not None and batch_path.is_file():
            if record.get("start") != start or record.get("end") != end or record.get("indices") != indices:
                raise RuntimeError(f"document resume batch {batch_index} has invalid bounds or ordering")
            if sha256_file(batch_path) != record.get("sha256"):
                raise RuntimeError(f"document resume batch {batch_index} failed checksum validation")
            array = np.load(batch_path, allow_pickle=False)
        else:
            if record is not None or batch_path.exists():
                raise RuntimeError(f"document resume batch {batch_index} is incomplete")
            batch_texts = [texts[index] for index in indices]
            print(
                f"encoding document batch {batch_index + 1}/{total_batches} "
                f"({start}:{end}, max_chars={max(map(len, batch_texts))})",
                flush=True,
            )
            encoded = backend.encode_documents(batch_texts, batch_size=batch_size)
            array = np.asarray(encoded, dtype=np.float32)
            if array.shape != (end - start, int(contract["dimension"])):
                raise RuntimeError(f"document batch {batch_index} has unexpected shape {array.shape}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"document batch {batch_index} contains non-finite values")
            temporary = batch_path.with_name(f".{batch_path.name}.tmp.npy")
            np.save(temporary, array, allow_pickle=False)
            temporary.replace(batch_path)
            record = {
                "start": start,
                "end": end,
                "indices": indices,
                "shape": list(array.shape),
                "sha256": sha256_file(batch_path),
            }
            batch_records[key] = record
            _atomic_write_json(manifest_path, manifest)
        if tuple(array.shape) != (end - start, int(contract["dimension"])):
            raise RuntimeError(f"document resume batch {batch_index} has unexpected stored shape {array.shape}")
        for local_index, original_index in enumerate(indices):
            if vectors[original_index] is not None:
                raise RuntimeError(f"document resume batch {batch_index} repeats original index {original_index}")
            vectors[original_index] = array[local_index].tolist()

    if any(vector is None for vector in vectors):
        raise RuntimeError("document resume did not produce a vector for every document")
    manifest["completed_batches"] = len(batch_records)
    _atomic_write_json(manifest_path, manifest)
    return [vector for vector in vectors if vector is not None], {
        "resume_dir": str(resume_dir),
        "batch_count": total_batches,
        "batch_ordering": batch_ordering,
        "resumed": resumed_from_checkpoint,
    }


def _average_precision(ranking: list[dict[str, Any]], positive_ids: set[str]) -> float:
    hits = 0
    total = 0.0
    for index, row in enumerate(ranking, 1):
        if row["candidate_id"] in positive_ids:
            hits += 1
            total += hits / index
    return total / len(positive_ids) if positive_ids else 0.0


def run_diagnostic(backend: Any, cases: list[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    unique_texts = list(dict.fromkeys(candidate["text"] for case in cases for candidate in case["candidates"]))
    passage_vectors = backend.encode_documents(unique_texts, batch_size=batch_size)
    vector_by_text = dict(zip(unique_texts, passage_vectors))
    query_vectors = backend.encode_queries([case["query"] for case in cases], batch_size=batch_size)
    rows = []
    for case, query_vector in zip(cases, query_vectors):
        ranked = []
        for candidate in case["candidates"]:
            ranked.append({"candidate_id": candidate["candidate_id"], "role": candidate["role"], "cosine": cosine(query_vector, vector_by_text[candidate["text"]])})
        ranked.sort(key=lambda row: (-row["cosine"], row["candidate_id"]))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        positives = {candidate["candidate_id"] for candidate in case["candidates"] if candidate["role"] == "positive"}
        hard = {candidate["candidate_id"] for candidate in case["candidates"] if candidate["role"] == "hard_negative"}
        positive_ranks = [row["rank"] for row in ranked if row["candidate_id"] in positives]
        hard_ranks = [row["rank"] for row in ranked if row["candidate_id"] in hard]
        first_positive = min(positive_ranks) if positive_ranks else None
        hard_before = sum(rank < first_positive for rank in hard_ranks) if first_positive is not None else 0
        rows.append({"case_id": case["case_id"], "suite": case.get("suite"), "category": case.get("category"), "query": case["query"], "ranking": ranked, "metrics": {"first_positive_rank": first_positive, "reciprocal_rank": 1.0 / first_positive if first_positive else 0.0, "average_precision": _average_precision(ranked, positives), "hard_negatives_above_first_positive": hard_before}})
    return {"case_count": len(rows), "rows": rows, "aggregate": diagnostic_aggregate(rows)}


def diagnostic_aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    positive_rows = [row for row in rows if row["metrics"].get("first_positive_rank") is not None]
    if not positive_rows:
        raise ValueError("diagnostic suite has no positive cases")
    return {
        "positive_case_count": len(positive_rows),
        "reciprocal_rank_mean": sum(row["metrics"]["reciprocal_rank"] for row in positive_rows) / len(positive_rows),
        "average_precision_mean": sum(row["metrics"]["average_precision"] for row in positive_rows) / len(positive_rows),
        "first_positive_rank_1_count": sum(row["metrics"]["first_positive_rank"] == 1 for row in positive_rows),
        "hard_before_best_positive_count": sum(row["metrics"]["hard_negatives_above_first_positive"] for row in positive_rows),
    }


def compare_diagnostic(candidate: dict[str, Any], q0_path: Path) -> dict[str, Any]:
    q0 = read_json(q0_path)
    q0_rows = {row["case_id"]: row for row in q0["policy_results"]}
    effects = []
    for row in candidate["rows"]:
        before = q0_rows[row["case_id"]]
        q0_rank = before["metrics"]["first_positive_rank"]
        candidate_rank = row["metrics"]["first_positive_rank"]
        if q0_rank is None and candidate_rank is None:
            continue
        if q0_rank is None or candidate_rank is None:
            raise RuntimeError(f"diagnostic positive-rank presence changed for {row['case_id']}")
        effects.append({
            "case_id": row["case_id"],
            "q0_first_positive_rank": q0_rank,
            "zembed_first_positive_rank": candidate_rank,
            "rank_delta": candidate_rank - q0_rank,
        })
    improved = sum(effect["rank_delta"] < 0 for effect in effects)
    same = sum(effect["rank_delta"] == 0 for effect in effects)
    regressed = sum(effect["rank_delta"] > 0 for effect in effects)
    max_rank_regression = max((effect["rank_delta"] for effect in effects), default=0)
    thresholds = {
        "reciprocal_rank_mean_at_least": 0.84,
        "average_precision_mean_at_least": 0.8383333333333334,
        "first_positive_rank_1_count_at_least": 14,
        "hard_before_best_positive_count_at_most": 6,
        "improved_cases_at_least": 3,
        "regressed_cases_at_most": 1,
        "max_rank_regression_at_most": 1,
    }
    aggregate = candidate["aggregate"]
    gates = {
        "reciprocal_rank": aggregate["reciprocal_rank_mean"] >= thresholds["reciprocal_rank_mean_at_least"],
        "average_precision": aggregate["average_precision_mean"] >= thresholds["average_precision_mean_at_least"],
        "rank_1": aggregate["first_positive_rank_1_count"] >= thresholds["first_positive_rank_1_count_at_least"],
        "hard_negative_intrusion": aggregate["hard_before_best_positive_count"] <= thresholds["hard_before_best_positive_count_at_most"],
        "improved_cases": improved >= thresholds["improved_cases_at_least"],
        "regressed_cases": regressed <= thresholds["regressed_cases_at_most"],
        "maximum_rank_regression": max_rank_regression <= thresholds["max_rank_regression_at_most"],
    }
    return {
        "q0_reference": str(q0_path),
        "q0_aggregate": diagnostic_aggregate([{"metrics": row["metrics"]} for row in q0["policy_results"]]),
        "zembed_aggregate": aggregate,
        "effects": effects,
        "improved_cases": improved,
        "same_cases": same,
        "regressed_cases": regressed,
        "maximum_rank_regression": max_rank_regression,
        "thresholds": thresholds,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_status(diagnostic_comparison: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    dense = comparison["dense_only"]
    deltas = dense["deltas_at_10"]
    q0 = dense["q0"]
    zembed = dense["zembed"]
    heldout_gates = {
        "mrr_gain_at_least_0_03": deltas["mrr"] >= 0.03,
        "ndcg_gain_at_least_0_03": deltas["ndcg_at_10"] >= 0.03,
        "recall_at_10_non_decreasing": zembed["10"]["recall_at_10"] >= q0["10"]["recall_at_10"],
        "recall_at_50_non_decreasing": zembed["50"]["recall_at_50"] >= q0["50"]["recall_at_50"],
        "recall_at_100_non_decreasing": zembed["100"]["recall_at_100"] >= q0["100"]["recall_at_100"],
        "precision_at_10_non_decreasing": zembed["10"]["precision_at_10"] >= q0["10"]["precision_at_10"],
        "hard_negative_intrusion_non_increasing": dense["hard_negative_top10"]["zembed"] <= dense["hard_negative_top10"]["q0"],
    }
    diagnostic_passed = diagnostic_comparison["passed"]
    heldout_passed = all(heldout_gates.values())
    meaningful_signal = deltas["mrr"] >= 0.01 or deltas["ndcg_at_10"] >= 0.01 or diagnostic_passed
    status = "direction_supported" if diagnostic_passed and heldout_passed else "directional_only" if meaningful_signal else "not_supported"
    return {"status": status, "diagnostic_passed": diagnostic_passed, "heldout_passed": heldout_passed, "heldout_gates": heldout_gates}


def run_retrieval(
    backend: Any,
    labels: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    batch_size: int,
    resume_dir: Path | None = None,
    length_bucketed: bool = False,
) -> dict[str, Any]:
    eligible = _eligible_documents(documents)
    passage_started = time.perf_counter()
    if resume_dir is None:
        passage_vectors = backend.encode_documents([document["text"] for document in eligible], batch_size=batch_size)
        resume_metadata: dict[str, Any] = {
            "resume_dir": None,
            "batch_count": None,
            "batch_ordering": "corpus_order",
            "resumed": False,
        }
    else:
        passage_vectors, resume_metadata = encode_documents_resumable(
            backend,
            eligible,
            batch_size=batch_size,
            resume_dir=resume_dir,
            length_bucketed=length_bucketed,
        )
    passage_elapsed = time.perf_counter() - passage_started
    queries = [label["query"] for label in labels["queries"]]
    query_started = time.perf_counter()
    query_vectors = backend.encode_queries(queries, batch_size=batch_size)
    query_elapsed = time.perf_counter() - query_started
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for label, query_vector in zip(labels["queries"], query_vectors):
        scored = []
        for document, vector in zip(eligible, passage_vectors):
            dense = cosine(query_vector, vector)
            scored.append({"document_id": document["document_id"], "dense_score": dense, "lexical_score": lexical_overlap(label["query"], document["text"])})
        for mode in MODES:
            def score(row: dict[str, Any]) -> float:
                return row["dense_score"] if mode == "dense_only" else 0.65 * row["dense_score"] + 0.35 * row["lexical_score"]
            ranking = sorted(scored, key=lambda row: (-score(row), row["document_id"]))
            trace = [{**row, "score": score(row), "rank": rank, "ranking_mode": mode} for rank, row in enumerate(ranking[:100], 1)]
            absent = bool(label.get("evidence_absent", False))
            result: dict[str, Any] = {"id": str(label.get("id", label["query"])), "query": label["query"], "evidence_absent": absent, "ranking": trace}
            result["hard_negative_ranks"] = {document_id: next((row["rank"] for row in trace if row["document_id"] == document_id), None) for document_id in label.get("hard_negative_document_ids", [])}
            if not absent:
                relevant = set(label["relevant_document_ids"])
                result["relevant_document_ids"] = sorted(relevant)
                result["relevant_ranks"] = {document_id: next((row["rank"] for row in trace if row["document_id"] == document_id), None) for document_id in relevant}
                result["metrics"] = {str(k): ranking_metrics(trace, relevant, k) for k in K_VALUES}
            rows_by_mode[mode].append(result)
    eligible_ids_hash = hashlib.sha256("\n".join(document["document_id"] for document in eligible).encode()).hexdigest()
    return {"eligible_document_count": len(eligible), "document_ids_sha256": eligible_ids_hash, "passage_encode_seconds": passage_elapsed, "query_encode_seconds": query_elapsed, "resume": resume_metadata, "modes": {mode: {"queries": rows, "aggregate": {str(k): aggregate(rows, k) for k in K_VALUES}} for mode, rows in rows_by_mode.items()}}


def compare_to_q0(candidate: dict[str, Any], q0: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for mode in MODES:
        q0_rows = {row["id"]: row for row in q0["modes"][mode]["queries"]}
        candidate_rows = {row["id"]: row for row in candidate["modes"][mode]["queries"]}
        effects = []
        for query_id, row in candidate_rows.items():
            before = q0_rows[query_id]
            if row["evidence_absent"]:
                continue
            effects.append({"id": query_id, "delta": {key: row["metrics"]["10"][key] - before["metrics"]["10"][key] for key in ("mrr", "ndcg_at_10", "recall_at_10", "precision_at_10")}, "q0": before["metrics"], "zembed": row["metrics"]})
        q0_hard = sum(any(rank is not None and rank <= 10 for rank in row.get("hard_negative_ranks", {}).values()) for row in q0["modes"][mode]["queries"] if not row["evidence_absent"])
        z_hard = sum(any(rank is not None and rank <= 10 for rank in row.get("hard_negative_ranks", {}).values()) for row in candidate["modes"][mode]["queries"] if not row["evidence_absent"])
        comparison[mode] = {"q0": q0["modes"][mode]["aggregate"], "zembed": candidate["modes"][mode]["aggregate"], "deltas_at_10": {key: candidate["modes"][mode]["aggregate"]["10"][key] - q0["modes"][mode]["aggregate"]["10"][key] for key in ("mrr", "ndcg_at_10", "recall_at_10", "precision_at_10")}, "hard_negative_top10": {"q0": q0_hard, "zembed": z_hard}, "effects": effects}
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model-cache", default="models/weights/huggingface")
    parser.add_argument("--labels", default="data/evals/retrieval-quality-heldout-v1.json")
    parser.add_argument("--documents", default="data/chroma_docs/chroma_documents.json")
    parser.add_argument("--q0", default="data/generated/qwen-query-encoding-v1/held-out/Q0_plain/retrieval-results.json")
    parser.add_argument("--diagnostic", default="data/generated/qwen-embedding-characterization-v1-corrected-pilot-run5/cases/diagnostic-cases.json")
    parser.add_argument("--confirmation", default="data/generated/qwen-embedding-characterization-v1-corrected-pilot-run5/cases/confirmation-cases.json")
    parser.add_argument("--q0-diagnostic", default="data/generated/qwen-embedding-characterization-v1-corrected-pilot-run5/Q0_plain/local-results.json")
    parser.add_argument("--output", default="data/generated/zembed-retrieval-validity-v1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume-dir", default=None, help="directory for fail-closed document batch checkpoints")
    parser.add_argument(
        "--length-bucketed",
        action="store_true",
        help="sort document batches by text length while restoring original corpus order",
    )
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    backend = ZembedMLXBackend(args.model, model_cache=ROOT / args.model_cache, batch_size=args.batch_size)
    contract = backend.contract_metadata()
    card_check = model_card_check(backend)
    contract_result = {"contract": contract, "model_card_check": card_check, "passed": card_check["passed"]}
    write_json(output / "model-contract/contract.json", contract_result)
    if not card_check["passed"]:
        raise SystemExit("zembed model-card contract validation failed")
    if args.contract_only:
        return
    labels = read_json(ROOT / args.labels)
    documents = read_json(ROOT / args.documents)
    q0 = read_json(ROOT / args.q0)
    cases = read_json(ROOT / args.diagnostic)["cases"] + read_json(ROOT / args.confirmation)["cases"]
    diagnostic = run_diagnostic(backend, cases, batch_size=args.batch_size)
    started = time.perf_counter()
    resume_dir = ROOT / args.resume_dir if args.resume_dir else output / "runtime/document-embeddings"
    candidate = run_retrieval(
        backend,
        labels,
        documents,
        batch_size=args.batch_size,
        resume_dir=resume_dir,
        length_bucketed=args.length_bucketed,
    )
    comparison = compare_to_q0(candidate, q0)
    diagnostic_comparison = compare_diagnostic(diagnostic, ROOT / args.q0_diagnostic)
    decision = evaluate_status(diagnostic_comparison, comparison)
    result = {
        "contract": contract_result,
        "diagnostic": diagnostic,
        "diagnostic_comparison": diagnostic_comparison,
        "candidate": candidate,
        "comparison": comparison,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "status": decision["status"],
    }
    write_json(output / "retrieval/results.json", result)
    write_json(
        output / "run-metadata.json",
        {
            "model": args.model,
            "batch_size": args.batch_size,
            "length_bucketed": args.length_bucketed,
            "labels_sha256": sha256_file(ROOT / args.labels),
            "documents_sha256": sha256_file(ROOT / args.documents),
            "q0_reference_sha256": sha256_file(ROOT / args.q0),
            "q0_diagnostic_reference_sha256": sha256_file(ROOT / args.q0_diagnostic),
            "fallback_used": False,
            "decision": decision,
        },
    )
    print(output / "retrieval/results.json")


if __name__ == "__main__":
    main()
