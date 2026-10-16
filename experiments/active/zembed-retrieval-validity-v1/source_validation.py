"""Run Qwen versus corrected zembed on source-homogeneous B0/C1 cases.

This is a bounded exploratory screen.  Qwen and zembed are both corpus
retrievers here; no Jina reranker or lexical fusion is used.  Each source and
condition is ranked against its own frozen anchor-document pool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = ROOT / "data/generated/zembed-source-homogeneous-v1/cases.json"
DEFAULT_MESSAGES = ROOT / "data/normalized/messages.json"
DEFAULT_OUTPUT = ROOT / "data/generated/zembed-source-homogeneous-v1"
DEFAULT_CACHE = ROOT / "models/weights/huggingface"
VALIDATION_PATH = Path(__file__).with_name("instagram_validation.py")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _message_text(message: dict[str, Any]) -> str:
    return str(message.get("content") or "").strip()


def _document_text(message_ids: list[str], by_id: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        _message_text(by_id[message_id])
        for message_id in message_ids
        if message_id in by_id and _message_text(by_id[message_id])
    )


def _build_inputs(
    cases: dict[str, Any],
    messages: list[dict[str, Any]],
    screen: Any,
) -> tuple[list[str], list[str], dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    by_id = {str(message["message_id"]): message for message in messages}
    query_texts: dict[str, str] = {}
    document_texts: dict[str, str] = {}
    conditions: dict[str, dict[str, dict[str, Any]]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for source, source_case in cases["sources"].items():
        source_queries = {str(key): str(value) for key, value in source_case["queries"].items()}
        query_texts.update(source_queries)
        source_hashes[source] = {
            "query_text_hash": _hash_texts([source_queries[key] for key in sorted(source_queries)]),
        }
        source_conditions: dict[str, dict[str, Any]] = {}
        for condition, source_rows in source_case["rows"].items():
            rows = {str(row["anchor_message_id"]): row for row in source_rows}
            source_conditions[condition] = rows
            for anchor_id, row in rows.items():
                message_ids = [str(item) for item in row["primary"].get("self_message_ids", [])]
                document_id = f"{condition}:{anchor_id}"
                document_texts[document_id] = _document_text(message_ids, by_id)
            source_hashes[source][f"{condition}_document_text_hash"] = _hash_texts(
                [document_texts[f"{condition}:{anchor_id}"] for anchor_id in sorted(rows)]
            )
        conditions[source] = source_conditions
    query_ids = sorted(query_texts)
    document_ids = sorted(document_texts)
    if not all(query_texts[query_id] for query_id in query_ids):
        raise ValueError("empty query text in source cases")
    if not all(document_texts[document_id] for document_id in document_ids):
        raise ValueError("empty document text in source cases")
    return query_ids, document_ids, query_texts, conditions, source_hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    screen = _load_module(VALIDATION_PATH, "source_zembed_validation_contract")
    cases = _load(Path(args.cases))
    messages = _load(Path(args.messages))
    query_ids, document_ids, query_texts, conditions, source_hashes = _build_inputs(cases, messages, screen)
    query_text_list = [query_texts[query_id] for query_id in query_ids]
    document_texts = {}
    by_id = {str(message["message_id"]): message for message in messages}
    for source_case in cases["sources"].values():
        for condition, rows in source_case["rows"].items():
            for row in rows:
                anchor_id = str(row["anchor_message_id"])
                document_texts[f"{condition}:{anchor_id}"] = _document_text(
                    [str(item) for item in row["primary"].get("self_message_ids", [])], by_id
                )
    models = screen._retrieval_models(args, query_text_list, [document_texts[item] for item in document_ids])
    output: dict[str, Any] = {
        "schema_version": "zembed-source-homogeneous-validation-v1",
        "status": cases.get("run_status", "unknown_case_status"),
        "cases_path": str(Path(args.cases)),
        "messages_path": str(Path(args.messages)),
        "retrieval_contract": {
            "qwen_owns_retrieval": True,
            "zembed_is_alternative_retriever": True,
            "reranker_used": False,
            "lexical_fusion_used": False,
            "source_local_candidate_pools": True,
            "candidate_pool_policy": "one B0/C1 document per selected anchor within source",
            "r50_interpretation": "ceiling when source pool has <=50 documents",
        },
        "models": {
            name: {key: value for key, value in model.items() if key not in {"query_vectors", "document_vectors"}}
            for name, model in models.items()
        },
        "sources": {},
    }
    for source, source_conditions in conditions.items():
        source_query_ids = sorted(source_conditions["baseline"])
        source_output: dict[str, Any] = {
            "anchor_count": len(source_query_ids),
            "query_text_hash": source_hashes[source]["query_text_hash"],
            "conditions": {},
            "document_text_hashes": {},
        }
        query_indexes = [query_ids.index(query_id) for query_id in source_query_ids]
        for condition, rows in source_conditions.items():
            condition_document_ids = sorted(f"{condition}:{anchor_id}" for anchor_id in rows)
            document_indexes = [document_ids.index(document_id) for document_id in condition_document_ids]
            source_output["document_text_hashes"][condition] = {
                document_id: _sha256_text(document_texts[document_id])
                for document_id in condition_document_ids
            }
            source_output["conditions"][condition] = {}
            for model_name, model in models.items():
                ranking = screen._rank(
                    condition,
                    rows,
                    source_query_ids,
                    condition_document_ids,
                    [model["query_vectors"][index] for index in query_indexes],
                    [model["document_vectors"][index] for index in document_indexes],
                )
                source_output["conditions"][condition][model_name] = ranking
            source_output["conditions"][condition]["comparison"] = screen._comparison(
                source_output["conditions"][condition]["qwen"],
                source_output["conditions"][condition]["zembed"],
            )
        output["sources"][source] = source_output
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--model-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--qwen-batch-size", type=int, default=32)
    parser.add_argument("--zembed-batch-size", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    _write(Path(args.output) / "validation.json", result)
    print(json.dumps({
        source: {
            condition: {
                model: result["sources"][source]["conditions"][condition][model]["mrr"]
                for model in ("qwen", "zembed")
            }
            for condition in ("baseline", "c1_full")
        }
        for source in result["sources"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
