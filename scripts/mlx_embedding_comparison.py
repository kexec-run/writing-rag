"""Compare local MLX embedding snapshots and a prior Core ML artifact.

This is an isolated experiment. It does not modify production vectors,
embedding metadata, or the SentenceTransformers backend. Qwen's MLX causal
model is used only up to its normalized transformer hidden state; vocabulary
logits are never treated as embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXTS = PROJECT_ROOT / "data/generated/ane-embedding-v1/texts.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/mlx-embedding-comparison-v1"
DEFAULT_ANE_VECTORS = PROJECT_ROOT / "data/generated/ane-wordwise-qwen3-4b-v1/coreml-vectors.npz"


def load_texts(path: Path, limit: int | None) -> list[str]:
    texts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError(f"Expected a JSON list of strings: {path}")
    return texts[:limit] if limit is not None else texts


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def pairwise_similarity(vectors: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(vectors)
    return normalized @ normalized.T


def ranking_overlap(left: np.ndarray, right: np.ndarray, top_k: int = 3) -> float:
    if left.shape != right.shape or len(left) < 2:
        return 0.0
    left_scores = pairwise_similarity(left)
    right_scores = pairwise_similarity(right)
    overlaps: list[float] = []
    for index in range(len(left)):
        left_scores[index, index] = -math.inf
        right_scores[index, index] = -math.inf
        left_top = set(np.argsort(-left_scores[index], kind="stable")[:top_k])
        right_top = set(np.argsort(-right_scores[index], kind="stable")[:top_k])
        overlaps.append(len(left_top & right_top) / top_k)
    return float(np.mean(overlaps))


def encode_mlx(
    model_path: Path,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode text using the transformer hidden state exposed by mlx-lm."""
    import mlx.core as mx
    import mlx_lm

    load_started = time.perf_counter()
    model, tokenizer = mlx_lm.load(str(model_path))
    load_seconds = time.perf_counter() - load_started

    tokenized = [tokenizer.encode(text)[:max_length] for text in texts]
    if not tokenized or any(not ids for ids in tokenized):
        raise ValueError("MLX tokenizer produced an empty input")

    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    batches: list[np.ndarray] = []
    inference_started = time.perf_counter()
    for start in range(0, len(tokenized), batch_size):
        batch_ids = tokenized[start : start + batch_size]
        lengths = np.asarray([len(ids) for ids in batch_ids], dtype=np.int32)
        width = int(lengths.max())
        input_ids = np.full((len(batch_ids), width), pad_token_id, dtype=np.int32)
        for row, ids in enumerate(batch_ids):
            input_ids[row, : len(ids)] = ids

        hidden = model.model(mx.array(input_ids))
        mx.eval(hidden)
        row_indices = mx.arange(len(batch_ids))
        last_indices = mx.array(lengths - 1)
        pooled = hidden[row_indices, last_indices, :]
        pooled = pooled.astype(mx.float32)
        mx.eval(pooled)
        batches.append(np.array(pooled, dtype=np.float32))

    vectors = normalize_rows(np.vstack(batches).astype(np.float32, copy=False))
    inference_seconds = time.perf_counter() - inference_started
    metadata = {
        "runtime": "mlx",
        "resolved_model": str(model_path),
        "batch_size": batch_size,
        "max_length": max_length,
        "text_count": len(texts),
        "vector_dimension": int(vectors.shape[1]),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_text": inference_seconds / len(texts),
        "pooling": "last_valid_token_transformer_hidden_state",
        "normalization": "l2",
    }
    return vectors, metadata


def compare_to_ane(vectors: np.ndarray, ane_vectors_path: Path) -> dict[str, Any]:
    payload = np.load(ane_vectors_path)
    ane_vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if vectors.shape != ane_vectors.shape:
        return {
            "available": False,
            "reason": "shape_mismatch",
            "mlx_shape": list(vectors.shape),
            "ane_shape": list(ane_vectors.shape),
        }
    row_cosines = [cosine(left, right) for left, right in zip(vectors, ane_vectors)]
    return {
        "available": True,
        "ane_vectors_path": str(ane_vectors_path),
        "ane_artifact_runtime": "Core ML CPU_AND_NE; ANE activity not independently proven",
        "mean_row_cosine": float(np.mean(row_cosines)),
        "minimum_row_cosine": float(np.min(row_cosines)),
        "maximum_row_cosine": float(np.max(row_cosines)),
        "mean_absolute_delta": float(np.mean(np.abs(vectors - ane_vectors))),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MLX embedding comparison",
        "",
        "This is an isolated comparison. No production retrieval vectors were changed.",
        "",
        "## Runs",
        "",
        "| Label | Dimension | Texts | Load seconds | Inference seconds | Seconds/text |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in report["runs"].items():
        lines.append(
            f"| {label} | {row['vector_dimension']} | {row['text_count']} | "
            f"{row['model_load_seconds']:.3f} | {row['inference_seconds']:.3f} | "
            f"{row['seconds_per_text']:.3f} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for label, comparison in report["comparisons"].items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(comparison, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--texts", type=Path, default=DEFAULT_TEXTS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ane-vectors", type=Path, default=DEFAULT_ANE_VECTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch size and max length must be positive")

    texts = load_texts(args.texts, args.limit)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "texts_path": str(args.texts),
        "text_count": len(texts),
        "runs": {},
        "comparisons": {},
    }
    vectors_by_label: dict[str, np.ndarray] = {}
    for label, raw_path in args.model:
        model_path = Path(raw_path).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        vectors, metadata = encode_mlx(
            model_path,
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        safe_label = label.replace("/", "_").replace(" ", "-")
        np.savez_compressed(output_dir / f"{safe_label}-vectors.npz", vectors=vectors)
        (output_dir / f"{safe_label}-metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report["runs"][label] = metadata
        vectors_by_label[label] = vectors

    labels = list(vectors_by_label)
    for index, left_label in enumerate(labels):
        left = vectors_by_label[left_label]
        report["comparisons"][f"{left_label}_internal"] = {
            "mean_off_diagonal_cosine": float(
                (pairwise_similarity(left).sum() - len(left)) / max(len(left) * (len(left) - 1), 1)
            ),
        }
        if left.shape[1] == 2560 and args.ane_vectors.exists():
            report["comparisons"][f"{left_label}_vs_prior_ane"] = compare_to_ane(left, args.ane_vectors)
        for right_label in labels[index + 1 :]:
            right = vectors_by_label[right_label]
            if left.shape == right.shape:
                report["comparisons"][f"{left_label}_vs_{right_label}"] = {
                    "mean_row_cosine": float(np.mean([cosine(a, b) for a, b in zip(left, right)])),
                    "ranking_overlap_at_3": ranking_overlap(left, right),
                }

    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "comparison.md", report)
    print(f"report={output_dir / 'comparison.md'}")
    print(f"json={output_dir / 'comparison.json'}")
    for label, metadata in report["runs"].items():
        print(f"{label}_dimension={metadata['vector_dimension']}")
        print(f"{label}_inference_seconds={metadata['inference_seconds']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
