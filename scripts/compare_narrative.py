"""Compare a generated post with a holdout post without using the holdout to generate."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

from build_markdown_content import narrative_roles
from pipeline_common import write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I’ve|I've|my|me|we|our|us)\b", re.IGNORECASE)
UNCERTAINTY_RE = re.compile(r"\b(maybe|could|likely|uncertain|appears|suggests|not sure|unknown)\b", re.IGNORECASE)
DISCARDED_RE = re.compile(r"\b(discard\w*|reject\w*|abandon\w*|not enough|does not solve)\b", re.IGNORECASE)
CAUSAL_RE = re.compile(r"\b(because|therefore|instead|so|which means|as a result|this supports)\b", re.IGNORECASE)
MEASUREMENT_RE = re.compile(r"\b(mean|stddev|correlation|histogram|variance|diff|residual|metric|measurement)\b", re.IGNORECASE)


def paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def narrative_chunks(text: str) -> list[dict[str, Any]]:
    """Keep paragraph-sized evidence while carrying the nearest Markdown heading."""
    chunks: list[dict[str, Any]] = []
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        if not body:
            return
        for paragraph in paragraphs("\n".join(body)):
            if re.fullmatch(r"[-|: ]+", paragraph):
                continue
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "heading": heading,
                    "text": paragraph,
                    "kind": classify_chunk(paragraph),
                }
            )
        body = []

    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            flush()
            heading = match.group(1)
        else:
            body.append(line)
    flush()
    return chunks


def classify_chunk(text: str) -> str:
    stripped = text.strip()
    words = TOKEN_RE.findall(stripped)
    if (
        stripped.startswith("```")
        or stripped.startswith("|")
        or " -> " in stripped
        or "\\" in stripped
        or sum(character in "|`{}[]$" for character in stripped)
        > max(4, len(stripped) * 0.08)
    ):
        return "technical"
    if len(words) < 6:
        return "label"
    return "prose"


def ngrams(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    return {tuple(tokens[index:index + size]) for index in range(max(0, len(tokens) - size + 1))}


def longest_common_subsequence(left: list[str], right: list[str]) -> int:
    row = [0] * (len(right) + 1)
    for left_item in left:
        previous = 0
        for index, right_item in enumerate(right, start=1):
            saved = row[index]
            if left_item == right_item:
                row[index] = previous + 1
            else:
                row[index] = max(row[index], row[index - 1])
            previous = saved
    return row[-1]


def profile(text: str) -> dict[str, Any]:
    parts = paragraphs(text)
    role_sequence: list[str] = []
    role_counts: dict[str, int] = {}
    for part in parts:
        roles = narrative_roles(part, None)
        primary = roles[0]
        role_sequence.append(primary)
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "word_count": len(TOKEN_RE.findall(text)),
        "paragraph_count": len(parts),
        "heading_count": len(re.findall(r"^#{1,6}\s+", text, re.MULTILINE)),
        "first_person_count": len(FIRST_PERSON_RE.findall(text)),
        "uncertainty_count": len(UNCERTAINTY_RE.findall(text)),
        "discarded_approach_count": len(DISCARDED_RE.findall(text)),
        "causal_transition_count": len(CAUSAL_RE.findall(text)),
        "measurement_term_count": len(MEASUREMENT_RE.findall(text)),
        "role_counts": role_counts,
        "role_sequence": role_sequence,
        "first_person_opening": bool(FIRST_PERSON_RE.search("\n\n".join(parts[:3]))),
    }


def _rate_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_words = max(1, left["word_count"])
    right_words = max(1, right["word_count"])
    keys = (
        "first_person_count",
        "uncertainty_count",
        "discarded_approach_count",
        "causal_transition_count",
        "measurement_term_count",
    )
    similarities = []
    for key in keys:
        left_rate = 100.0 * left[key] / left_words
        right_rate = 100.0 * right[key] / right_words
        similarities.append(1.0 / (1.0 + abs(left_rate - right_rate)))
    return sum(similarities) / len(similarities)


def chunk_similarity(candidate: str, reference: str) -> dict[str, float]:
    candidate_profile = profile(candidate)
    reference_profile = profile(reference)
    candidate_roles = set(candidate_profile["role_counts"])
    reference_roles = set(reference_profile["role_counts"])
    role_union = candidate_roles | reference_roles
    role_similarity = (
        len(candidate_roles & reference_roles) / len(role_union)
        if role_union
        else 1.0
    )
    candidate_terms = {token.casefold() for token in TOKEN_RE.findall(candidate)}
    reference_terms = {token.casefold() for token in TOKEN_RE.findall(reference)}
    term_union = candidate_terms | reference_terms
    lexical_similarity = (
        len(candidate_terms & reference_terms) / len(term_union)
        if term_union
        else 1.0
    )
    length_similarity = min(
        candidate_profile["word_count"], reference_profile["word_count"]
    ) / max(1, candidate_profile["word_count"], reference_profile["word_count"])
    style_signal_similarity = _rate_similarity(candidate_profile, reference_profile)
    score = (
        0.35 * role_similarity
        + 0.25 * style_signal_similarity
        + 0.20 * length_similarity
        + 0.20 * lexical_similarity
    )
    return {
        "score": round(score, 4),
        "role_similarity": round(role_similarity, 4),
        "style_signal_similarity": round(style_signal_similarity, 4),
        "length_similarity": round(length_similarity, 4),
        "lexical_similarity": round(lexical_similarity, 4),
    }


def chunkwise_compare(candidate_text: str, reference_text: str) -> dict[str, Any]:
    candidate_chunks = narrative_chunks(candidate_text)
    reference_chunks = narrative_chunks(reference_text)
    matches = []
    for reference in reference_chunks:
        candidates = [
            (chunk_similarity(candidate["text"], reference["text"]), candidate)
            for candidate in candidate_chunks
        ]
        similarity, candidate = max(
            candidates,
            key=lambda item: (item[0]["score"], -item[1]["chunk_index"]),
            default=(
                {
                    "score": 0.0,
                    "role_similarity": 0.0,
                    "style_signal_similarity": 0.0,
                    "length_similarity": 0.0,
                    "lexical_similarity": 0.0,
                },
                {"chunk_index": None, "heading": None, "text": ""},
            ),
        )
        matches.append(
            {
                "reference_chunk": reference,
                "candidate_chunk": candidate,
                "similarity": similarity,
            }
        )
    scores = [
        match["similarity"]["score"]
        for match in matches
        if match["reference_chunk"]["kind"] == "prose"
    ]
    by_kind = {}
    for kind in ("prose", "technical", "label"):
        kind_scores = [
            match["similarity"]["score"]
            for match in matches
            if match["reference_chunk"]["kind"] == kind
        ]
        by_kind[kind] = {
            "count": len(kind_scores),
            "mean_best_chunk_score": round(
                sum(kind_scores) / max(1, len(kind_scores)), 4
            ),
            "coverage_at_0_45": round(
                sum(score >= 0.45 for score in kind_scores) / max(1, len(kind_scores)),
                4,
            ),
        }
    return {
        "candidate_chunk_count": len(candidate_chunks),
        "reference_chunk_count": len(reference_chunks),
        "mean_best_chunk_score": round(sum(scores) / max(1, len(scores)), 4),
        "reference_chunk_coverage_at_0_45": round(
            sum(score >= 0.45 for score in scores) / max(1, len(scores)), 4
        ),
        "by_kind": by_kind,
        "matches": matches,
    }


def compare(generated_text: str, reference_text: str) -> dict[str, Any]:
    generated = profile(generated_text)
    reference = profile(reference_text)
    generated_roles = set(generated["role_counts"])
    reference_roles = set(reference["role_counts"])
    role_union = generated_roles | reference_roles
    role_jaccard = len(generated_roles & reference_roles) / len(role_union) if role_union else 1.0
    generated_sequence = generated["role_sequence"]
    reference_sequence = reference["role_sequence"]
    lcs = longest_common_subsequence(generated_sequence, reference_sequence)
    sequence_similarity = lcs / max(1, len(reference_sequence))
    reference_ngrams = ngrams(reference_text)
    generated_ngrams = ngrams(generated_text)
    overlap = generated_ngrams & reference_ngrams
    return {
        "generated": generated,
        "reference": reference,
        "similarity": {
            "role_jaccard": round(role_jaccard, 4),
            "role_sequence_lcs_ratio": round(sequence_similarity, 4),
            "shared_fivegram_count": len(overlap),
            "shared_fivegram_ratio_of_generated": round(
                len(overlap) / max(1, len(generated_ngrams)), 4
            ),
        },
        "interpretation": {
            "narrative_structure_is_approximate": role_jaccard >= 0.45 and sequence_similarity >= 0.35,
            "possible_phrase_overfit": len(overlap) >= 5,
            "personal_content_requires_review": generated["first_person_count"] > 0,
        },
        "chunkwise": chunkwise_compare(generated_text, reference_text),
    }


def compare_versions(
    generated_text: str,
    previous_text: str,
    reference_text: str,
) -> dict[str, Any]:
    generated = compare(generated_text, reference_text)
    previous = compare(previous_text, reference_text)
    decisions = []
    wins = {"generated": 0, "previous": 0, "tie": 0}
    wins_by_kind = {
        kind: {"generated": 0, "previous": 0, "tie": 0}
        for kind in ("prose", "technical", "label")
    }
    for new_match, old_match in zip(
        generated["chunkwise"]["matches"],
        previous["chunkwise"]["matches"],
    ):
        delta = round(
            new_match["similarity"]["score"]
            - old_match["similarity"]["score"],
            4,
        )
        winner = "generated" if delta > 0.03 else "previous" if delta < -0.03 else "tie"
        wins[winner] += 1
        kind = new_match["reference_chunk"]["kind"]
        wins_by_kind[kind][winner] += 1
        decisions.append(
            {
                "reference_chunk_index": new_match["reference_chunk"]["chunk_index"],
                "reference_heading": new_match["reference_chunk"]["heading"],
                "reference_kind": kind,
                "generated_match": new_match,
                "previous_match": old_match,
                "score_delta": delta,
                "winner": winner,
            }
        )
    return {
        "generated_vs_reference": generated,
        "previous_vs_reference": previous,
        "per_reference_chunk": decisions,
        "chunk_wins": wins,
        "chunk_wins_by_kind": wins_by_kind,
        "mean_best_chunk_score_delta": round(
            generated["chunkwise"]["mean_best_chunk_score"]
            - previous["chunkwise"]["mean_best_chunk_score"],
            4,
        ),
        "coverage_delta_at_0_45": round(
            generated["chunkwise"]["reference_chunk_coverage_at_0_45"]
            - previous["chunkwise"]["reference_chunk_coverage_at_0_45"],
            4,
        ),
    }


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def semantic_version_comparison(
    generated_text: str,
    previous_text: str,
    reference_text: str,
    *,
    model_name: str,
    device: str | None,
) -> dict[str, Any]:
    """Compare each holdout chunk semantically after generation has been frozen."""
    from embed_chunks import QwenEmbeddingBackend

    generated_chunks = narrative_chunks(generated_text)
    previous_chunks = narrative_chunks(previous_text)
    reference_chunks = narrative_chunks(reference_text)
    texts = list(
        dict.fromkeys(
            chunk["text"]
            for chunks in (generated_chunks, previous_chunks, reference_chunks)
            for chunk in chunks
        )
    )
    backend = QwenEmbeddingBackend(model_name, device=device)
    vectors = backend.encode(texts, batch_size=1)
    vector_by_text = dict(zip(texts, vectors))

    rows = []
    wins_by_kind = {
        kind: {"generated": 0, "previous": 0, "tie": 0}
        for kind in ("prose", "technical", "label")
    }
    for reference in reference_chunks:
        reference_vector = vector_by_text[reference["text"]]
        generated_score, generated_chunk = max(
            [
                (
                    cosine(reference_vector, vector_by_text[chunk["text"]]),
                    chunk,
                )
                for chunk in generated_chunks
            ],
            key=lambda item: item[0],
        )
        previous_score, previous_chunk = max(
            [
                (
                    cosine(reference_vector, vector_by_text[chunk["text"]]),
                    chunk,
                )
                for chunk in previous_chunks
            ],
            key=lambda item: item[0],
        )
        delta = generated_score - previous_score
        winner = "generated" if delta > 0.03 else "previous" if delta < -0.03 else "tie"
        wins_by_kind[reference["kind"]][winner] += 1
        rows.append(
            {
                "reference_chunk": reference,
                "generated_chunk": generated_chunk,
                "previous_chunk": previous_chunk,
                "generated_score": round(generated_score, 4),
                "previous_score": round(previous_score, 4),
                "score_delta": round(delta, 4),
                "winner": winner,
            }
        )
    prose_rows = [row for row in rows if row["reference_chunk"]["kind"] == "prose"]
    return {
        "model": model_name,
        "method": "best_cosine_match_per_holdout_chunk",
        "prose_chunk_wins": wins_by_kind["prose"],
        "wins_by_kind": wins_by_kind,
        "generated_prose_mean": round(
            sum(row["generated_score"] for row in prose_rows) / max(1, len(prose_rows)),
            4,
        ),
        "previous_prose_mean": round(
            sum(row["previous_score"] for row in prose_rows) / max(1, len(prose_rows)),
            4,
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--previous")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device")
    parser.add_argument("--output", default="data/generated/narrative_comparison.json")
    args = parser.parse_args()
    generated_text = Path(args.generated).read_text(encoding="utf-8")
    reference_text = Path(args.reference).read_text(encoding="utf-8")
    report = (
        compare_versions(
            generated_text,
            Path(args.previous).read_text(encoding="utf-8"),
            reference_text,
        )
        if args.previous
        else compare(generated_text, reference_text)
    )
    if args.semantic:
        if not args.previous:
            raise ValueError("--semantic currently requires --previous")
        report["semantic_chunk_comparison"] = semantic_version_comparison(
            generated_text,
            Path(args.previous).read_text(encoding="utf-8"),
            reference_text,
            model_name=args.embedding_model,
            device=args.device,
        )
    write_json(args.output, report)
    print(f"Narrative report: {args.output}")
    current = report.get("generated_vs_reference", report)
    print(f"Role Jaccard: {current['similarity']['role_jaccard']}")
    print(f"Role sequence similarity: {current['similarity']['role_sequence_lcs_ratio']}")
    print(f"Shared five-grams: {current['similarity']['shared_fivegram_count']}")
    print(f"Mean best-chunk score: {current['chunkwise']['mean_best_chunk_score']}")
    print(f"Phrase-overfit flag: {current['interpretation']['possible_phrase_overfit']}")
    if args.previous:
        print(f"Chunk wins: {report['chunk_wins']}")
        print(f"Prose chunk wins: {report['chunk_wins_by_kind']['prose']}")
    if args.semantic:
        print(
            "Semantic prose chunk wins: "
            f"{report['semantic_chunk_comparison']['prose_chunk_wins']}"
        )


if __name__ == "__main__":
    main()
