"""Run the writing-memory build in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)



def missing_required_inputs(project_root: Path = PROJECT_ROOT) -> list[str]:
    required = (
        Path("data/raw/reddit/reddit_chat.json"),
        Path("data/raw/whisper/transcript.json"),
    )
    return [str(path) for path in required if not (project_root / path).is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Run the optional reasoning-chain ablation stage.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("qwen3", "auto", "hashing"),
        default="qwen3",
        help="Use Qwen for the canonical retrieval path; choose auto or hashing explicitly for diagnostics.",
    )
    args = parser.parse_args()

    missing = missing_required_inputs(PROJECT_ROOT)
    if missing:
        parser.error("required input files are missing: " + ", ".join(missing))

    graph_steps = []
    if any((PROJECT_ROOT / "data/raw/chatgpt").glob("*.json")):
        graph_steps.append(
            ("ChatGPT normalization and branch graph", ["scripts/chatgpt/process_exports.py"])
        )
    if (
        (PROJECT_ROOT / "data/raw/twitter/twitter_no_metadata.json").exists()
        and (PROJECT_ROOT / "data/raw/twitter/twitter_metadata.json").exists()
    ):
        graph_steps.append(
            ("Twitter normalization and reply/quote graph", ["scripts/twitter/process_exports.py"])
        )

    core_steps = graph_steps + [
        ("Whisper extraction", ["scripts/extract_whisper.py"]),
        ("Canonical messages", ["scripts/create_canonical_messages.py"]),
        ("Structure-aware conversation chunks", ["scripts/create_candidate_windows.py"]),
        (
            "Core content/style retrieval collections",
            ["scripts/build_retrieval_documents.py"],
        ),
        (
            "Embeddings",
            ["scripts/embed_chunks.py", "--backend", args.embedding_backend],
        ),
    ]
    for name, command in core_steps:
        run_step(name, command)

    if args.with_reasoning:
        run_step("Optional empirical reasoning chains", ["scripts/build_reasoning_chains.py"])
        run_step(
            "Optional enriched retrieval views",
            [
                "scripts/build_chroma_documents.py",
                "--output",
                "data/chroma_docs/reasoning_documents.json",
            ],
        )

    run_step("Structural validation", ["scripts/validate_outputs.py"])
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
