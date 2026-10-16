"""Build content-memory documents from heading-aware Markdown and MDX sources.

This adapter is for article/project documents. It intentionally does not use
conversation chunking and supports an explicit exclusion list for holdout
documents such as a reference post. MDX is a content view here; style examples
must be built separately by the long-form style adapter.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from writing_rag.runtime.pipeline_common import count_tokens, write_json


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    return text[end + 5 :] if end >= 0 else text


def clean_mdx_source(text: str) -> str:
    """Remove MDX wiring while retaining prose, headings, and code evidence."""
    lines = []
    in_code = False
    for line in strip_frontmatter(text).splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        if not in_code and re.match(r"^\s*(?:import|export)\b", line):
            continue
        if not in_code and re.match(r"^\s*</?[A-Z][^>]*>\s*$", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def source_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in {".md", ".mdx"}
        ),
        key=lambda path: path.name.casefold(),
    )


ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "observation": ("noticed", "appeared", "looked", "visible", "saw", "artifact"),
    "motivation": (
        "goal",
        "wanted",
        "mattered",
        "practical goal",
        "why this matters",
        "aim",
    ),
    "initial_hypothesis": ("hypothesis", "thought", "assume", "maybe", "could be"),
    "discarded_approach": (
        "discarded",
        "discard",
        "not enough",
        "does not solve",
        "goose chase",
        "second suspicion",
        "cannot be compared directly",
        "fair test is not",
    ),
    "reframe": ("reframe", "shifted from", "what changed", "fair test", "testable"),
    "experiment": ("experiment", "compare", "control", "input", "output"),
    "measurement": ("metric", "mean", "stddev", "correlation", "histogram", "variance"),
    "finding": ("result", "shows", "observed", "current result", "supports"),
    "mechanism": ("caused", "path", "resize", "re-encode", "pipeline", "because"),
    "decision": (
        "decision",
        "best practical",
        "practical way",
        "chose",
        "chosen",
        "choice",
        "reproduction",
    ),
    "conclusion": ("conclusion", "therefore", "in other words", "summary"),
}


def narrative_roles(text: str, heading: str | None) -> list[str]:
    haystack = f"{heading or ''}\n{text}".casefold()
    roles = [
        role
        for role, patterns in ROLE_PATTERNS.items()
        if any(pattern in haystack for pattern in patterns)
    ]
    return roles or ["evidence"]


def split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if any(item.strip() for item in body):
                sections.append((heading, body))
            heading = match.group(1)
            body = []
        else:
            body.append(line)
    if any(item.strip() for item in body):
        sections.append((heading, body))
    return [(heading, "\n".join(body).strip()) for heading, body in sections]


def chunk_section(text: str, max_tokens: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join(current + [paragraph])
        if current and count_tokens(candidate) > max_tokens:
            chunks.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_documents(
    input_dir: str | Path,
    *,
    exclude: set[str] | None = None,
    max_tokens: int = 700,
) -> list[dict[str, Any]]:
    root = Path(input_dir)
    excluded = exclude or set()
    documents: list[dict[str, Any]] = []
    number = 0
    for path in source_files(root):
        if path.name in excluded:
            continue
        source_text = clean_mdx_source(path.read_text(encoding="utf-8"))
        for heading, section in split_markdown_sections(source_text):
            for part_index, chunk in enumerate(
                chunk_section(section, max_tokens), start=1
            ):
                number += 1
                roles = narrative_roles(chunk, heading)
                documents.append(
                    {
                        "document_id": f"markdown_{number:06d}",
                        "view": "content",
                        "collection": "content_memory",
                        "text": chunk,
                        "metadata": {
                            "source": "markdown",
                            "source_format": path.suffix.casefold().lstrip("."),
                            "content_eligible": True,
                            "source_path": str(path),
                            "source_file": path.name,
                            "heading": heading,
                            "section_part": part_index,
                            "narrative_roles": ",".join(roles),
                            "narrative_sequence": number,
                            "source_scope": "content",
                            "speaker_filter": "document",
                            "style_owner": "none",
                            "reasoning_status": "not_generated",
                        },
                    }
                )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument(
        "--output", default="data/generated/markdown_content_documents.json"
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero")
    documents = build_documents(
        args.input_dir, exclude=set(args.exclude), max_tokens=args.max_tokens
    )
    write_json(args.output, documents)
    print(f"Markdown content documents: {len(documents)}")
    print(f"Excluded files: {', '.join(args.exclude) or 'none'}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
