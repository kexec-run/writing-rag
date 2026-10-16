"""Score obsidian-kb-poc-v1 vault arms: graph shape + related sample."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
VAULT_ROOT = PROJECT_ROOT / "data/generated/obsidian-kb-poc-v1/vaults"
SLICE = EXPERIMENT_DIR / "slice.json"

WIKILINK = re.compile(r"\[\[([^\]|#]+)")
ARMS = ("a-flat", "b-spine", "c-virtual", "d-baked")


def read_notes(root: Path) -> dict[str, str]:
    notes: dict[str, str] = {}
    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root))
        if rel.startswith("_meta/"):
            continue
        notes[rel] = path.read_text(encoding="utf-8")
    return notes


def edges(notes: dict[str, str]) -> list[tuple[str, str]]:
    names = {rel[: -len(".md")] for rel in notes}
    out: list[tuple[str, str]] = []
    for rel, text in notes.items():
        source = rel[: -len(".md")]
        for target in WIKILINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if target not in names:
                # hub links use folder paths; normalise basename match
                base = target.rsplit("/", 1)[-1]
                match = next((n for n in names if n == base or n.endswith("/" + base)), None)
                if match is None:
                    continue
                target = match
            out.append((source, target))
    return out


def components(nodes: set[str], edge_list: list[tuple[str, str]]) -> list[int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for a, b in edge_list:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[str] = set()
    sizes = []
    for node in nodes:
        if node in seen:
            continue
        stack, size = [node], 0
        seen.add(node)
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def score_arm(arm: str) -> dict:
    root = VAULT_ROOT / arm
    notes = read_notes(root)
    edge_list = edges(notes)
    degrees: dict[str, int] = defaultdict(int)
    for a, b in edge_list:
        degrees[a] += 1
        degrees[b] += 1
    values = sorted(degrees.values())
    comps = components({rel[: -len(".md")] for rel in notes}, edge_list)
    hub_notes = [rel for rel in notes if rel.startswith("hubs/")]
    word_count = sum(len(text.split()) for text in notes.values())
    return {
        "arm": arm,
        "file_count": len(notes),
        "total_bytes": sum(len(t.encode()) for t in notes.values()),
        "word_count": word_count,
        "edge_count": len(edge_list),
        "degree_mean": round(sum(values) / len(values), 3) if values else 0.0,
        "degree_max": max(values) if values else 0,
        "degree_p50": values[len(values) // 2] if values else 0,
        "component_count": len(comps),
        "largest_component_share": round(
            comps[0] / len(notes), 3
        )
        if comps
        else 0.0,
        "orphan_notes": sum(1 for rel in notes if degrees.get(rel[: -len(".md")], 0) == 0),
        "hub_note_count": len(hub_notes),
        "hub_mediated_edges": sum(1 for a, b in edge_list if a.startswith("hubs/") or b.startswith("hubs/")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/generated/obsidian-kb-poc-v1/mechanical-metrics.json"))
    args = parser.parse_args()
    slice_payload = json.loads(SLICE.read_text(encoding="utf-8"))
    results = {"schema_version": "obsidian-kb-poc-mechanical-v1", "arms": [score_arm(a) for a in ARMS]}
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["arms"], indent=2))
    print(f"windows in slice: {slice_payload['window_count']}")


if __name__ == "__main__":
    main()
