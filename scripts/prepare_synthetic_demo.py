"""Copy invented source-shaped examples into ignored data/raw locations."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests/fixtures/synthetic_import"
INPUTS = {
    Path("instagram/message_1.json"): Path("instagram/message_1.json"),
    Path("whatsapp/demo_conversation.txt"): Path("whatsapp/demo_conversation.txt"),
    Path("reddit/reddit_chat.json"): Path("reddit/reddit_chat.json"),
    Path("whisper/transcript.json"): Path("whisper/transcript.json"),
}


def prepare_inputs(project_root: Path = PROJECT_ROOT) -> list[Path]:
    fixture_root = project_root / "tests/fixtures/synthetic_import"
    destinations = [project_root / "data/raw" / relative for relative in INPUTS.values()]
    existing = [path for path in destinations if path.exists() or path.is_symlink()]
    if existing:
        rendered = ", ".join(str(path.relative_to(project_root)) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing raw input(s): {rendered}")

    missing = [fixture_root / source for source in INPUTS]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"synthetic fixture is missing: {missing[0]}")

    copied: list[Path] = []
    for source, relative_destination in INPUTS.items():
        destination = project_root / "data/raw" / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_root / source, destination)
        copied.append(destination)
    return copied


def main() -> int:
    copied = prepare_inputs()
    print(f"Prepared {len(copied)} synthetic source files under data/raw/ (ignored local data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
