from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_pipeline import missing_required_inputs


class RequiredPipelineInputTests(unittest.TestCase):
    def test_missing_inputs_are_reported_by_repo_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                missing_required_inputs(Path(temporary)),
                [
                    "data/raw/reddit/reddit_chat.json",
                    "data/raw/whisper/transcript.json",
                ],
            )

    def test_complete_required_input_set_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "data/raw/reddit/reddit_chat.json",
                "data/raw/whisper/transcript.json",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            self.assertEqual(missing_required_inputs(root), [])


if __name__ == "__main__":
    unittest.main()
