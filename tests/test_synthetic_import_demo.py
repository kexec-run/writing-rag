from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_synthetic_demo import prepare_inputs
from scripts.run_synthetic_demo import _existing_demo_inputs


class SyntheticImportPreparationTests(unittest.TestCase):
    def test_copies_only_the_four_source_shaped_fixtures(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tests/fixtures").mkdir(parents=True)
            source_root = project_root / "tests/fixtures/synthetic_import"
            import shutil

            shutil.copytree(source_root, root / "tests/fixtures/synthetic_import")
            copied = prepare_inputs(root)
            self.assertEqual(len(copied), 4)
            self.assertTrue((root / "data/raw/instagram/message_1.json").is_file())
            self.assertTrue((root / "data/raw/whatsapp/demo_conversation.txt").is_file())
            self.assertTrue((root / "data/raw/reddit/reddit_chat.json").is_file())
            self.assertTrue((root / "data/raw/whisper/transcript.json").is_file())

    def test_refuses_to_overwrite_existing_raw_input(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tests/fixtures").mkdir(parents=True)
            import shutil

            shutil.copytree(
                project_root / "tests/fixtures/synthetic_import",
                root / "tests/fixtures/synthetic_import",
            )
            existing = root / "data/raw/instagram/message_1.json"
            existing.parent.mkdir(parents=True)
            existing.write_text("preserve me", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                prepare_inputs(root)
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve me")

    def test_demo_refuses_nonfixture_raw_input(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tests/fixtures").mkdir(parents=True)
            import shutil

            shutil.copytree(
                project_root / "tests/fixtures/synthetic_import",
                root / "tests/fixtures/synthetic_import",
            )
            prepare_inputs(root)
            extra = root / "data/raw/chatgpt/private-export.json"
            extra.parent.mkdir(parents=True)
            extra.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly the four synthetic fixtures"):
                _existing_demo_inputs(root)

    def test_demo_refuses_fixture_path_with_changed_bytes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tests/fixtures").mkdir(parents=True)
            import shutil

            shutil.copytree(
                project_root / "tests/fixtures/synthetic_import",
                root / "tests/fixtures/synthetic_import",
            )
            prepare_inputs(root)
            (root / "data/raw/instagram/message_1.json").write_text(
                "not the fixture", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "differs from its fixture"):
                _existing_demo_inputs(root)

    def test_demo_refuses_data_parent_symlink_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            external = Path(temporary) / "external"
            root.mkdir()
            external.mkdir()
            (root / "data").symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "through a data symlink"):
                _existing_demo_inputs(root)

            self.assertEqual(list(external.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
