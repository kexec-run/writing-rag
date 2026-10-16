from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_publication_privacy import (
    REVIEW_SCHEMA,
    apply_review_decisions,
    create_snapshot,
    load_review_manifest,
    snapshot_paths,
    validate_review_manifest,
    violations,
)


class PublicationPrivacyTests(unittest.TestCase):
    def test_synthetic_source_fixture_is_allowed(self) -> None:
        self.assertEqual(
            violations([("tests/fixtures/demo.json", '{"sender":"Demo Author"}', False)]),
            [],
        )

    def test_dummy_credential_is_rejected_with_specific_reason(self) -> None:
        issues = violations([("README.md", "token=hf_" + "A" * 24, False)])
        self.assertEqual(issues, ["credential_pattern:README.md"])

    def test_private_absolute_path_is_rejected_with_specific_reason(self) -> None:
        issues = violations(
            [("docs/example.md", "source: /Users/" + "demo/dev/private/data/messages.json", False)]
        )
        self.assertEqual(issues, ["private_absolute_path:docs/example.md"])

    def test_volume_private_absolute_path_is_rejected(self) -> None:
        private_path = "/Volumes/" + "groot/dev/private/data/messages.json"
        issues = violations(
            [("docs/example.md", private_path, False)]
        )
        self.assertEqual(issues, ["private_absolute_path:docs/example.md"])

    def test_aws_credentials_path_is_rejected(self) -> None:
        self.assertEqual(
            violations([(".aws/credentials", "", False)]),
            ["prohibited_path:.aws/credentials"],
        )

    def test_private_key_header_is_rejected(self) -> None:
        key_header = "-" * 5 + "BEGIN OPENSSH PRIVATE KEY" + "-" * 5
        self.assertEqual(
            violations([("docs/key.txt", key_header, False)]),
            ["credential_pattern:docs/key.txt"],
        )

    def test_aws_secret_assignment_is_rejected(self) -> None:
        setting_name = "aws_secret_" + "access_key"
        token = setting_name + "=" + "A" * 40
        self.assertEqual(
            violations([("docs/config.txt", token, False)]),
            ["credential_pattern:docs/config.txt"],
        )

    def test_prohibited_data_path_is_rejected(self) -> None:
        issues = violations([("data/raw/messages.json", "{}", False)])
        self.assertEqual(issues, ["prohibited_path:data/raw/messages.json"])

    def test_symlink_is_rejected_without_following_it(self) -> None:
        issues = violations([("tests/fixtures/external.json", "", True)])
        self.assertEqual(issues, ["symlink:tests/fixtures/external.json"])

    def test_snapshot_walk_errors_fail_closed(self) -> None:
        def walk_with_error(_root, *, followlinks, onerror):
            self.assertFalse(followlinks)
            onerror(PermissionError("inaccessible snapshot directory"))
            return iter(())

        with (
            patch(
                "scripts.check_publication_privacy.os.walk",
                side_effect=walk_with_error,
            ),
            self.assertRaisesRegex(
                PermissionError, "inaccessible snapshot directory"
            ),
        ):
            snapshot_paths(self.root)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "source"
        self.root.mkdir()
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)

    def make_manifest(self, files: list[tuple[str, bytes, str, str]]) -> dict:
        records = []
        for relative, content, decision, reason in files:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "tracked": False,
                    "decision": decision,
                    "reason": reason,
                }
            )
        return {"schema_version": REVIEW_SCHEMA, "files": records}

    def test_reviewed_file_set_passes_and_creates_snapshot(self) -> None:
        manifest = self.make_manifest(
            [("README.md", b"public synthetic content\n", "keep", "reviewed synthetic docs")]
        )
        issues, included = validate_review_manifest(self.root, manifest)
        self.assertEqual(issues, [])

        destination = self.root.parent / "snapshot"
        create_snapshot(self.root, destination, included)

        self.assertEqual((destination / "README.md").read_bytes(), b"public synthetic content\n")
        remotes = subprocess.run(
            ["git", "-C", str(destination), "remote"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(remotes, "")

    def test_changed_bytes_fail_hash_lock(self) -> None:
        manifest = self.make_manifest(
            [("README.md", b"before\n", "keep", "reviewed")]
        )
        (self.root / "README.md").write_bytes(b"after\n")

        issues, _ = validate_review_manifest(self.root, manifest)

        self.assertIn("hash_mismatch:README.md", issues)

    def test_review_decision_approves_current_bytes_after_unreviewed_edit(self) -> None:
        manifest = self.make_manifest(
            [("README.md", b"before\n", "needs_review", "pending")]
        )
        manifest_path = self.root.parent / "review.json"
        manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
        (self.root / "README.md").write_bytes(b"reviewed after edit\n")

        apply_review_decisions(
            self.root,
            manifest_path,
            [("README.md", "keep", "reviewed current bytes")],
        )

        updated = load_review_manifest(manifest_path)["files"][0]
        self.assertEqual(updated["sha256"], hashlib.sha256(b"reviewed after edit\n").hexdigest())
        self.assertEqual(updated["decision"], "keep")

    def test_unlisted_file_fails_closed(self) -> None:
        manifest = self.make_manifest(
            [("README.md", b"reviewed\n", "keep", "reviewed")]
        )
        (self.root / "extra.txt").write_text("unlisted", encoding="utf-8")

        issues, _ = validate_review_manifest(self.root, manifest)

        self.assertIn("unreviewed_path:extra.txt", issues)

    def test_unresolved_review_blocks_snapshot(self) -> None:
        manifest = self.make_manifest(
            [("README.md", b"unknown\n", "needs_review", "pending")]
        )

        issues, _ = validate_review_manifest(self.root, manifest)

        self.assertIn("unresolved_review:README.md", issues)

    def test_binary_file_requires_explicit_review_reason(self) -> None:
        manifest = self.make_manifest(
            [("figure.png", b"\x89PNG\r\n", "keep", "looks okay")]
        )

        issues, _ = validate_review_manifest(self.root, manifest)

        self.assertIn("binary_review_reason_required:figure.png", issues)

    def test_binary_private_path_is_rejected_even_with_review_reason(self) -> None:
        manifest = self.make_manifest(
            [("data/raw.bin", b"\x89PNG\r\n", "keep", "binary reviewed: synthetic")]
        )
        issues, _ = validate_review_manifest(self.root, manifest)
        self.assertIn("prohibited_path:data/raw.bin", issues)

    def test_ignored_file_in_snapshot_tree_is_rejected(self) -> None:
        manifest = self.make_manifest(
            [("README.md", b"public\n", "keep", "reviewed synthetic docs")]
        )
        (self.root / ".gitignore").write_text("data/\n", encoding="utf-8")
        (self.root / "data").mkdir()
        (self.root / "data/private.json").write_text("{}", encoding="utf-8")
        issues, _ = validate_review_manifest(
            self.root, manifest, exact_snapshot_tree=True
        )
        self.assertIn("unlisted_snapshot_path:data/private.json", issues)
        self.assertIn("prohibited_path:data/private.json", issues)

    def test_snapshot_tree_rejects_git_history_and_remotes(self) -> None:
        manifest = self.make_manifest(
            [("README.md", b"public\n", "keep", "reviewed synthetic docs")]
        )
        subprocess.run(
            ["git", "-C", str(self.root), "remote", "add", "origin", "https://example.invalid/repo.git"],
            check=True,
        )
        refs = self.root / ".git/refs/heads"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text("deadbeef\n", encoding="utf-8")

        issues, _ = validate_review_manifest(
            self.root, manifest, exact_snapshot_tree=True
        )

        self.assertIn("snapshot_git_remote_present", issues)
        self.assertIn("snapshot_git_metadata_not_clean:refs/heads/main", issues)

    def test_excluded_file_is_not_required_in_snapshot(self) -> None:
        manifest = self.make_manifest(
            [("TODO.md", b"private account detail\n", "exclude", "contains private identifier")]
        )
        (self.root / "TODO.md").unlink()

        issues, included = validate_review_manifest(self.root, manifest)

        self.assertEqual(issues, [])
        self.assertEqual(included, [])

    def test_symlink_is_rejected_by_release_manifest_validation(self) -> None:
        target = self.root.parent / "outside.txt"
        target.write_text("external", encoding="utf-8")
        (self.root / "linked.txt").symlink_to(target)
        manifest = {
            "schema_version": REVIEW_SCHEMA,
            "files": [
                {
                    "path": "linked.txt",
                    "sha256": None,
                    "tracked": False,
                    "decision": "exclude",
                    "reason": "symlink excluded",
                }
            ],
        }

        issues, _ = validate_review_manifest(self.root, manifest)

        self.assertIn("symlink:linked.txt", issues)

    def test_symlink_parent_is_rejected_by_release_manifest_validation(self) -> None:
        outside = self.root.parent / "outside-dir"
        outside.mkdir()
        (outside / "secret.txt").write_text("external", encoding="utf-8")
        (self.root / "linked-dir").symlink_to(outside, target_is_directory=True)
        manifest = {
            "schema_version": REVIEW_SCHEMA,
            "files": [
                {
                    "path": "linked-dir/secret.txt",
                    "sha256": None,
                    "tracked": False,
                    "decision": "keep",
                    "reason": "reviewed",
                }
            ],
        }

        issues, _ = validate_review_manifest(self.root, manifest)

        self.assertIn("symlink:linked-dir/secret.txt", issues)

    def test_malformed_manifest_root_is_reported_as_value_error(self) -> None:
        from scripts.check_publication_privacy import load_review_manifest

        manifest_path = self.root / "manifest.json"
        manifest_path.write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "invalid public-release review manifest"):
            load_review_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
