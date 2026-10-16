"""Reject obvious private paths, credentials, and symlinks in a release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SECRET_PATTERNS = (
    re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\baws_(?:secret_access_key|session_token)\s*=\s*[A-Za-z0-9/+=]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
)
PRIVATE_PATH_PATTERN = re.compile(
    r"(?:/Users/[^/\s]+/(?:dev|PrivateBackups|\.codex|\.agents)|"
    r"/Volumes/[^/\s]+/(?:dev|PrivateBackups|\.codex|\.agents))/"
)
PRIVATE_PATH_PARTS = {
    "data",
    ".cache",
    ".codegraph",
    ".codebase-memory",
    ".serena",
    ".aws",
    ".ssh",
}
PRIVATE_FILE_NAMES = {"credentials", "id_rsa", "id_ed25519"}
ALLOWED_ENV_EXAMPLE = Path(".env.example")
REVIEW_SCHEMA = "writing-rag-public-release-review-v1"
REVIEW_DECISIONS = {"keep", "sanitize", "exclude", "needs_review"}


def violations(entries: Iterable[tuple[str, str, bool]]) -> list[str]:
    """Return deterministic, reason-specific failures for path/content entries."""
    issues: list[str] = []
    for relative_name, content, is_symlink in sorted(entries):
        path = Path(relative_name.replace("\\", "/"))
        if is_symlink:
            issues.append(f"symlink:{relative_name}")
            continue
        if path.is_absolute() or ".." in path.parts:
            issues.append(f"unsafe_path:{relative_name}")
        if any(part in PRIVATE_PATH_PARTS for part in path.parts):
            issues.append(f"prohibited_path:{relative_name}")
        if path.name.casefold() in PRIVATE_FILE_NAMES or path.suffix.casefold() in {
            ".pem", ".p12", ".pfx", ".key"
        }:
            issues.append(f"prohibited_path:{relative_name}")
        if path.parts and path.parts[0] == "models" and "weights" in path.parts:
            issues.append(f"prohibited_path:{relative_name}")
        if path.name.startswith(".env") and path != ALLOWED_ENV_EXAMPLE:
            issues.append(f"prohibited_path:{relative_name}")
        if path.suffix.casefold() in {".db", ".sqlite", ".sqlite3"}:
            issues.append(f"prohibited_path:{relative_name}")
        if PRIVATE_PATH_PATTERN.search(content):
            issues.append(f"private_absolute_path:{relative_name}")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            issues.append(f"credential_pattern:{relative_name}")
    return sorted(set(issues))


def candidate_paths(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    names = completed.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return [
        root / name
        for name in names
        if name and ((root / name).exists() or (root / name).is_symlink())
    ]


def snapshot_paths(root: Path) -> list[Path]:
    """List every physical file/symlink except the snapshot's root Git metadata."""
    found: list[Path] = []
    root_git = root / ".git"
    if root_git.is_symlink():
        found.append(root_git)
    def fail_on_walk_error(error: OSError) -> None:
        raise error

    for directory, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=fail_on_walk_error
    ):
        current = Path(directory)
        if current == root and ".git" in dirnames:
            dirnames.remove(".git")
        for name in list(dirnames):
            path = current / name
            if path.is_symlink():
                found.append(path)
                dirnames.remove(name)
        found.extend(current / name for name in filenames)
    return sorted(set(found))


def clean_snapshot_git_issues(root: Path) -> list[str]:
    """Require the snapshot's Git metadata to be fresh, empty, and remote-free."""
    git_dir = root / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        return ["snapshot_git_metadata_missing_or_symlinked"]

    allowed_files = {"HEAD", "config", "description", "info/exclude"}
    issues: list[str] = []
    for path in git_dir.rglob("*"):
        if path.is_symlink():
            issues.append(f"snapshot_git_symlink:{path.relative_to(root).as_posix()}")
        elif path.is_file():
            relative = path.relative_to(git_dir).as_posix()
            if relative.startswith("hooks/") and relative.endswith(".sample"):
                continue
            if relative not in allowed_files:
                issues.append(f"snapshot_git_metadata_not_clean:{relative}")

    remotes = subprocess.run(
        ["git", "-C", str(root), "remote"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remotes:
        issues.append("snapshot_git_remote_present")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode == 0:
        issues.append("snapshot_git_history_present")
    return issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_symlink_component(root: Path, path: Path) -> bool:
    """Return True if the path or any component below root is a symlink."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def initialize_review_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite review manifest: {manifest_path}")
    tracked_result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked = set(tracked_result.stdout.decode("utf-8", errors="surrogateescape").split("\0"))
    files: list[dict[str, Any]] = []
    for path in candidate_paths(root):
        relative = path.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "path": relative,
            "sha256": None,
            "tracked": relative in tracked,
            "decision": "needs_review",
            "reason": "pending content review",
        }
        if has_symlink_component(root, path):
            record["reason"] = "symlink must be excluded from the release snapshot"
        else:
            try:
                record["sha256"] = sha256_file(path)
            except OSError as error:
                record["reason"] = f"unreadable during inventory: {type(error).__name__}"
        files.append(record)
    return {
        "schema_version": REVIEW_SCHEMA,
        "files": sorted(files, key=lambda item: item["path"]),
    }


def load_review_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REVIEW_SCHEMA
        or not isinstance(payload.get("files"), list)
    ):
        raise ValueError("invalid public-release review manifest schema")
    return payload


def apply_review_decisions(
    root: Path, manifest_path: Path, decisions: list[tuple[str, str, str]]
) -> None:
    payload = load_review_manifest(manifest_path)
    records = {record["path"]: record for record in payload["files"]}
    for relative, decision, reason in decisions:
        if relative not in records:
            raise ValueError(f"decision references an unknown file: {relative}")
        if decision not in REVIEW_DECISIONS:
            raise ValueError(f"invalid review decision for {relative}: {decision}")
        if not reason.strip():
            raise ValueError(f"review decision needs a reason: {relative}")
        path = root / relative
        if has_symlink_component(root, path) or not path.is_file():
            raise ValueError(f"reviewed path is no longer a regular file: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != records[relative].get("sha256"):
            if records[relative].get("decision") != "needs_review":
                raise ValueError(f"approved file changed after review: {relative}")
            # An unresolved file may be edited before its first review. The
            # explicit decision then approves the current bytes and hash.
            records[relative]["sha256"] = actual_hash
        records[relative]["decision"] = decision
        records[relative]["reason"] = reason.strip()

    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def validate_review_manifest(
    root: Path, payload: dict[str, Any], *, exact_snapshot_tree: bool = False
) -> tuple[list[str], list[Path]]:
    issues: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    for record in payload["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            issues.append("invalid_manifest_record")
            continue
        relative = record["path"]
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or ".git" in candidate.parts:
            issues.append(f"unsafe_manifest_path:{relative}")
            continue
        if relative in records:
            issues.append(f"duplicate_manifest_path:{relative}")
            continue
        records[relative] = record

    current = {path.relative_to(root).as_posix(): path for path in candidate_paths(root)}
    for relative in sorted(current.keys() - records.keys()):
        issues.append(f"unreviewed_path:{relative}")
    for relative in sorted(records.keys() - current.keys()):
        path = root / relative
        if has_symlink_component(root, path):
            issues.append(f"symlink:{relative}")
        elif records[relative].get("decision") != "exclude":
            issues.append(f"missing_path:{relative}")

    included: list[Path] = []
    for relative in sorted(current.keys() & records.keys()):
        path = current[relative]
        record = records[relative]
        decision = record.get("decision")
        if decision not in REVIEW_DECISIONS:
            issues.append(f"invalid_decision:{relative}")
            continue
        if decision == "needs_review":
            issues.append(f"unresolved_review:{relative}")
        if not str(record.get("reason", "")).strip():
            issues.append(f"missing_review_reason:{relative}")
        if has_symlink_component(root, path):
            issues.append(f"symlink:{relative}")
            continue
        if not path.is_file():
            issues.append(f"not_regular_file:{relative}")
            continue
        issues.extend(violations([(relative, "", False)]))
        try:
            actual_hash = sha256_file(path)
        except OSError as error:
            issues.append(f"unreadable:{relative}:{type(error).__name__}")
            continue
        if record.get("sha256") != actual_hash:
            issues.append(f"hash_mismatch:{relative}")
        if decision in {"keep", "sanitize"}:
            included.append(path)
            try:
                content = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                # Binary files are hash-locked and require a specific manual review reason.
                if "binary reviewed" not in str(record.get("reason", "")).casefold():
                    issues.append(f"binary_review_reason_required:{relative}")
                continue
            except OSError as error:
                issues.append(f"unreadable:{relative}:{type(error).__name__}")
                continue
            issues.extend(violations([(relative, content, False)]))
    if exact_snapshot_tree:
        expected = {
            relative
            for relative, record in records.items()
            if record.get("decision") in {"keep", "sanitize"}
        }
        physical = {
            path.relative_to(root).as_posix(): path for path in snapshot_paths(root)
        }
        for relative in sorted(physical.keys() - expected):
            path = physical[relative]
            issues.append(f"unlisted_snapshot_path:{relative}")
            if has_symlink_component(root, path):
                issues.append(f"symlink:{relative}")
            issues.extend(violations([(relative, "", path.is_symlink())]))
        for relative in sorted(expected - physical.keys()):
            issues.append(f"missing_snapshot_path:{relative}")
        issues.extend(clean_snapshot_git_issues(root))
    return sorted(set(issues)), included


def create_snapshot(root: Path, destination: Path, included: list[Path]) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        for source in included:
            relative = source.relative_to(root)
            if has_symlink_component(root, source):
                raise OSError(f"snapshot source contains a symlink component: {relative.as_posix()}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
            if sha256_file(source) != sha256_file(target):
                raise OSError(f"snapshot copy hash mismatch: {relative.as_posix()}")
        subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--init-review-manifest", type=Path)
    parser.add_argument(
        "--decision",
        nargs=3,
        action="append",
        metavar=("PATH", "DECISION", "REASON"),
        help="record one reviewed file decision in --review-manifest",
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--snapshot-tree",
        action="store_true",
        help="also reject ignored, untracked, or otherwise unlisted physical files in a snapshot",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.init_review_manifest:
        payload = initialize_review_manifest(root, args.init_review_manifest)
        args.init_review_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.init_review_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Initialized review inventory for {len(payload['files'])} candidate files.")
        return 0
    if args.decision:
        if not args.review_manifest:
            parser.error("--decision requires --review-manifest")
        try:
            apply_review_decisions(root, args.review_manifest, [tuple(item) for item in args.decision])
        except (OSError, ValueError) as error:
            print(f"review_decision_failed:{error}", file=sys.stderr)
            return 1
        print(f"Recorded {len(args.decision)} reviewed file decision(s).")
        return 0
    if args.review_manifest:
        try:
            payload = load_review_manifest(args.review_manifest)
            issues, included = validate_review_manifest(
                root, payload, exact_snapshot_tree=args.snapshot_tree
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"invalid_review_manifest:{error}", file=sys.stderr)
            return 1
        if issues:
            for issue in issues:
                print(issue, file=sys.stderr)
            print(f"Release validation failed: {len(issues)} issue(s).", file=sys.stderr)
            return 1
        if args.snapshot:
            try:
                create_snapshot(root, args.snapshot.resolve(), included)
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"snapshot_creation_failed:{error}", file=sys.stderr)
                return 1
        print(f"Release review passed for {len(included)} included files.")
        if args.snapshot:
            print(f"Snapshot created at {args.snapshot.resolve()}")
        return 0

    entries: list[tuple[str, str, bool]] = []
    scan_issues: list[str] = []
    for path in candidate_paths(root):
        relative = path.relative_to(root).as_posix()
        if has_symlink_component(root, path):
            entries.append((relative, "", True))
            continue
        if not path.is_file():
            continue
        try:
            content = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            scan_issues.append(f"binary_requires_review:{relative}")
            continue
        except OSError as error:
            scan_issues.append(f"unreadable:{relative}:{type(error).__name__}")
            continue
        entries.append((relative, content, False))

    issues = sorted({*violations(entries), *scan_issues})
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        print(f"Publication privacy check failed: {len(issues)} issue(s).", file=sys.stderr)
        return 1
    print(f"Publication privacy check passed for {len(entries)} candidate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
