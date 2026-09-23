"""Create the immutable v1 baseline archive, or verify an existing archive."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "runs" / "baseline_v1"
SOURCE_GROUPS = {
    "classifier": ROOT / "runs" / "classifier" / "20260921-170607-243133-baseline",
    "segmenter": ROOT / "runs" / "segmentation" / "20260923-072439-397263-full",
}
EXTRA_FILES = {
    "environment/pyproject.toml": ROOT / "pyproject.toml",
    "environment/uv.lock": ROOT / "uv.lock",
    "environment/python-version.txt": ROOT / ".python-version",
    "evidence/classifier_training_findings.json": ROOT
    / "audit"
    / "classifier_training_findings.json",
    "evidence/segmentation_training_findings.json": ROOT
    / "audit"
    / "segmentation_training_findings.json",
    "evidence/confidence_calibration_findings.json": ROOT
    / "audit"
    / "confidence_calibration_findings.json",
    "evidence/confidence_calibration_completion.md": ROOT
    / "audit"
    / "confidence_calibration_completion.md",
}
MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def set_read_only(path):
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def is_read_only(path):
    return not bool(Path(path).stat().st_mode & stat.S_IWRITE)


def source_inventory(source_groups=SOURCE_GROUPS, extra_files=EXTRA_FILES):
    inventory = []
    for group, source_root in source_groups.items():
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing baseline source directory: {source_root}")
        for source in sorted(
            (path for path in source_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(source_root).as_posix().casefold(),
        ):
            relative = Path(group) / source.relative_to(source_root)
            inventory.append((source, relative.as_posix(), group))
    for relative, source in sorted(extra_files.items()):
        if not source.is_file():
            raise FileNotFoundError(f"Missing baseline source file: {source}")
        inventory.append((source, relative, "project_context"))
    destinations = [item[1].casefold() for item in inventory]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate baseline destination path")
    return inventory


def current_git_head(root=ROOT):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def freeze_baseline(
    target=TARGET,
    source_groups=SOURCE_GROUPS,
    extra_files=EXTRA_FILES,
    git_head=None,
    created_at=None,
):
    target = Path(target)
    if target.exists():
        raise FileExistsError(
            f"Baseline target already exists and must never be overwritten: {target}"
        )
    inventory = source_inventory(source_groups, extra_files)
    target.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for source, relative, group in inventory:
            destination = target / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_hash = sha256(source)
            copy_hash = sha256(destination)
            if (
                source_hash != copy_hash
                or source.stat().st_size != destination.stat().st_size
            ):
                raise ValueError(f"Copy verification failed: {relative}")
            records.append(
                {
                    "path": Path(relative).as_posix(),
                    "source_path": str(source.resolve()),
                    "group": group,
                    "size_bytes": destination.stat().st_size,
                    "sha256": copy_hash,
                }
            )

        manifest = {
            "schema_version": 1,
            "baseline_id": "baseline_v1",
            "status": "SEALED_READ_ONLY",
            "created_at_utc": created_at or datetime.now(UTC).isoformat(),
            "source_git_head": git_head or current_git_head(),
            "purpose": (
                "Immutable comparison baseline containing the accepted classifier, "
                "segmenter, run evidence, and exact project environment snapshots."
            ),
            "source_runs": {
                group: str(path.resolve()) for group, path in source_groups.items()
            },
            "immutability_contract": {
                "overwrite_allowed": False,
                "files_marked_read_only": True,
                "verification_command": "uv run python scripts/verify_baseline_v1.py",
                "manifest_self_hash": MANIFEST_HASH_NAME,
            },
            "file_count": len(records),
            "total_size_bytes": sum(record["size_bytes"] for record in records),
            "files": records,
        }
        manifest_path = target / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest_hash = sha256(manifest_path)
        hash_path = target / MANIFEST_HASH_NAME
        hash_path.write_text(f"{manifest_hash}  {MANIFEST_NAME}\n", encoding="ascii")

        for path in (item for item in target.rglob("*") if item.is_file()):
            set_read_only(path)
        verify_baseline(target)
        return manifest, manifest_hash
    except Exception:
        for path in (item for item in target.rglob("*") if item.is_file()):
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(target, ignore_errors=True)
        raise


def verify_baseline(target=TARGET):
    target = Path(target)
    manifest_path = target / MANIFEST_NAME
    hash_path = target / MANIFEST_HASH_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = hash_path.read_text(encoding="ascii").split()[0]
    actual_manifest_hash = sha256(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError("Baseline manifest SHA256 mismatch")
    if manifest["status"] != "SEALED_READ_ONLY":
        raise ValueError("Baseline manifest is not sealed")

    expected = {record["path"]: record for record in manifest["files"]}
    actual = {
        path.relative_to(target).as_posix(): path
        for path in target.rglob("*")
        if path.is_file() and path.name not in {MANIFEST_NAME, MANIFEST_HASH_NAME}
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"Baseline membership mismatch; missing={missing}, extra={extra}"
        )
    for relative, path in actual.items():
        record = expected[relative]
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Baseline size mismatch: {relative}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Baseline SHA256 mismatch: {relative}")
    all_files = list(actual.values()) + [manifest_path, hash_path]
    writable = [str(path) for path in all_files if not is_read_only(path)]
    if writable:
        raise ValueError(f"Baseline contains writable files: {writable}")
    return {
        "status": "PASS",
        "file_count": len(actual),
        "total_size_bytes": sum(path.stat().st_size for path in actual.values()),
        "manifest_sha256": actual_manifest_hash,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the existing archive without writing anything.",
    )
    args = parser.parse_args()
    if args.verify_only:
        result = verify_baseline(TARGET)
    else:
        manifest, manifest_hash = freeze_baseline(TARGET)
        result = {
            "status": manifest["status"],
            "file_count": manifest["file_count"],
            "total_size_bytes": manifest["total_size_bytes"],
            "manifest_sha256": manifest_hash,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
