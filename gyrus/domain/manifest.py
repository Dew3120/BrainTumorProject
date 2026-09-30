"""Strict, read-only verification of the frozen Stage 8 inference package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .constants import (
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
)
from .errors import ManifestVerificationError

EXPECTED_MANIFEST_SHA256 = (
    "7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a"
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA256 without loading the whole artifact into memory."""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenThresholds:
    """Canonical threshold text read from the verified manifest."""

    segmentation: str
    reliability: str
    ood_review: str
    ood_refuse: None


@dataclass(frozen=True, slots=True)
class FrozenArtifact:
    """One allowlisted file in the frozen package."""

    relative_path: str
    sha256: str
    size_bytes: int
    role: str


@dataclass(frozen=True, slots=True)
class VerifiedFrozenStack:
    """Capability token issued only after complete verification."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    package: str
    source_commit: str
    thresholds: FrozenThresholds
    artifacts: tuple[FrozenArtifact, ...]
    raw_manifest: Mapping[str, Any]


def _stop(code: str, message: str) -> ManifestVerificationError:
    return ManifestVerificationError(code, f"INVALID/STOP: {message}")


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _stop("MANIFEST_PATH_INVALID", "artifact path is missing or not text")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise _stop("MANIFEST_PATH_INVALID", f"unsafe artifact path: {value!r}")
    normalized = candidate.as_posix()
    if normalized != value.replace("\\", "/"):
        raise _stop("MANIFEST_PATH_INVALID", f"non-canonical artifact path: {value!r}")
    return normalized


def _validate_thresholds(value: object) -> FrozenThresholds:
    if not isinstance(value, dict):
        raise _stop("MANIFEST_THRESHOLDS_INVALID", "threshold object is missing")
    expected = {
        "segmentation": SEGMENTATION_THRESHOLD_TEXT,
        "reliability": RELIABILITY_THRESHOLD_TEXT,
        "ood_review": OOD_REVIEW_THRESHOLD_TEXT,
        "ood_refuse": None,
    }
    if value != expected:
        raise _stop(
            "MANIFEST_THRESHOLD_DRIFT",
            f"registered thresholds differ: expected {expected!r}, got {value!r}",
        )
    return FrozenThresholds(**expected)


def verify_frozen_stack(
    root: str | Path,
    *,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> VerifiedFrozenStack:
    """Verify the complete strict allowlist and return a read-only capability token."""

    package_root = Path(root).expanduser()
    try:
        package_root = package_root.resolve(strict=True)
    except OSError as exc:
        raise _stop("FROZEN_STACK_MISSING", f"cannot resolve frozen stack: {exc}") from exc
    if not package_root.is_dir():
        raise _stop("FROZEN_STACK_NOT_DIRECTORY", str(package_root))

    manifest_path = package_root / "manifest.json"
    if manifest_path.is_symlink():
        raise _stop("MANIFEST_SYMLINK_REJECTED", str(manifest_path))
    if not manifest_path.is_file():
        raise _stop("MANIFEST_MISSING", str(manifest_path))
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise _stop(
            "MANIFEST_HASH_MISMATCH",
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}",
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _stop("MANIFEST_JSON_INVALID", str(exc)) from exc
    if not isinstance(payload, dict):
        raise _stop("MANIFEST_JSON_INVALID", "top level must be an object")
    if payload.get("schema_version") != 1 or payload.get("status") != "FROZEN":
        raise _stop("MANIFEST_STATE_INVALID", "schema_version=1 and status=FROZEN required")
    if payload.get("package") != "brain-tumor-research-final-stack":
        raise _stop("MANIFEST_PACKAGE_INVALID", repr(payload.get("package")))
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise _stop("MANIFEST_SOURCE_COMMIT_INVALID", repr(source_commit))
    thresholds = _validate_thresholds(payload.get("thresholds"))

    artifact_rows = payload.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise _stop("MANIFEST_ARTIFACTS_INVALID", "nonempty artifact list required")
    seen: set[str] = set()
    artifacts: list[FrozenArtifact] = []
    for row in artifact_rows:
        if not isinstance(row, dict):
            raise _stop("MANIFEST_ARTIFACT_INVALID", "artifact entry must be an object")
        relative = _safe_relative_path(row.get("path"))
        if relative in seen:
            raise _stop("MANIFEST_DUPLICATE_ARTIFACT", relative)
        seen.add(relative)
        expected_hash = row.get("sha256")
        size_bytes = row.get("size_bytes")
        role = row.get("role")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise _stop("MANIFEST_ARTIFACT_HASH_INVALID", relative)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise _stop("MANIFEST_ARTIFACT_SIZE_INVALID", relative)
        if not isinstance(role, str) or not role.strip():
            raise _stop("MANIFEST_ARTIFACT_ROLE_INVALID", relative)

        candidate = package_root.joinpath(*PurePosixPath(relative).parts)
        if candidate.is_symlink():
            raise _stop("MANIFEST_ARTIFACT_SYMLINK_REJECTED", relative)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise _stop("MANIFEST_ARTIFACT_MISSING", f"{relative}: {exc}") from exc
        if not resolved.is_relative_to(package_root) or not resolved.is_file():
            raise _stop("MANIFEST_ARTIFACT_PATH_ESCAPE", relative)
        if resolved.stat().st_size != size_bytes:
            raise _stop("MANIFEST_ARTIFACT_SIZE_MISMATCH", relative)
        actual_hash = sha256_file(resolved)
        if actual_hash != expected_hash:
            raise _stop(
                "MANIFEST_ARTIFACT_HASH_MISMATCH",
                f"{relative}: expected {expected_hash}, got {actual_hash}",
            )
        artifacts.append(FrozenArtifact(relative, expected_hash, size_bytes, role))

    actual_files = {
        item.relative_to(package_root).as_posix()
        for item in package_root.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if actual_files != seen:
        missing = sorted(seen - actual_files)
        unexpected = sorted(actual_files - seen)
        raise _stop(
            "MANIFEST_ALLOWLIST_MISMATCH",
            f"missing={missing!r}; unexpected={unexpected!r}",
        )

    return VerifiedFrozenStack(
        root=package_root,
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha256,
        package=payload["package"],
        source_commit=source_commit,
        thresholds=thresholds,
        artifacts=tuple(artifacts),
        raw_manifest=MappingProxyType(payload),
    )
