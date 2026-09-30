import hashlib
import json
from pathlib import Path

import pytest

from gyrus.domain.errors import ManifestVerificationError
from gyrus.domain.manifest import sha256_file, verify_frozen_stack

ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(root: Path, *, artifact_path: str = "models/value.bin") -> str:
    artifact = root / artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"frozen")
    manifest = {
        "schema_version": 1,
        "status": "FROZEN",
        "package": "brain-tumor-research-final-stack",
        "source_commit": "a" * 40,
        "thresholds": {
            "segmentation": "0.50",
            "reliability": "0.9490250173235835",
            "ood_review": "0.36854275476100085",
            "ood_refuse": None,
        },
        "artifacts": [
            {
                "path": artifact_path,
                "sha256": hashlib.sha256(b"frozen").hexdigest(),
                "size_bytes": 6,
                "role": "fixture",
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def test_real_frozen_stack_verifies_without_writes():
    stack = verify_frozen_stack(ROOT / "runs/final_stack")
    assert stack.manifest_sha256 == (
        "7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a"
    )
    assert len(stack.artifacts) == 31
    assert stack.thresholds.segmentation == "0.50"


def test_unexpected_file_fails_strict_allowlist(tmp_path):
    digest = _write_manifest(tmp_path)
    (tmp_path / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="ALLOWLIST_MISMATCH"):
        verify_frozen_stack(tmp_path, expected_manifest_sha256=digest)


def test_threshold_drift_fails_before_artifact_use(tmp_path):
    _write_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["thresholds"]["segmentation"] = "0.5"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="THRESHOLD_DRIFT"):
        verify_frozen_stack(tmp_path, expected_manifest_sha256=sha256_file(path))
