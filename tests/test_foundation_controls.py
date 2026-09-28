"""Tests for baseline sealing and registered experiment infrastructure."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_baseline_v1 import freeze_baseline, verify_baseline
from validate_experiment_tracking import validate_registry


def make_writable_tree(root):
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def test_actual_baseline_is_sealed_and_hash_verified():
    baseline = ROOT / "runs" / "baseline_v1"
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    present = sum((baseline / item["path"]).is_file() for item in manifest["files"])
    if present == 0:
        pytest.skip("Local baseline payload is not materialized in this checkout")
    result = verify_baseline(baseline)
    assert result["status"] == "PASS"
    assert result["file_count"] == 59
    assert result["manifest_sha256"] == (
        "c8a36879aa6e717d60641c5930ebb083aa3b362d98d10f2601584633d4e88a80"
    )


def test_freeze_is_create_once_and_detects_mutation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pt").write_bytes(b"frozen model")
    extra = tmp_path / "config.json"
    extra.write_text('{"seed": 42}\n', encoding="utf-8")
    target = tmp_path / "baseline"
    try:
        freeze_baseline(
            target=target,
            source_groups={"model": source},
            extra_files={"environment/config.json": extra},
            git_head="a" * 40,
            created_at="2026-09-24T00:00:00+00:00",
        )
        assert verify_baseline(target)["file_count"] == 2
        with pytest.raises(FileExistsError, match="must never be overwritten"):
            freeze_baseline(
                target=target,
                source_groups={"model": source},
                extra_files={"environment/config.json": extra},
                git_head="a" * 40,
            )
        copied = target / "model" / "model.pt"
        os.chmod(copied, stat.S_IWRITE | stat.S_IREAD)
        copied.write_bytes(b"changed")
        with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
            verify_baseline(target)
    finally:
        if target.exists():
            make_writable_tree(target)


def test_registered_experiment_directories_are_complete_and_frozen():
    result = validate_registry(ROOT)
    registry = json.loads(
        (ROOT / "experiments" / "registry.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "PASS"
    assert result["experiment_count"] == len(registry["experiments"])
    assert {item["id"] for item in result["experiments"]} == {
        item["id"] for item in registry["experiments"]
    }
    assert all(
        item["status"] in {"REGISTERED_NOT_STARTED", "AUDIT_COMPLETE", "COMPLETE"}
        for item in result["experiments"]
    )
