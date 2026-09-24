"""Verify the closed E01 largest-component experiment."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e01_lcc_filter"
ARTIFACTS = EXPERIMENT / "artifacts"
LABELS = ("glioma", "meningioma", "pituitary")
FILES = (
    "hypothesis.md",
    "config.json",
    "result.json",
    "decision.md",
    "artifacts/per_case_metrics.csv",
    "artifacts/dice_delta_distribution.png",
    "artifacts/summary.md",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    result = json.loads((EXPERIMENT / "result.json").read_text(encoding="utf-8"))
    registry = json.loads(
        (ROOT / "experiments/registry.json").read_text(encoding="utf-8")
    )
    registry_item = next(
        item for item in registry["experiments"] if item["id"] == "e01_lcc_filter"
    )
    if config["status"] != "COMPLETE" or registry_item["status"] != "COMPLETE":
        raise ValueError("E01 lifecycle status is inconsistent")
    if result["status"] != "FAIL" or result["gate_passed"]:
        raise ValueError("E01 decision is not the registered failed outcome")
    with (ARTIFACTS / "per_case_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E01 per-case population is incomplete or duplicated")
    if any("/test/" in row["path"].replace("\\", "/") for row in rows):
        raise ValueError("Original test path entered E01")
    for label in LABELS:
        members = [row for row in rows if row["category"] == label]
        baseline = float(np.mean([float(row["baseline_dice"]) for row in members]))
        filtered = float(np.mean([float(row["filtered_dice"]) for row in members]))
        if not math.isclose(
            baseline, result["baseline_mean_dice"][label], abs_tol=1e-12
        ):
            raise ValueError(f"E01 baseline mean mismatch for {label}")
        if not math.isclose(
            filtered, result["filtered_mean_dice"][label], abs_tol=1e-12
        ):
            raise ValueError(f"E01 filtered mean mismatch for {label}")
    tolerance = result["mask_change_summary"]["comparison_tolerance"]
    expected_counts = {
        "changed_cases": sum(row["mask_changed"] == "True" for row in rows),
        "unchanged_cases": sum(row["mask_changed"] == "False" for row in rows),
        "improved_cases": sum(float(row["dice_delta"]) > tolerance for row in rows),
        "worsened_cases": sum(float(row["dice_delta"]) < -tolerance for row in rows),
        "equal_cases": sum(abs(float(row["dice_delta"])) <= tolerance for row in rows),
        "components_removed": sum(int(row["removed_components"]) for row in rows),
        "pixels_removed": sum(int(row["removed_pixels"]) for row in rows),
    }
    for key, expected in expected_counts.items():
        if result["mask_change_summary"][key] != expected:
            raise ValueError(f"E01 summary mismatch: {key}")
    if result["test_images_accessed"] != 0 or result["threshold_changed"]:
        raise ValueError("E01 violated test-access or threshold constraints")
    manifest = {
        "status": "PASS",
        "experiment_id": "e01_lcc_filter",
        "registered_decision": "FAIL",
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "files": [
            {
                "path": f"experiments/e01_lcc_filter/{name}",
                "sha256": sha256(EXPERIMENT / name),
                "bytes": (EXPERIMENT / name).stat().st_size,
            }
            for name in FILES
        ],
    }
    (ARTIFACTS / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
