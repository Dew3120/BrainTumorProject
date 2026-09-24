"""Verify the closed E05 weighted-training experiment and write its manifest."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e05_glioma_weighted"
ARTIFACTS = EXPERIMENT / "artifacts"
LABELS = ("glioma", "meningioma", "pituitary")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    result = json.loads((EXPERIMENT / "result.json").read_text(encoding="utf-8"))
    registry = json.loads(
        (ROOT / "experiments/registry.json").read_text(encoding="utf-8")
    )
    registered = next(
        item for item in registry["experiments"] if item["id"] == "e05_glioma_weighted"
    )
    if config["status"] != "COMPLETE" or registered["status"] != "COMPLETE":
        raise ValueError("E05 lifecycle status mismatch")
    if result["status"] != "FAIL" or result["gate_passed"]:
        raise ValueError("E05 expected closed failed decision")
    with (ARTIFACTS / "per_case_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E05 comparison population mismatch")
    for label in LABELS:
        members = [row for row in rows if row["category"] == label]
        baseline = float(np.mean([float(row["baseline_dice"]) for row in members]))
        candidate = float(np.mean([float(row["candidate_dice"]) for row in members]))
        if not math.isclose(baseline, result["baseline_mean_dice"][label], abs_tol=1e-12):
            raise ValueError(f"E05 baseline mismatch: {label}")
        if not math.isclose(candidate, result["candidate_mean_dice"][label], abs_tol=1e-12):
            raise ValueError(f"E05 candidate mismatch: {label}")
    if not result["required_followup_complete"]:
        raise ValueError("E05 required follow-up is incomplete")
    if result["test_images_accessed"] != 0:
        raise ValueError("E05 test access violation")
    required = (
        "hypothesis.md",
        "config.json",
        "result.json",
        "decision.md",
        "artifacts/selected_checkpoint.json",
        "artifacts/run_manifest.json",
        "artifacts/per_case_comparison.csv",
        "artifacts/category_dice_comparison.png",
        "artifacts/visual_review.json",
        "artifacts/worst_validation_by_category.png",
        "artifacts/best_validation_by_category.png",
        "artifacts/segmentation_failure_modes.csv",
        "artifacts/segmenter_pixel_calibration.json",
        "artifacts/segmenter_case_reliability.json",
        "artifacts/summary.md",
    )
    manifest = {
        "status": "PASS",
        "experiment_id": "e05_glioma_weighted",
        "registered_decision": "FAIL",
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "files": [
            {
                "path": f"experiments/e05_glioma_weighted/{name}",
                "sha256": sha256(EXPERIMENT / name),
                "bytes": (EXPERIMENT / name).stat().st_size,
            }
            for name in required
        ],
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    (ARTIFACTS / "experiment_manifest.json").write_bytes(payload.encode("utf-8"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
