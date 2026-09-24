"""Verify the closed E06 boundary-aware training experiment."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e06_boundary_loss"
ARTIFACTS = EXPERIMENT / "artifacts"


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
        item for item in registry["experiments"] if item["id"] == "e06_boundary_loss"
    )
    if config["status"] != "COMPLETE" or registered["status"] != "COMPLETE":
        raise ValueError("E06 lifecycle status mismatch")
    if result["status"] != "FAIL" or result["gate_passed"]:
        raise ValueError("E06 expected closed failed decision")
    with (ARTIFACTS / "per_case_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E06 case population mismatch")
    glioma = [row for row in rows if row["category"] == "glioma"]
    baseline_boundary = float(
        np.mean([float(row["baseline_boundary_f1"]) for row in glioma])
    )
    candidate_boundary = float(
        np.mean([float(row["candidate_boundary_f1"]) for row in glioma])
    )
    if not math.isclose(
        baseline_boundary, result["boundary_f1"]["baseline"]["glioma"], abs_tol=1e-12
    ):
        raise ValueError("E06 baseline boundary F1 mismatch")
    if not math.isclose(
        candidate_boundary,
        result["boundary_f1"]["candidate"]["glioma"],
        abs_tol=1e-12,
    ):
        raise ValueError("E06 candidate boundary F1 mismatch")
    if not result["required_followup_complete"] or result["test_images_accessed"]:
        raise ValueError("E06 follow-up or test-access mismatch")
    required = (
        "hypothesis.md",
        "config.json",
        "result.json",
        "decision.md",
        "artifacts/selected_checkpoint.json",
        "artifacts/run_manifest.json",
        "artifacts/per_case_comparison.csv",
        "artifacts/dice_and_boundary_f1_comparison.png",
        "artifacts/visual_review.json",
        "artifacts/worst_validation_by_category.png",
        "artifacts/best_validation_by_category.png",
        "artifacts/segmentation_failure_modes.csv",
        "artifacts/segmenter_pixel_calibration.json",
        "artifacts/segmenter_case_reliability.json",
        "artifacts/summary.md",
    )
    missing = [name for name in required if not (EXPERIMENT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing E06 artifacts: {missing}")
    included_files = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file()
        and path != ARTIFACTS / "experiment_manifest.json"
        and path.name != ".gitkeep"
    )
    manifest = {
        "status": "PASS",
        "experiment_id": "e06_boundary_loss",
        "registered_decision": "FAIL",
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in included_files
        ],
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    (ARTIFACTS / "experiment_manifest.json").write_bytes(payload.encode("utf-8"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
