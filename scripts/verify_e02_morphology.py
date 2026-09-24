"""Verify the closed E02 morphology experiment."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e02_morphology"
ARTIFACTS = EXPERIMENT / "artifacts"
LABELS = ("glioma", "meningioma", "pituitary")
PREFIXES = ("kernel_3", "kernel_5")
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
    registered = next(
        item for item in registry["experiments"] if item["id"] == "e02_morphology"
    )
    if config["status"] != "COMPLETE" or registered["status"] != "COMPLETE":
        raise ValueError("E02 lifecycle status mismatch")
    if result["status"] != "FAIL" or result["gate_passed"]:
        raise ValueError("E02 decision mismatch")
    with (ARTIFACTS / "per_case_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E02 case population mismatch")
    if any("/test/" in row["path"].replace("\\", "/") for row in rows):
        raise ValueError("Original test path entered E02")
    for prefix in PREFIXES:
        for label in LABELS:
            members = [row for row in rows if row["category"] == label]
            mean = float(np.mean([float(row[f"{prefix}_dice"]) for row in members]))
            expected = result["variants"][prefix]["mean_dice"][label]
            if not math.isclose(mean, expected, abs_tol=1e-12):
                raise ValueError(f"E02 mean mismatch: {prefix}/{label}")
        if result["variants"][prefix]["gate_passed"]:
            raise ValueError(f"Unexpected E02 passing variant: {prefix}")
    if result["selected_variant"] is not None:
        raise ValueError("E02 selected a failed variant")
    if result["test_images_accessed"] != 0 or result["threshold_changed"]:
        raise ValueError("E02 violated access or threshold constraints")
    manifest = {
        "status": "PASS",
        "experiment_id": "e02_morphology",
        "registered_decision": "FAIL",
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "files": [
            {
                "path": f"experiments/e02_morphology/{name}",
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
