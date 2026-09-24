"""Verify the closed E04 two-scale inference experiment."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e04_multiscale"
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
    registered = next(
        item for item in registry["experiments"] if item["id"] == "e04_multiscale"
    )
    if config["status"] != "COMPLETE" or registered["status"] != "COMPLETE":
        raise ValueError("E04 lifecycle status mismatch")
    if result["status"] not in {"PASS", "FAIL"}:
        raise ValueError("E04 lacks a final decision")
    with (ARTIFACTS / "per_case_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E04 case population mismatch")
    if any("/test/" in row["path"].replace("\\", "/") for row in rows):
        raise ValueError("Original test path entered E04")
    for label in LABELS:
        members = [row for row in rows if row["category"] == label]
        baseline = float(np.mean([float(row["baseline_dice"]) for row in members]))
        candidate = float(
            np.mean([float(row["multiscale_dice"]) for row in members])
        )
        if not math.isclose(baseline, result["baseline_mean_dice"][label], abs_tol=1e-12):
            raise ValueError(f"E04 baseline mean mismatch: {label}")
        if not math.isclose(
            candidate, result["multiscale_mean_dice"][label], abs_tol=1e-12
        ):
            raise ValueError(f"E04 multiscale mean mismatch: {label}")
        delta = result["mean_dice_delta"][label]
        expected_check = (
            delta >= config["gate"][f"{label}_mean_dice_delta_minimum"]
        )
        if result["gate_checks"][label] != expected_check:
            raise ValueError(f"E04 gate mismatch: {label}")
    if result["gate_passed"] != all(result["gate_checks"].values()):
        raise ValueError("E04 conjunctive decision mismatch")
    if result["test_images_accessed"] != 0 or result["threshold_changed"]:
        raise ValueError("E04 violated access or threshold constraints")
    manifest = {
        "status": "PASS",
        "experiment_id": "e04_multiscale",
        "registered_decision": result["status"],
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "files": [
            {
                "path": f"experiments/e04_multiscale/{name}",
                "sha256": sha256(EXPERIMENT / name),
                "bytes": (EXPERIMENT / name).stat().st_size,
            }
            for name in FILES
        ],
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    (ARTIFACTS / "experiment_manifest.json").write_bytes(payload.encode("utf-8"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
