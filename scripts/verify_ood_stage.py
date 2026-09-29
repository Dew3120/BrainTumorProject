"""Independently verify completion and immutable constraints for O01-O06."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = ["o01_ood_benchmark", "o02_feature_ood", "o03_encoder_mahalanobis", "o04_model_confidence_ood", "o05_ensemble_ood", "o06_ood_policy"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registry = json.loads((ROOT / "experiments/registry.json").read_text())
    statuses = {row["id"]: row["status"] for row in registry["experiments"]}
    if any(statuses.get(experiment_id) != "COMPLETE" for experiment_id in IDS):
        raise RuntimeError("Registry does not mark every OOD experiment COMPLETE")
    for experiment_id in IDS:
        folder = ROOT / "experiments" / experiment_id
        result = json.loads((folder / "result.json").read_text())
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"{experiment_id} result is incomplete")
        manifest = json.loads((folder / "artifacts/experiment_manifest.json").read_text())
        for item in manifest["files"]:
            path = ROOT / item["path"]
            if not path.exists() or sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"{experiment_id} artifact mismatch: {item['path']}")
        verification = result.get("verification")
        if verification:
            required = {
                "brisc_test_images_accessed": 0,
                "neural_retraining_or_refitting": 0,
                "segmentation_threshold": 0.5,
                "reliability_policy_threshold": "0.9490250173235835",
            }
            for key, expected in required.items():
                if verification.get(key) != expected:
                    raise RuntimeError(f"{experiment_id} constraint mismatch: {key}")
    o05 = json.loads((ROOT / "experiments/o05_ensemble_ood/result.json").read_text())
    if not o05["gate"]["passed"] or o05["selected_metrics"]["auroc"] < 0.92:
        raise RuntimeError("H9 ensemble condition did not pass")
    o06 = json.loads((ROOT / "experiments/o06_ood_policy/result.json").read_text())
    if o06["thresholds"]["review"] != 0.36854275476100085:
        raise RuntimeError("OOD REVIEW threshold changed")
    print("OOD stage verification PASS: 6/6 complete; H9 ensemble gate PASS; constraints unchanged")


if __name__ == "__main__":
    main()
