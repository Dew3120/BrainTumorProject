"""Register Amendment 20 and O01-O06 before any OOD result is computed."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.json"
AMENDMENT = ROOT / "audit/preregistration_amendment_20.md"

EXPERIMENTS = {
    "o01_ood_benchmark": {"gate": "DESCRIPTIVE", "purpose": "Frozen OOD benchmark construction"},
    "o02_feature_ood": {"gate": "AUROC >= 0.85", "purpose": "Image-statistics detector"},
    "o03_encoder_mahalanobis": {"gate": "AUROC >= 0.90", "purpose": "Frozen-segmenter bottleneck Mahalanobis detector"},
    "o04_model_confidence_ood": {"gate": "classifier AUROC >= 0.85 AND segmenter AUROC >= 0.85", "purpose": "Frozen-model confidence detectors"},
    "o05_ensemble_ood": {"gate": "selected held-out AUROC >= 0.92", "purpose": "Registered combined OOD detector"},
    "o06_ood_policy": {"gate": "DESCRIPTIVE", "purpose": "OOD operating policy and application contract"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8-sig"))
    if any(item["id"] in EXPERIMENTS for item in registry["experiments"]):
        raise RuntimeError("OOD experiment is already registered")
    amendment_hash = sha256(AMENDMENT)
    registry["amendments"].append({"path": "audit/preregistration_amendment_20.md", "sha256": amendment_hash})
    for experiment_id, spec in EXPERIMENTS.items():
        folder = ROOT / "experiments" / experiment_id
        artifacts = folder / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=False)
        (artifacts / ".gitkeep").write_text("\n", encoding="utf-8")
        hypothesis = (
            f"# {experiment_id}\n\n"
            f"**Status:** REGISTERED_NOT_STARTED\n\n"
            f"**Hypothesis:** {spec['purpose']}.\n\n"
            f"**Registered gate:** `{spec['gate']}`.\n\n"
            "This package is governed by `audit/preregistration_amendment_20.md`. "
            "No gate relaxation, experiment substitution, sealed-test access, neural retraining, or prior-threshold change is permitted.\n"
        )
        (folder / "hypothesis.md").write_text(hypothesis, encoding="utf-8", newline="\n")
        config = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": "REGISTERED_NOT_STARTED",
            "hypothesis_ids": ["H9"],
            "amendment": "audit/preregistration_amendment_20.md",
            "amendment_sha256": amendment_hash,
            "gate": spec["gate"],
            "bootstrap_resamples": 2000,
            "seed": 42,
            "segmentation_threshold": 0.5,
            "reliability_policy_threshold": "0.9490250173235835",
            "brisc_test_images_accessed": 0,
            "neural_retraining_or_refitting": False,
        }
        write_json(folder / "config.json", config)
        write_json(folder / "result.json", {"schema_version": 1, "experiment_id": experiment_id, "status": "NOT_RUN"})
        (folder / "decision.md").write_text(
            f"# Decision — {experiment_id}\n\n**Status:** PENDING EXECUTION\n",
            encoding="utf-8",
            newline="\n",
        )
        registry["experiments"].append({
            "id": experiment_id,
            "path": f"experiments/{experiment_id}",
            "hypothesis_ids": ["H9"],
            "status": "REGISTERED_NOT_STARTED",
        })
    write_json(REGISTRY, registry)
    print(f"Registered Amendment 20 ({amendment_hash}) and {len(EXPERIMENTS)} experiments")


if __name__ == "__main__":
    main()
