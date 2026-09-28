"""Fit, verify, and freeze the preregistered R02 operational estimator."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import sklearn
import stage4_reliability_common as common
from fit_segmenter_case_reliability import make_model

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x00_r02_deployable_model"
ARTIFACTS = EXPERIMENT / "artifacts"
MODELS = ROOT / "models"
SCALER_PATH = MODELS / "r02_reliability_scaler.pkl"
LOGISTIC_PATH = MODELS / "r02_reliability_logistic.pkl"
MODEL_MANIFEST_PATH = MODELS / "r02_reliability_manifest.json"
REFERENCE_PATH = ARTIFACTS / "operational_reference_predictions.csv"
TOLERANCE = 1e-12

SOURCE_HASHES = {
    "audit/preregistration_amendment_14.md": "8abe7d0fb56c116f2c3f3f667e0263564dcb89b5021b8a3dc61959487849b7d5",
    "experiments/r02_reduced_features/result.json": "fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2",
    "experiments/r02_reduced_features/artifacts/experiment_manifest.json": "1dc386a3cd342f14fcb827960c1ec89f749b2413962c768aec1c22bdf74cb819",
    "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv": "9b986435b0da0fdb340a16730209ebb895115ece694386c298830e3956b2fc6a",
    "experiments/r07_coefficient_bootstrap/result.json": "fc8b1bd80826913ff990ea993bf1c0c8bdf028b614101f967afc4282c01cb948",
    "experiments/r07_coefficient_bootstrap/artifacts/coefficient_summary.csv": "b1f72477a76afa2ed74cb01481d3e514d339d405f38cc2ced5576a22cbbe2b52",
    "experiments/r07_coefficient_bootstrap/artifacts/experiment_manifest.json": "afddd3023655ec9fc539611a50eef3335d36923894e49aa99e831f9743e5dd10",
    "experiments/c03_isotonic/artifacts/isotonic_mappings.json": "fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def max_abs(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
    return float(np.max(np.abs(left - right), initial=0.0))


def load_registered_data(config):
    r01_config = read_json(ROOT / "experiments/r01_ablation/config.json")
    r02_config = read_json(ROOT / "experiments/r02_reduced_features/config.json")
    loader_config = {
        **r02_config,
        "c03_result_sha256": r01_config["c03_result_sha256"],
        "expected_c03_hard_mask_changes": config["expected_c03_hard_mask_changes"],
    }
    data = common.load_calibrated_e08(loader_config)
    all_features = tuple(r01_config["features"])
    matrix, contract = common.feature_matrix(data["probability"], all_features)
    lookup = {name: index for index, name in enumerate(all_features)}
    selected = tuple(config["features"])
    features = matrix[:, [lookup[name] for name in selected]]
    dice = common.dice_scores(data["probability"], data["targets"], threshold=0.5)
    target = dice >= 0.5
    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    if features.shape != (680, 10) or not np.isfinite(features).all():
        raise ValueError("Operational feature matrix is invalid")
    if int(target.sum()) != 598 or int((~target).sum()) != 82:
        raise ValueError("Operational target counts differ from registration")
    if len(set(groups)) != 583:
        raise ValueError("Operational similarity-group count differs from registration")
    if float(config["segmentation_threshold"]) != 0.5:
        raise ValueError("Segmentation threshold changed")
    return data, features, target, contract


def registered_coefficients(feature_order):
    path = ROOT / "experiments/r07_coefficient_bootstrap/artifacts/coefficient_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["feature"]: row for row in csv.DictReader(handle)}
    return np.asarray(
        [float(rows[name]["full_population_standardized_coefficient"]) for name in feature_order],
        dtype=np.float64,
    )


def estimator_state(model):
    scaler = model.named_steps["scale"]
    logistic = model.named_steps["logistic"]
    return {
        "mean": scaler.mean_.copy(),
        "scale": scaler.scale_.copy(),
        "coefficients": logistic.coef_[0].copy(),
        "intercept": logistic.intercept_.copy(),
        "classes": logistic.classes_.copy(),
    }


def verify_sources():
    observed = {}
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        value = sha256(path)
        if value != expected:
            raise ValueError(f"Source hash mismatch: {relative}")
        observed[relative] = value
    return observed


def write_reference(rows, target, probabilities):
    with REFERENCE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "similarity_group", "fitting_target", "operational_probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row, outcome, probability in zip(rows, target, probabilities, strict=True):
            writer.writerow(
                {
                    "path": row["path"],
                    "similarity_group": row["similarity_group"],
                    "fitting_target": int(outcome),
                    "operational_probability": repr(float(probability)),
                }
            )


def main():
    config_path = EXPERIMENT / "config.json"
    config = read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X00 is not in its registered pre-run state")
    source_hashes = verify_sources()

    data_a, features_a, target_a, contract_a = load_registered_data(config)
    model_a = make_model()
    model_a.fit(features_a, target_a)
    state_a = estimator_state(model_a)
    probability_a = model_a.predict_proba(features_a)[:, 1]

    documented = registered_coefficients(config["features"])
    coefficient_error = max_abs(state_a["coefficients"], documented)
    if coefficient_error > TOLERANCE:
        raise ValueError(f"R07 coefficient reproduction failed: {coefficient_error}")

    _data_b, features_b, target_b, contract_b = load_registered_data(config)
    if not np.array_equal(features_a, features_b) or not np.array_equal(target_a, target_b):
        raise ValueError("Independent input reconstruction is not bitwise identical")
    model_b = make_model()
    model_b.fit(features_b, target_b)
    state_b = estimator_state(model_b)
    probability_b = model_b.predict_proba(features_b)[:, 1]
    independent_errors = {
        key: max_abs(state_a[key], state_b[key])
        for key in ("mean", "scale", "coefficients", "intercept", "classes")
    }
    independent_errors["probabilities"] = max_abs(probability_a, probability_b)
    if max(independent_errors.values()) > TOLERANCE:
        raise ValueError(f"Independent deterministic fit failed: {independent_errors}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    with SCALER_PATH.open("wb") as handle:
        pickle.dump(model_a.named_steps["scale"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with LOGISTIC_PATH.open("wb") as handle:
        pickle.dump(model_a.named_steps["logistic"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with SCALER_PATH.open("rb") as handle:
        scaler = pickle.load(handle)
    with LOGISTIC_PATH.open("rb") as handle:
        logistic = pickle.load(handle)
    reloaded_probability = logistic.predict_proba(scaler.transform(features_a))[:, 1]
    reload_error = max_abs(probability_a, reloaded_probability)
    if reload_error > TOLERANCE:
        raise ValueError(f"Serialization round trip failed: {reload_error}")

    write_reference(data_a["rows"], target_a, probability_a)
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }
    model_manifest = {
        "schema_version": 1,
        "status": "OPERATIONAL_ARTIFACT_FROZEN",
        "scope": "Operational full-population fit; not OOF evaluation",
        "feature_order": config["features"],
        "target": config["target"],
        "population": config["population"],
        "estimator": config["estimator"],
        "scaler_mean": state_a["mean"].tolist(),
        "scaler_scale": state_a["scale"].tolist(),
        "standardized_coefficients": state_a["coefficients"].tolist(),
        "intercept": float(state_a["intercept"][0]),
        "classes": state_a["classes"].tolist(),
        "runtime": runtime,
        "source_hashes": source_hashes,
        "artifacts": {
            "scaler": {"path": "models/r02_reliability_scaler.pkl", "sha256": sha256(SCALER_PATH)},
            "logistic": {"path": "models/r02_reliability_logistic.pkl", "sha256": sha256(LOGISTIC_PATH)},
            "reference_predictions": {
                "path": "experiments/x00_r02_deployable_model/artifacts/operational_reference_predictions.csv",
                "sha256": sha256(REFERENCE_PATH),
            },
        },
        "verification": {
            "absolute_tolerance": TOLERANCE,
            "r07_coefficient_max_abs_error": coefficient_error,
            "independent_fit_max_abs_errors": independent_errors,
            "serialization_reload_probability_max_abs_error": reload_error,
            "historical_full_population_probability_vector_existed": False,
            "feature_contract": contract_a,
            "second_feature_contract": contract_b,
            "c03_changed_pixels": data_a["changed_pixels"],
            "c03_changed_images": data_a["changed_images"],
            "external_images_accessed": 0,
            "brisc_test_images_accessed": 0,
            "neural_model_retrained": False,
            "segmentation_threshold_changed": False,
        },
    }
    write_json(MODEL_MANIFEST_PATH, model_manifest)

    result = {
        "schema_version": 1,
        "experiment_id": "x00_r02_deployable_model",
        "status": "PASS",
        "decision": "OPERATIONAL_ARTIFACT_FROZEN",
        "scope": "Operational fitting only; R02 OOF metrics remain reported evidence",
        "population": config["population"],
        "verification": model_manifest["verification"],
        "runtime": runtime,
        "model_artifacts": model_manifest["artifacts"],
        "model_manifest": {
            "path": "models/r02_reliability_manifest.json",
            "sha256": sha256(MODEL_MANIFEST_PATH),
        },
    }
    write_json(EXPERIMENT / "result.json", result)
    (EXPERIMENT / "decision.md").write_text(
        "# X00 decision\n\n"
        "**Decision:** PASS — OPERATIONAL ARTIFACT FROZEN\n\n"
        "The fixed R02 specification reproduced R07's full-population standardized "
        f"coefficients with maximum absolute error `{coefficient_error:.17g}`. Two "
        "independent fits and the serialized reload reproduced all operational "
        f"probabilities within `{TOLERANCE}` (observed reload error "
        f"`{reload_error:.17g}`). The artifact may be used for external inference. "
        "This is not an OOF evaluation; R02's OOF metrics remain the reported evidence.\n",
        encoding="utf-8",
        newline="\n",
    )
    config["status"] = "COMPLETE"
    config["verification_result"] = "PASS"
    write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    entry = next(item for item in registry["experiments"] if item["id"] == config["experiment_id"])
    if entry["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X00 registry lifecycle state changed")
    entry["status"] = "COMPLETE"
    write_json(registry_path, registry)

    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"\n## {timestamp} — Freeze the selected R02 reliability estimator\n\n"
            "- **Decision made:** Accept X00 as `PASS` and freeze the deterministic full-population R02 estimator for operational inference. Preserve R02 OOF results as the reported performance evidence.\n"
            f"- **Triggering artifact:** `models/r02_reliability_manifest.json`, SHA256 `{sha256(MODEL_MANIFEST_PATH)}`; X00 result SHA256 `{sha256(EXPERIMENT / 'result.json')}`.\n"
            f"- **Rationale:** R07 coefficients reproduced with maximum absolute error `{coefficient_error:.17g}`. Independent feature reconstruction and fitting reproduced probabilities with error `{independent_errors['probabilities']:.17g}`; serialization reload error was `{reload_error:.17g}`, all within `1e-12`.\n"
            "- **Verification:** Frozen source hashes, 680 cases, 598 successes, 82 failures, 583 groups, locked ten-feature order, feature contract, 22,891 C03 changed pixels across 679 images, and threshold `0.50` matched. External and sealed-test image access, neural retraining, and threshold changes were zero.\n"
            "- **Alternatives considered:** Fold-model averaging and coefficient-only reconstruction remained rejected. The exact fixed specification was fitted once for deployment under Amendment 14.\n"
            "- **Who approved:** User explicitly authorized Amendment 14 and the fit-and-freeze operation. Codex applied the predeclared deterministic checks without relaxation.\n"
        )

    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path.name not in {"experiment_manifest.json", ".gitkeep"}
    )
    included.extend(
        [
            SCALER_PATH,
            LOGISTIC_PATH,
            MODEL_MANIFEST_PATH,
            ROOT / "scripts/freeze_r02_operational_model.py",
            ROOT / "tests/test_r02_operational_model.py",
        ]
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(included)
        ],
    }
    write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
