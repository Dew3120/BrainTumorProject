"""Build the immutable Stage 8 package and fail closed on any mismatch."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "runs/final_stack"
STAGING = ROOT / "runs/.final_stack_staging"
TOLERANCE = 1e-12

SOURCES = {
    "models/classifier_v1.pt": (
        "runs/classifier/20260921-170607-243133-baseline/best_model.pt",
        "3bc044027b83b2de225ebde3fb817be8827d2b427c599824697b13ff7a727535",
        "Selected v1 four-class classifier; raw softmax probabilities",
    ),
    "models/segmenter_v1_fallback.pt": (
        "runs/segmentation/20260923-072439-397263-full/top3_checkpoints/epoch_056_macro_dice_0.762662.pt",
        "16dc8745993eef959d0f2924c9147a45320673f6fc0b465ab08f8ad7b2f723de",
        "E08 non-glioma route and O03 bottleneck source",
    ),
    "models/segmenter_e05_weighted.pt": (
        "runs/segmentation_v2_weighted/20260924-055049-892543-weighted/top3_checkpoints/epoch_045_macro_dice_0.738782.pt",
        "e17e7f5afec1c56e9faf12e12db263eff4a3ebc184cfcd7bdfcfff0c1a977c6a",
        "E08 ensemble component; not selected standalone",
    ),
    "models/segmenter_e06_boundary.pt": (
        "runs/segmentation_v2_boundary_loss/20260924-063051-647842-boundary/top3_checkpoints/epoch_040_macro_dice_0.745706.pt",
        "b98afabc8118471e01f560ff6a727ce25c08001574bf12005d77d1be773f43ab",
        "E08 ensemble component; not selected standalone",
    ),
    "models/segmenter_e07_glioma_specialist.pt": (
        "runs/segmentation_v2_glioma_specialized/20260924-105229-241365-specialist/top3_checkpoints/epoch_031_glioma_dice_0.580685.pt",
        "bb68ce4f0a9203d74a5ec9dea2e83468d3d833ac285a13ebbdb1d793a0d3bbe0",
        "E08 glioma route component; not selected standalone",
    ),
    "calibration/c03_isotonic_mappings.json": (
        "experiments/c03_isotonic/artifacts/isotonic_mappings.json",
        "fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa",
        "Selected C03 operational all-validation isotonic mapping",
    ),
    "reliability/r02_reliability_scaler.pkl": (
        "models/r02_reliability_scaler.pkl",
        "c99a67f8157515d815a9b7a05555b45e26b5a2da0e458b359cc83d63fe30dce8",
        "Selected R02 deployment scaler",
    ),
    "reliability/r02_reliability_logistic.pkl": (
        "models/r02_reliability_logistic.pkl",
        "9f8ea8530c1c2ee6a0773c6ec7c0b92e6da7923ddf3c50d3d32176379339f044",
        "Selected R02 deployment logistic model",
    ),
    "reliability/r02_reliability_manifest.json": (
        "models/r02_reliability_manifest.json",
        "0d56aab5d562a34325003cee116c4f9d5f8fbe3fcc5278717338706d58e5303f",
        "R02 feature order, scaler state, coefficients, and provenance",
    ),
}

COPIES = {
    "configs/classifier_v1_config.json": "runs/classifier/20260921-170607-243133-baseline/config.json",
    "configs/segmenter_v1_config.json": "runs/segmentation/20260923-072439-397263-full/config.json",
    "configs/segmenter_e05_config.json": "runs/segmentation_v2_weighted/20260924-055049-892543-weighted/config.json",
    "configs/segmenter_e06_config.json": "runs/segmentation_v2_boundary_loss/20260924-063051-647842-boundary/config.json",
    "configs/segmenter_e07_config.json": "runs/segmentation_v2_glioma_specialized/20260924-105229-241365-specialist/config.json",
    "configs/e08_selection.json": "experiments/e08_final_selection/result.json",
    "configs/c03_result.json": "experiments/c03_isotonic/result.json",
    "configs/o05_result.json": "experiments/o05_ensemble_ood/result.json",
    "ood/o06_policy.json": "experiments/o06_ood_policy/result.json",
    "ood/message_catalogue.json": "experiments/o06_ood_policy/artifacts/message_catalogue.json",
    "runtime/final_stack_runtime.py": "scripts/final_stack_runtime.py",
    "environment/pyproject.toml": "pyproject.toml",
    "environment/uv.lock": "uv.lock",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def copy(relative: str, source: str) -> None:
    destination = STAGING / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / source, destination)


def verify_selections() -> dict:
    e08 = read_json(ROOT / "experiments/e08_final_selection/result.json")
    c03 = read_json(ROOT / "experiments/c03_isotonic/result.json")
    o05 = read_json(ROOT / "experiments/o05_ensemble_ood/result.json")
    o06 = read_json(ROOT / "experiments/o06_ood_policy/result.json")
    classifier = read_json(ROOT / "audit/confidence_calibration_findings.json")
    checks = {
        "h7_calibration_passed": e08["calibration_condition"]["gate_status"] == "PASS",
        "h7_reliability_passed": e08["h7_reliability_condition"]["passed"] is True,
        "e08_selected": e08["decision"] == "NOMINATE_C03_CALIBRATED_E08_WITH_R02_RELIABILITY",
        "c03_selected": c03["status"] == "PASS" and c03["source_model"] == "e08_ensemble",
        "classifier_raw_selected": classifier["component_decisions"]["classifier"]["accepted_probability_source"] == "raw classifier probabilities",
        "h9_passed": o05["gate"]["passed"] is True and o05["selection"]["selected"] == "secondary_logistic",
        "ood_policy_accepted": o06["decision"] == "POLICY_ACCEPTED",
        "refuse_disabled": o06["thresholds"]["refuse"] is None and o06["thresholds"]["refuse_enabled"] is False,
        "segmentation_threshold": o06["verification"]["segmentation_threshold"] == 0.5,
        "reliability_threshold": o06["verification"]["reliability_policy_threshold"] == "0.9490250173235835",
        "ood_threshold": o06["thresholds"]["review"] == 0.36854275476100085,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Selection verification failed: {checks}")
    return checks


def recover_o03() -> dict:
    embeddings_path = ROOT / "experiments/o03_encoder_mahalanobis/artifacts/frozen_embeddings.npz"
    score_path = ROOT / "experiments/o03_encoder_mahalanobis/artifacts/mahalanobis_scores.csv"
    benchmark_path = ROOT / "experiments/o01_ood_benchmark/artifacts/benchmark_manifest.csv"
    archive = np.load(embeddings_path)
    sample_ids = archive["sample_id"].astype(str)
    embeddings = archive["embedding"].astype(np.float64)
    manifest = {row["sample_id"]: row for row in read_csv(benchmark_path)}
    if embeddings.shape != (2680, 128) or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("O03 embedding artifact contract changed")
    if set(sample_ids) != set(manifest):
        raise RuntimeError("O03 embedding/benchmark membership mismatch")
    fit_id = np.asarray(
        [manifest[sid]["detector_split"] == "fit" and manifest[sid]["domain"] == "ID" for sid in sample_ids]
    )
    if int(fit_id.sum()) != 600:
        raise RuntimeError("O03 fitting membership changed")
    gaussian = LedoitWolf(assume_centered=False).fit(embeddings[fit_id])
    observed = gaussian.mahalanobis(embeddings)
    recorded = {row["sample_id"]: float(row["mahalanobis_squared"]) for row in read_csv(score_path)}
    expected = np.asarray([recorded[sid] for sid in sample_ids], dtype=np.float64)
    maximum = float(np.max(np.abs(observed - expected)))
    if maximum > TOLERANCE:
        raise RuntimeError(f"O03 reconstruction exceeded 1e-12: {maximum}")
    output = STAGING / "ood/o03_mahalanobis_state.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        location=gaussian.location_.astype(np.float64),
        covariance=gaussian.covariance_.astype(np.float64),
        precision=gaussian.precision_.astype(np.float64),
    )
    result = {
        "status": "PASS",
        "method": "deterministic LedoitWolf recovery from registered frozen embeddings",
        "fit_id_cases": 600,
        "embedding_dimensions": 128,
        "reproduction_tolerance": TOLERANCE,
        "maximum_absolute_distance_error": maximum,
        "source_embeddings_sha256": sha256(embeddings_path),
        "source_scores_sha256": sha256(score_path),
    }
    write_json(STAGING / "ood/o03_recovery.json", result)
    return result


def build_o02_state() -> dict:
    result = read_json(ROOT / "experiments/o02_feature_ood/result.json")
    model = result["model"]
    coefficients = [item["standardized_coefficient"] for item in model["coefficients"]]
    state = {
        "feature_order": model["features"],
        "scaler_mean": model["scaler_mean"],
        "scaler_scale": model["scaler_scale"],
        "coefficients": coefficients,
        "intercept": model["intercept"],
    }
    rows = read_csv(ROOT / "experiments/o02_feature_ood/artifacts/image_features_and_scores.csv")
    matrix = np.asarray([[float(row[name]) for name in state["feature_order"]] for row in rows])
    logits = ((matrix - np.asarray(state["scaler_mean"])) / np.asarray(state["scaler_scale"])) @ np.asarray(coefficients) + state["intercept"]
    reconstructed = expit(logits)
    expected = np.asarray([float(row["ood_probability"]) for row in rows])
    maximum = float(np.max(np.abs(reconstructed - expected)))
    if maximum > TOLERANCE:
        raise RuntimeError(f"O02 state reproduction exceeded 1e-12: {maximum}")
    state["reproduction_maximum_absolute_error"] = maximum
    write_json(STAGING / "ood/o02_state.json", state)
    return state


def keyed(path: Path, fields: tuple[str, ...]) -> dict[str, list[float]]:
    return {
        row["sample_id"]: [float(row[field]) for field in fields]
        for row in read_csv(path)
    }


def build_o05_state() -> dict:
    benchmark = read_csv(ROOT / "experiments/o01_ood_benchmark/artifacts/benchmark_manifest.csv")
    o02 = keyed(ROOT / "experiments/o02_feature_ood/artifacts/image_features_and_scores.csv", ("ood_probability",))
    o03 = keyed(ROOT / "experiments/o03_encoder_mahalanobis/artifacts/mahalanobis_scores.csv", ("mahalanobis_squared",))
    o04 = keyed(ROOT / "experiments/o04_model_confidence_ood/artifacts/model_confidence_scores.csv", ("classifier_ood_score", "segmenter_entropy_ood_score"))
    sample_ids = [row["sample_id"] for row in benchmark]
    if set(sample_ids) != set(o02) or set(sample_ids) != set(o03) or set(sample_ids) != set(o04):
        raise RuntimeError("O05 source memberships differ")
    raw = np.asarray([o02[sid] + o03[sid] + o04[sid] for sid in sample_ids], dtype=np.float64)
    fit_id = np.asarray([row["detector_split"] == "fit" and row["domain"] == "ID" for row in benchmark])
    if int(fit_id.sum()) != 600:
        raise RuntimeError("O05 CDF reference membership changed")
    names = ("o02_probability", "o03_mahalanobis", "o04_classifier", "o04_segmenter_entropy")
    references = {name: np.sort(raw[fit_id, index]) for index, name in enumerate(names)}
    np.savez_compressed(STAGING / "ood/o05_normalization_reference.npz", **references)
    normalized = np.column_stack(
        [np.searchsorted(references[name], raw[:, index], side="right") / 600 for index, name in enumerate(names)]
    )
    result = read_json(ROOT / "experiments/o05_ensemble_ood/result.json")
    selected = result["secondary_logistic"]
    coefficients = [item["secondary_standardized_coefficient"] for item in selected["standardized_coefficients"]]
    state = {
        "selected_method": result["selection"]["selected"],
        "score_order": list(names),
        "cdf_reference": "600 BRISC fit-ID cases, right-inclusive empirical CDF",
        "scaler_mean": selected["scaler_mean"],
        "scaler_scale": selected["scaler_scale"],
        "coefficients": coefficients,
        "intercept": selected["intercept"],
    }
    reconstructed = expit(
        ((normalized - np.asarray(state["scaler_mean"])) / np.asarray(state["scaler_scale"])) @ np.asarray(coefficients)
        + state["intercept"]
    )
    scores = {row["sample_id"]: float(row["selected_score"]) for row in read_csv(ROOT / "experiments/o05_ensemble_ood/artifacts/ensemble_scores.csv")}
    expected = np.asarray([scores[sid] for sid in sample_ids])
    maximum = float(np.max(np.abs(reconstructed - expected)))
    if maximum > TOLERANCE:
        raise RuntimeError(f"O05 state reproduction exceeded 1e-12: {maximum}")
    state["reproduction_maximum_absolute_error"] = maximum
    write_json(STAGING / "ood/o05_state.json", state)
    return state


def make_fixtures() -> None:
    folder = STAGING / "verification/fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:192, 0:256]
    gradient = np.rint(255 * (0.6 * x / 255 + 0.4 * y / 191)).astype(np.uint8)
    ellipse = np.zeros((221, 173), dtype=np.uint8)
    yy, xx = np.mgrid[:221, :173]
    ellipse[((yy - 108) / 82) ** 2 + ((xx - 85) / 61) ** 2 <= 1] = 92
    ellipse[((yy - 95) / 23) ** 2 + ((xx - 63) / 17) ** 2 <= 1] = 210
    rng = np.random.default_rng(20260930)
    noise = rng.integers(0, 256, size=(160, 207), dtype=np.uint8)
    for name, value in (("gradient.png", gradient), ("elliptic.png", ellipse), ("noise.png", noise)):
        Image.fromarray(value, mode="L").save(folder / name, format="PNG", optimize=False, compress_level=6)


def artifact_entries(provenance: dict[str, tuple[str | None, str]]) -> list[dict]:
    entries = []
    for path in sorted(item for item in STAGING.rglob("*") if item.is_file() and item.name != "manifest.json"):
        relative = path.relative_to(STAGING).as_posix()
        source, role = provenance.get(relative, (None, "Generated final-stack verification artifact"))
        entry = {
            "path": relative,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "role": role,
        }
        if source:
            entry["source_path"] = source
            entry["source_sha256"] = sha256(ROOT / source)
        entries.append(entry)
    return entries


def load_runtime():
    path = STAGING / "runtime/final_stack_runtime.py"
    spec = importlib.util.spec_from_file_location("frozen_final_stack_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def main() -> None:
    if DESTINATION.exists() or STAGING.exists():
        raise RuntimeError("Final destination or staging directory already exists")
    checks = verify_selections()
    provenance: dict[str, tuple[str | None, str]] = {}
    try:
        STAGING.mkdir(parents=True)
        for relative, (source, expected, role) in SOURCES.items():
            if sha256(ROOT / source) != expected:
                raise RuntimeError(f"Source hash mismatch: {source}")
            copy(relative, source)
            if sha256(STAGING / relative) != expected:
                raise RuntimeError(f"Copied hash mismatch: {relative}")
            provenance[relative] = (source, role)
        for relative, source in COPIES.items():
            copy(relative, source)
            provenance[relative] = (source, "Configuration, runtime, or policy provenance")

        o03 = recover_o03()
        o02 = build_o02_state()
        o05 = build_o05_state()
        make_fixtures()
        policy = read_json(STAGING / "ood/o06_policy.json")
        r02_manifest = read_json(STAGING / "reliability/r02_reliability_manifest.json")
        manifest = {
            "schema_version": 1,
            "status": "FROZEN",
            "package": "brain-tumor-research-final-stack",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "selection": {
                "classifier": "v1 raw probabilities",
                "segmenter": "E08 arithmetic mean: E05 + E06 + classifier-routed E07/v1",
                "calibrator": "C03 final-all-validation isotonic mapping",
                "reliability": "R02 10-feature L2 logistic regression",
                "ood": "O05 secondary logistic ensemble",
                "policy": "O06 ACCEPT/REVIEW; REFUSE disabled",
            },
            "thresholds": {
                "segmentation": "0.50",
                "reliability": "0.9490250173235835",
                "ood_review": "0.36854275476100085",
                "ood_refuse": None,
            },
            "classifier": {"classes": ["glioma", "meningioma", "pituitary", "no_tumor"], "probability_source": "raw softmax"},
            "segmenter": {"size": 128, "routing": {"glioma": "e07", "all_other_classes": "v1"}, "ensemble_components": ["e05", "e06", "routed_e07_or_v1"]},
            "calibration": {"method": "isotonic", "knots": 501, "operational_mapping": "final_all_validation_mapping"},
            "reliability": {"feature_order": r02_manifest["feature_order"], "scaler_mean": r02_manifest["scaler_mean"], "scaler_scale": r02_manifest["scaler_scale"], "coefficients": r02_manifest["standardized_coefficients"], "intercept": r02_manifest["intercept"]},
            "ood": {"selected_method": o05["selected_method"], "score_order": o05["score_order"], "o03_bottleneck_module": "model.1.submodule.1.submodule.1.submodule", "o03_reproduction_max_abs_error": o03["maximum_absolute_distance_error"], "o02_reproduction_max_abs_error": o02["reproduction_maximum_absolute_error"], "o05_reproduction_max_abs_error": o05["reproduction_maximum_absolute_error"], "refuse_enabled": policy["thresholds"]["refuse_enabled"]},
            "verification": {"selection_checks": checks, "reconstruction_absolute_tolerance": TOLERANCE, "brisc_test_images_accessed": 0, "neural_retraining": 0, "threshold_changes": 0},
            "artifacts": artifact_entries(provenance),
        }
        write_json(STAGING / "manifest.json", manifest)

        runtime_module = load_runtime()
        runtime = runtime_module.FinalStackRuntime(STAGING)
        expected = {}
        for fixture in sorted((STAGING / "verification/fixtures").glob("*.png")):
            expected[fixture.name] = runtime.infer_path(fixture)
        runtime.close()
        write_json(STAGING / "verification/expected_outputs.json", expected)
        manifest["artifacts"] = artifact_entries(provenance)
        write_json(STAGING / "manifest.json", manifest)

        first = runtime_module.FinalStackRuntime(STAGING)
        second = runtime_module.FinalStackRuntime(STAGING)
        for fixture in sorted((STAGING / "verification/fixtures").glob("*.png")):
            actual = first.infer_path(fixture)
            repeated = second.infer_path(fixture)
            runtime_module.compare_outputs(actual, expected[fixture.name], tolerance=TOLERANCE)
            runtime_module.compare_outputs(repeated, expected[fixture.name], tolerance=TOLERANCE)
        first.close()
        second.close()
        STAGING.rename(DESTINATION)
    except Exception:
        if STAGING.exists():
            shutil.rmtree(STAGING)
        raise
    print(json.dumps({"status": "PASS", "destination": str(DESTINATION), "o03_reproduction": o03["maximum_absolute_distance_error"], "artifact_count": len(manifest["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()
