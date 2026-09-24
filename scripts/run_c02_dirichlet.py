"""Run and close C02 binary Dirichlet calibration on frozen E08 outputs."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c02_dirichlet"
ARTIFACTS = EXPERIMENT / "artifacts"


def fit_binary_dirichlet(probabilities, targets):
    """Fit the identifiable binary Dirichlet map by pixel NLL."""
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    target = np.asarray(targets, dtype=np.float64)
    log_foreground = np.log(probability).reshape(-1)
    log_background = np.log1p(-probability).reshape(-1)
    y = target.reshape(-1)

    def objective(parameters):
        transformed = (
            parameters[0] * log_foreground
            + parameters[1] * log_background
            + parameters[2]
        )
        error = expit(transformed) - y
        loss = float(np.mean(np.logaddexp(0.0, transformed) - y * transformed))
        gradient = np.array(
            [
                np.mean(error * log_foreground),
                np.mean(error * log_background),
                np.mean(error),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        objective,
        np.array([1.0, -1.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-20.0, 20.0)] * 3,
        options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Binary Dirichlet fitting failed: {result.message}")
    return {
        "foreground_log_coefficient": float(result.x[0]),
        "background_log_coefficient": float(result.x[1]),
        "intercept": float(result.x[2]),
    }


def apply_binary_dirichlet(probabilities, parameters):
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    transformed = (
        parameters["foreground_log_coefficient"] * np.log(probability)
        + parameters["background_log_coefficient"] * np.log1p(-probability)
        + parameters["intercept"]
    )
    return transformed, expit(transformed)


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C02 is not in its registered pre-run state")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    oof_logits = np.empty_like(raw_logits)
    oof_probability = np.empty_like(raw_probability)
    fold_parameters = []
    for fold in range(5):
        fit = assignment != fold
        evaluate = assignment == fold
        parameters = fit_binary_dirichlet(raw_probability[fit], targets[fit])
        transformed, probability = apply_binary_dirichlet(
            raw_probability[evaluate], parameters
        )
        oof_logits[evaluate] = transformed
        oof_probability[evaluate] = probability
        fold_parameters.append(
            {
                "fold": fold + 1,
                **parameters,
                "fit_images": int(fit.sum()),
                "evaluation_images": int(evaluate.sum()),
            }
        )
    final_parameters = fit_binary_dirichlet(raw_probability, targets)
    regions = common.lesion_regions(targets)
    all_indices = np.arange(680)
    aggregate = {}
    bootstrap = {}
    groups = [row["similarity_group"] for row in rows]
    for region in common.REGIONS:
        raw, raw_nll, raw_brier = common.region_metrics(
            raw_probability, raw_logits, targets, regions, all_indices, region
        )
        calibrated, calibrated_nll, calibrated_brier = common.region_metrics(
            oof_probability, oof_logits, targets, regions, all_indices, region
        )
        aggregate[region] = {"raw": raw, "oof_calibrated": calibrated}
        bootstrap[region] = {
            "nll_change": common.paired_group_bootstrap(
                raw_nll, calibrated_nll, groups
            ),
            "brier_change": common.paired_group_bootstrap(
                raw_brier, calibrated_brier, groups
            ),
        }
    per_category = {}
    label_names = np.asarray([row["label"] for row in rows])
    for label in common.LABELS:
        indices = np.flatnonzero(label_names == label)
        per_category[label] = {}
        for region in common.REGIONS:
            raw, _, _ = common.region_metrics(
                raw_probability, raw_logits, targets, regions, indices, region
            )
            calibrated, _, _ = common.region_metrics(
                oof_probability, oof_logits, targets, regions, indices, region
            )
            per_category[label][region] = {
                "raw": {
                    key: raw[key]
                    for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")
                },
                "oof_calibrated": {
                    key: calibrated[key]
                    for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")
                },
            }
    raw_mask = raw_probability >= 0.5
    calibrated_mask = oof_probability >= 0.5
    hard_changed_pixels = int(np.count_nonzero(raw_mask != calibrated_mask))
    hard_changed_images = int(np.any(raw_mask != calibrated_mask, axis=(1, 2)).sum())
    checks = {}
    for region in common.REGIONS:
        raw = aggregate[region]["raw"]
        calibrated = aggregate[region]["oof_calibrated"]
        checks[f"{region}_nll_improved"] = (
            calibrated["pixel_nll"] < raw["pixel_nll"]
        )
        checks[f"{region}_ece_at_most_0_05"] = calibrated["pixel_ece"] <= 0.05
    checks["hard_mask_changed_pixels_equals_0"] = hard_changed_pixels == 0
    checks["hard_mask_changed_images_equals_0"] = hard_changed_images == 0
    gate_passed = all(checks.values())
    common.render_binary_diagram(
        ARTIFACTS / "lesion_centered_reliability.png",
        "C02 E08 lesion-centered Dirichlet reliability",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    common.render_binary_diagram(
        ARTIFACTS / "whole_image_reliability.png",
        "C02 E08 whole-image Dirichlet reliability",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    common.write_csv(ARTIFACTS / "fold_parameters.csv", fold_parameters)
    output = {
        "schema_version": 1,
        "experiment_id": "c02_dirichlet",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "Internal BRISC calibration comparison; not clinical validation",
        "source_model": "e08_ensemble",
        "validation_cases": 680,
        "fold_parameters": fold_parameters,
        "final_all_validation": {
            "status": "proposed operational only; not used for reported metrics",
            **final_parameters,
        },
        "metrics": {
            "aggregate": aggregate,
            "per_category": per_category,
            "paired_group_bootstrap": bootstrap,
        },
        "hard_mask_changed_pixels": hard_changed_pixels,
        "hard_mask_changed_images": hard_changed_images,
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "Every regional NLL improves, every regional ECE is at most 0.05, and no hard-mask bit changes at 0.50.",
        },
        "verification": {
            "cache_run": str(cache_run),
            "cache_manifest_sha256": common.sha256(cache_run / "manifest.json"),
            "fold_assignment_sha256": common.sha256(common.FOLDS_PATH),
            "calibration_outputs_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "All metrics use the internal validation partition used during model development.",
            "Lesion-centered and background regions use reference masks for evaluation only.",
            "Patient independence and external generalization are unverified.",
        ],
    }
    common.write_json(EXPERIMENT / "result.json", output)
    config["status"] = "COMPLETE"
    common.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = common.read_json(registry_path)
    next(item for item in registry["experiments"] if item["id"] == "c02_dirichlet")[
        "status"
    ] = "COMPLETE"
    common.write_json(registry_path, registry)
    decision = "ACCEPTED" if gate_passed else "REJECTED"
    (EXPERIMENT / "decision.md").write_text(
        f"""# Two-class Dirichlet pixel calibration decision

**Decision:** {decision}

The predeclared conjunctive gate was {'satisfied' if gate_passed else 'not satisfied'} and was not relaxed. Reported metrics use only out-of-fold probabilities. Hard-mask identity at threshold 0.50 is a mandatory condition.

No model was retrained, no threshold was changed, and no sealed-test image was accessed.
""",
        encoding="utf-8",
    )
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    manifest = {
        "status": "PASS",
        "experiment_id": "c02_dirichlet",
        "registered_decision": output["status"],
        "test_images_accessed": 0,
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": common.sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in included
        ],
    }
    common.write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": output["status"],
                "fold_parameters": fold_parameters,
                "aggregate": aggregate,
                "hard_mask_changed_pixels": hard_changed_pixels,
                "hard_mask_changed_images": hard_changed_images,
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
