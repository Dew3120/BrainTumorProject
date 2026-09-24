"""Run and close C04 reference-region split temperature calibration."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c04_split_temperature"
ARTIFACTS = EXPERIMENT / "artifacts"


def fit_temperature(logits, targets, regions, indices, region_name):
    """Fit one positive temperature with equal total weight per image."""
    selected_logits = []
    selected_targets = []
    selected_weights = []
    image_weight = 1.0 / len(indices)
    for index in indices:
        mask = common.region_mask(regions, index, region_name)
        count = int(mask.sum())
        selected_logits.append(logits[index][mask])
        selected_targets.append(targets[index][mask].astype(np.float64))
        selected_weights.append(np.full(count, image_weight / count, dtype=np.float64))
    x = np.concatenate(selected_logits)
    y = np.concatenate(selected_targets)
    weight = np.concatenate(selected_weights)

    def objective(value):
        transformed = x / np.exp(value[0])
        error = expit(transformed) - y
        loss = np.sum(weight * (np.logaddexp(0.0, transformed) - y * transformed))
        gradient = -np.sum(weight * error * transformed)
        return float(loss), np.array([gradient])

    result = minimize(
        objective,
        np.zeros(1),
        method="L-BFGS-B",
        jac=True,
        bounds=[(np.log(0.05), np.log(20.0))],
        options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return {
        "temperature": float(np.exp(result.x[0])),
        "fit_images": len(indices),
        "fit_pixels": len(x),
        "fit_objective": float(result.fun),
    }


def apply_split_temperature(logits, regions, indices, lesion, background):
    output = np.empty_like(logits[indices])
    for output_index, image_index in enumerate(indices):
        lesion_mask = regions[image_index]
        output[output_index][lesion_mask] = (
            logits[image_index][lesion_mask] / lesion["temperature"]
        )
        output[output_index][~lesion_mask] = (
            logits[image_index][~lesion_mask] / background["temperature"]
        )
    return output, expit(output)


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C04 is not in its registered pre-run state")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    regions = common.lesion_regions(targets)
    oof_logits = np.empty_like(raw_logits)
    oof_probability = np.empty_like(raw_probability)
    fold_parameters = []
    for fold in range(5):
        fit_indices = np.flatnonzero(assignment != fold)
        evaluation_indices = np.flatnonzero(assignment == fold)
        lesion = fit_temperature(
            raw_logits, targets, regions, fit_indices, "lesion_centered"
        )
        background = fit_temperature(
            raw_logits, targets, regions, fit_indices, "background"
        )
        transformed, probability = apply_split_temperature(
            raw_logits, regions, evaluation_indices, lesion, background
        )
        oof_logits[evaluation_indices] = transformed
        oof_probability[evaluation_indices] = probability
        fold_parameters.append(
            {
                "fold": fold + 1,
                "fit_images": len(fit_indices),
                "evaluation_images": len(evaluation_indices),
                "lesion_centered_temperature": lesion["temperature"],
                "background_temperature": background["temperature"],
                "lesion_centered_fit_pixels": lesion["fit_pixels"],
                "background_fit_pixels": background["fit_pixels"],
            }
        )
    all_indices = np.arange(len(rows))
    final_lesion = fit_temperature(
        raw_logits, targets, regions, all_indices, "lesion_centered"
    )
    final_background = fit_temperature(
        raw_logits, targets, regions, all_indices, "background"
    )
    groups = [row["similarity_group"] for row in rows]
    aggregate = {}
    bootstrap = {}
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
    label_names = np.asarray([row["label"] for row in rows])
    per_category = {}
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
                "raw": {key: raw[key] for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")},
                "oof_calibrated": {key: calibrated[key] for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")},
            }
    raw_mask = raw_probability >= 0.5
    calibrated_mask = oof_probability >= 0.5
    changed = raw_mask != calibrated_mask
    checks = {
        "lesion_centered_ece_at_most_0_05": aggregate["lesion_centered"]["oof_calibrated"]["pixel_ece"] <= 0.05,
        "background_ece_at_most_0_02": aggregate["background"]["oof_calibrated"]["pixel_ece"] <= 0.02,
        "lesion_centered_brier_not_worse": aggregate["lesion_centered"]["oof_calibrated"]["pixel_brier"] <= aggregate["lesion_centered"]["raw"]["pixel_brier"] + 1e-12,
        "background_brier_not_worse": aggregate["background"]["oof_calibrated"]["pixel_brier"] <= aggregate["background"]["raw"]["pixel_brier"] + 1e-12,
    }
    gate_passed = all(checks.values())
    common.render_binary_diagram(
        ARTIFACTS / "lesion_centered_reliability.png",
        "C04 E08 lesion-centered split-temperature reliability",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    common.render_binary_diagram(
        ARTIFACTS / "whole_image_reliability.png",
        "C04 E08 whole-image split-temperature reliability",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    common.write_csv(ARTIFACTS / "fold_parameters.csv", fold_parameters)
    output = {
        "schema_version": 1,
        "experiment_id": "c04_split_temperature",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "Oracle reference-region diagnostic; not operational inference",
        "source_model": "e08_ensemble",
        "validation_cases": len(rows),
        "fold_parameters": fold_parameters,
        "final_all_validation": {
            "status": "proposed oracle diagnostic only; not used for reported metrics and not operationally eligible",
            "lesion_centered": final_lesion,
            "background": final_background,
        },
        "metrics": {"aggregate": aggregate, "per_category": per_category, "paired_group_bootstrap": bootstrap},
        "hard_mask_changed_pixels": int(np.count_nonzero(changed)),
        "hard_mask_changed_images": int(np.any(changed, axis=(1, 2)).sum()),
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "Lesion-centered ECE <=0.05, background ECE <=0.02, and Brier does not worsen in either gated region.",
        },
        "operational_eligibility": "INELIGIBLE_ORACLE_REFERENCE_ROUTING",
        "verification": {
            "cache_run": str(cache_run),
            "cache_manifest_sha256": common.sha256(cache_run / "manifest.json"),
            "fold_assignment_sha256": common.sha256(common.FOLDS_PATH),
            "calibration_outputs_are_out_of_fold": True,
            "reference_masks_used_for_routing": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "C04 requires reference-defined pixel routing and cannot be deployed at inference.",
            "All metrics use the internal validation partition used during model development.",
            "Patient independence and external generalization are unverified.",
        ],
    }
    common.write_json(EXPERIMENT / "result.json", output)
    config["status"] = "COMPLETE"
    common.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = common.read_json(registry_path)
    next(item for item in registry["experiments"] if item["id"] == "c04_split_temperature")["status"] = "COMPLETE"
    common.write_json(registry_path, registry)
    decision = "ACCEPTED_LOCALLY_ORACLE_ONLY" if gate_passed else "REJECTED"
    (EXPERIMENT / "decision.md").write_text(
        f"""# Separate-region temperature calibration decision

**Decision:** {decision}

The predeclared four-check conjunctive gate was {'satisfied' if gate_passed else 'not satisfied'} and was not relaxed. Reported metrics use only out-of-fold probabilities. The hard threshold remained `0.50`.

C04 uses the reference mask to route pixels between temperatures. It is an oracle diagnostic, is unavailable during real inference, and cannot satisfy H7 regardless of its local result.

No model was retrained and no sealed-test image was accessed.
""",
        encoding="utf-8",
    )
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    included = sorted(path for path in EXPERIMENT.rglob("*") if path.is_file() and path != ARTIFACTS / "experiment_manifest.json")
    manifest = {
        "status": "PASS",
        "experiment_id": "c04_split_temperature",
        "registered_decision": output["status"],
        "operational_eligibility": output["operational_eligibility"],
        "test_images_accessed": 0,
        "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": common.sha256(path), "bytes": path.stat().st_size} for path in included],
    }
    common.write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(json.dumps({"status": output["status"], "operational_eligibility": output["operational_eligibility"], "fold_parameters": fold_parameters, "metrics": {name: {"raw": {key: aggregate[name]["raw"][key] for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")}, "oof_calibrated": {key: aggregate[name]["oof_calibrated"][key] for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")}} for name in common.REGIONS}, "hard_mask_changed_pixels": output["hard_mask_changed_pixels"], "hard_mask_changed_images": output["hard_mask_changed_images"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
