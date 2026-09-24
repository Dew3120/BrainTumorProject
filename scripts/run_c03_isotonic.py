"""Run and close C03 isotonic calibration on frozen E08 outputs."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c03_isotonic"
ARTIFACTS = EXPERIMENT / "artifacts"


def fit_isotonic(probabilities, targets):
    """Fit exact weighted PAV after aggregating identical probabilities."""
    probability = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    order = np.argsort(probability, kind="stable")
    sorted_probability = probability[order]
    sorted_target = target[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_probability)]).astype(np.float64)
    positive = np.add.reduceat(sorted_target, starts)
    unique_probability = sorted_probability[starts].astype(np.float64)
    target_mean = positive / counts
    estimator = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    )
    estimator.fit(unique_probability, target_mean, sample_weight=counts)
    return {
        "x_thresholds": estimator.X_thresholds_.astype(float).tolist(),
        "y_thresholds": estimator.y_thresholds_.astype(float).tolist(),
        "fit_pixels": len(probability),
        "unique_input_probabilities": len(unique_probability),
    }


def apply_isotonic(probabilities, parameters):
    probability = np.asarray(probabilities, dtype=np.float64)
    calibrated = np.interp(
        probability,
        np.asarray(parameters["x_thresholds"]),
        np.asarray(parameters["y_thresholds"]),
    )
    clipped = np.clip(calibrated, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)), calibrated


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C03 is not in its registered pre-run state")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    oof_logits = np.empty_like(raw_logits)
    oof_probability = np.empty_like(raw_probability)
    fold_parameters = []
    fold_mappings = []
    for fold in range(5):
        fit = assignment != fold
        evaluate = assignment == fold
        parameters = fit_isotonic(raw_probability[fit], targets[fit])
        transformed, probability = apply_isotonic(
            raw_probability[evaluate], parameters
        )
        oof_logits[evaluate] = transformed
        oof_probability[evaluate] = probability
        fold_mappings.append({"fold": fold + 1, **parameters})
        fold_parameters.append(
            {
                "fold": fold + 1,
                "fit_images": int(fit.sum()),
                "evaluation_images": int(evaluate.sum()),
                "fit_pixels": parameters["fit_pixels"],
                "unique_input_probabilities": parameters[
                    "unique_input_probabilities"
                ],
                "mapping_knots": len(parameters["x_thresholds"]),
            }
        )
    final_parameters = fit_isotonic(raw_probability, targets)
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
    hard_changes_by_category = {}
    for label in common.LABELS:
        indices = np.flatnonzero(label_names == label)
        changed = raw_mask[indices] != calibrated_mask[indices]
        hard_changes_by_category[label] = {
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_images": int(np.any(changed, axis=(1, 2)).sum()),
            "images": len(indices),
        }
    checks = {}
    for region in common.REGIONS:
        raw = aggregate[region]["raw"]
        calibrated = aggregate[region]["oof_calibrated"]
        checks[f"{region}_ece_at_most_0_04"] = calibrated["pixel_ece"] <= 0.04
        checks[f"{region}_brier_not_worse"] = (
            calibrated["pixel_brier"] <= raw["pixel_brier"] + 1e-12
        )
    gate_passed = all(checks.values())
    common.render_binary_diagram(
        ARTIFACTS / "lesion_centered_reliability.png",
        "C03 E08 lesion-centered isotonic reliability",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    common.render_binary_diagram(
        ARTIFACTS / "whole_image_reliability.png",
        "C03 E08 whole-image isotonic reliability",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    common.write_csv(ARTIFACTS / "mapping_summary.csv", fold_parameters)
    common.write_json(
        ARTIFACTS / "isotonic_mappings.json",
        {
            "schema_version": 1,
            "reported_metrics_source": "five out-of-fold mappings",
            "fold_mappings": fold_mappings,
            "final_all_validation_mapping": {
                "status": "proposed operational only; not used for reported metrics",
                **final_parameters,
            },
        },
    )
    output = {
        "schema_version": 1,
        "experiment_id": "c03_isotonic",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "Internal BRISC calibration comparison; not clinical validation",
        "source_model": "e08_ensemble",
        "validation_cases": 680,
        "fold_mapping_summaries": fold_parameters,
        "final_all_validation": {
            "status": "proposed operational only; not used for reported metrics",
            "artifact": "artifacts/isotonic_mappings.json",
            "fit_pixels": final_parameters["fit_pixels"],
            "unique_input_probabilities": final_parameters[
                "unique_input_probabilities"
            ],
            "mapping_knots": len(final_parameters["x_thresholds"]),
        },
        "metrics": {
            "aggregate": aggregate,
            "per_category": per_category,
            "paired_group_bootstrap": bootstrap,
        },
        "hard_mask_changed_pixels": hard_changed_pixels,
        "hard_mask_changed_images": hard_changed_images,
        "hard_mask_changes_by_category": hard_changes_by_category,
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "Every regional ECE is at most 0.04 and every regional Brier score does not worsen (tolerance 1e-12). Hard-mask changes are descriptive, not a gate condition.",
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
    next(item for item in registry["experiments"] if item["id"] == "c03_isotonic")[
        "status"
    ] = "COMPLETE"
    common.write_json(registry_path, registry)
    decision = "ACCEPTED" if gate_passed else "REJECTED"
    (EXPERIMENT / "decision.md").write_text(
        f"""# Isotonic pixel calibration decision

**Decision:** {decision}

The predeclared six-check conjunctive gate was {'satisfied' if gate_passed else 'not satisfied'} and was not relaxed. Reported metrics use only out-of-fold probabilities. The unchanged threshold is `0.50`; hard-mask changes are reported descriptively and are not part of the C03 gate.

Reference masks were used only as supervised fitting targets and to define evaluation regions. The fitted mapping receives only a raw segmenter probability at inference.

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
        "experiment_id": "c03_isotonic",
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
                "fold_mapping_summaries": fold_parameters,
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
