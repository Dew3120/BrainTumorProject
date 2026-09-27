"""Run and close R02 reduced calibrated-E08 reliability model."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import stage4_reliability_common as common

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r02_reduced_features"
ARTIFACTS = EXPERIMENT / "artifacts"


def verify_hash(path, expected, label):
    observed = calibration.sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return observed


def reliability_metrics(reliability, baseline_reliability, success):
    ece, mce, bins = calibration.equal_frequency_binary(
        reliability, success.astype(np.float64), bins=15
    )
    brier = float(np.mean(np.square(reliability - success)))
    baseline_brier = float(np.mean(np.square(baseline_reliability - success)))
    if baseline_brier <= 0:
        raise ValueError("Prevalence baseline Brier must be positive")
    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier),
        "bins": bins,
    }


def coefficient_summary(models, feature_names):
    matrix = np.vstack(
        [model.named_steps["logistic"].coef_[0] for model in models]
    )
    if matrix.shape != (5, len(feature_names)) or not np.isfinite(matrix).all():
        raise ValueError("Reduced coefficient matrix is invalid")
    rows = []
    for index, name in enumerate(feature_names):
        values = matrix[:, index]
        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
        zero = int(np.sum(values == 0))
        rows.append(
            {
                "feature": name,
                **{
                    f"fold_{fold + 1}_standardized_coefficient": float(value)
                    for fold, value in enumerate(values)
                },
                "mean_standardized_coefficient": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "positive_folds": positive,
                "negative_folds": negative,
                "zero_folds": zero,
                "all_folds_same_nonzero_sign": max(positive, negative) == 5,
                "stable_sign": (
                    "positive"
                    if positive == 5
                    else "negative"
                    if negative == 5
                    else "zero"
                    if zero == 5
                    else "mixed"
                ),
            }
        )
    return rows


def subgroup_aurocs(reliability, success, group_names):
    output = {}
    for name in sorted(set(group_names)):
        indices = np.flatnonzero(group_names == name)
        failures = int((~success[indices]).sum())
        successes = int(success[indices].sum())
        output[name] = {
            "cases": len(indices),
            "failures": failures,
            "successes": successes,
            "failure_detection_auroc": (
                common.failure_auroc(reliability[indices], success[indices])
                if failures and successes
                else None
            ),
        }
    return output


def top_risk_capture(reliability, dice, rows):
    top_risk_cases = int(np.ceil(0.20 * len(reliability)))
    ranked = sorted(
        range(len(reliability)),
        key=lambda index: (
            -(1.0 - reliability[index]),
            calibration.normalized_path(rows[index]["path"]),
        ),
    )
    zero_dice = dice == 0
    captured = int(zero_dice[np.asarray(ranked[:top_risk_cases])].sum())
    total = int(zero_dice.sum())
    return {
        "top_risk_cases": top_risk_cases,
        "captured": captured,
        "total": total,
        "fraction": captured / total if total else None,
    }


def gate_passes(reduced_auroc, minimum_auroc):
    return float(reduced_auroc) >= float(minimum_auroc)


def candidate_indices(all_features, candidates, expected_candidates):
    if list(candidates) != list(expected_candidates):
        raise ValueError("R02 candidate list differs from committed R01 screen")
    if len(candidates) != 10 or len(set(candidates)) != 10:
        raise ValueError("R02 candidate list must contain ten unique features")
    lookup = {name: index for index, name in enumerate(all_features)}
    if len(lookup) != len(all_features):
        raise ValueError("Full feature list contains duplicates")
    try:
        return np.asarray([lookup[name] for name in candidates], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown R02 candidate feature: {error.args[0]}") from error


def model_summary(
    reliability,
    prevalence_baseline,
    success,
    dice,
    rows,
    true_labels,
    predicted_labels,
    bootstrap_values,
):
    return {
        "failure_detection_auroc": common.failure_auroc(reliability, success),
        "bootstrap": {
            "repetitions": len(bootstrap_values),
            "mean": float(bootstrap_values.mean()),
            "ci95_lower": float(np.percentile(bootstrap_values, 2.5)),
            "ci95_upper": float(np.percentile(bootstrap_values, 97.5)),
        },
        "calibration": reliability_metrics(
            reliability, prevalence_baseline, success
        ),
        "zero_dice_capture_top_20_percent_risk": top_risk_capture(
            reliability, dice, rows
        ),
        "true_category_aurocs_descriptive": subgroup_aurocs(
            reliability, success, true_labels
        ),
        "classifier_predicted_category_aurocs_descriptive": subgroup_aurocs(
            reliability, success, predicted_labels
        ),
    }


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R02 is not in its registered pre-run state")

    source_checks = (
        (config["amendment_04_path"], "amendment_04_sha256", "Amendment 04"),
        (config["r01_result_path"], "r01_result_sha256", "R01 result"),
        (config["r01_manifest_path"], "r01_manifest_sha256", "R01 manifest"),
        (config["r01_oof_path"], "r01_oof_sha256", "R01 OOF table"),
        (
            "experiments/r01_ablation/config.json",
            "r01_config_sha256",
            "R01 config",
        ),
        (config["c03_mappings_path"], "c03_mappings_sha256", "C03 mappings"),
        (
            config["fold_assignment_path"],
            "fold_assignment_sha256",
            "fold assignment",
        ),
        (
            config["feature_reference_script_path"],
            "feature_reference_script_sha256",
            "feature reference",
        ),
        (config["stage4_common_path"], "stage4_common_sha256", "Stage 4 common"),
        (
            config["classifier_predictions_path"],
            "classifier_predictions_sha256",
            "classifier predictions",
        ),
    )
    verified_hashes = {
        label: verify_hash(ROOT / path, config[key], label)
        for path, key, label in source_checks
    }

    r01_result = calibration.read_json(ROOT / config["r01_result_path"])
    r01_config = calibration.read_json(ROOT / "experiments/r01_ablation/config.json")
    expected_candidates = [
        item["feature"] for item in r01_result["r02_candidate_screen"]["features"]
    ]
    candidates = config["candidate_selection"]["retained_features"]
    all_features = r01_result["baseline"]["features"]
    selected_indices = candidate_indices(
        all_features, candidates, expected_candidates
    )

    loader_config = {
        **config,
        "c03_result_sha256": r01_config["c03_result_sha256"],
        "expected_c03_hard_mask_changes": {
            "pixels": config["fail_closed"]["expected_c03_changed_pixels"],
            "images": config["fail_closed"]["expected_c03_changed_images"],
        },
    }
    data = common.load_calibrated_e08(loader_config)
    if len(data["rows"]) != config["population"]["cases"]:
        raise ValueError("R02 population count mismatch")
    features, full_contract = common.feature_matrix(
        data["probability"], tuple(all_features)
    )
    if not np.isfinite(features[:, selected_indices]).all():
        raise ValueError("Reduced feature matrix contains non-finite values")
    reduced_contract = {
        **full_contract,
        "selected_feature_count": len(candidates),
        "selected_features": list(candidates),
        "all_selected_features_inference_time_only": True,
    }

    dice = common.dice_scores(data["probability"], data["targets"])
    success = dice >= config["target"]["threshold"]
    if not success.any() or success.all():
        raise ValueError("R02 reliability target lacks both classes")
    full_reliability, full_prevalence, _full_models = common.cross_fit(
        features, success, data["assignment"]
    )
    reduced_reliability, reduced_prevalence, reduced_models = common.cross_fit(
        features, success, data["assignment"], selected_indices
    )

    r01_oof_rows = calibration.read_csv(ROOT / config["r01_oof_path"])
    if len(r01_oof_rows) != len(data["rows"]):
        raise ValueError("R01 OOF row count mismatch")
    committed_full_reliability = np.empty(len(r01_oof_rows), dtype=np.float64)
    for index, (current, committed) in enumerate(
        zip(data["rows"], r01_oof_rows, strict=True)
    ):
        if current["path"] != committed["path"]:
            raise ValueError(f"R01 OOF path-order mismatch at row {index}")
        committed_full_reliability[index] = float(committed["oof_reliability"])
    maximum_oof_difference = float(
        np.max(np.abs(full_reliability - committed_full_reliability))
    )
    tolerance = float(
        config["fail_closed"]["baseline_reproduction_absolute_tolerance"]
    )
    if maximum_oof_difference > tolerance:
        raise ValueError("R01 OOF reliability reproduction mismatch")
    full_auroc = common.failure_auroc(full_reliability, success)
    registered_full_auroc = float(config["gate"]["comparator_auroc"])
    full_auroc_difference = abs(full_auroc - registered_full_auroc)
    if full_auroc_difference > tolerance:
        raise ValueError("R01 baseline AUROC reproduction mismatch")

    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    bootstrap_samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    full_bootstrap = common.bootstrap_auroc(
        full_reliability, success, bootstrap_samples
    )
    reduced_bootstrap = common.bootstrap_auroc(
        reduced_reliability, success, bootstrap_samples
    )
    paired_difference = reduced_bootstrap - full_bootstrap

    true_labels = np.asarray([row["label"] for row in data["rows"]])
    classifier_rows_source = calibration.read_csv(
        ROOT / config["classifier_predictions_path"]
    )
    classifier_rows = {row["path"]: row for row in classifier_rows_source}
    if len(classifier_rows) != len(classifier_rows_source):
        raise ValueError("Classifier predictions contain duplicate paths")
    predicted_labels = []
    for row in data["rows"]:
        classifier_row = classifier_rows.get(row["path"])
        if classifier_row is None or classifier_row["label"] != row["label"]:
            raise ValueError(f"Classifier path/label mismatch: {row['path']}")
        predicted_labels.append(classifier_row["predicted"])
    predicted_labels = np.asarray(predicted_labels)

    full_summary = model_summary(
        full_reliability,
        full_prevalence,
        success,
        dice,
        data["rows"],
        true_labels,
        predicted_labels,
        full_bootstrap,
    )
    reduced_summary = model_summary(
        reduced_reliability,
        reduced_prevalence,
        success,
        dice,
        data["rows"],
        true_labels,
        predicted_labels,
        reduced_bootstrap,
    )
    coefficients = coefficient_summary(reduced_models, candidates)
    minimum_auroc = float(config["gate"]["minimum_reduced_auroc"])
    passed = gate_passes(
        reduced_summary["failure_detection_auroc"], minimum_auroc
    )
    decision = "PASS" if passed else "FAIL"

    oof_rows = []
    for index, row in enumerate(data["rows"]):
        oof_rows.append(
            {
                "path": row["path"],
                "true_label": row["label"],
                "classifier_predicted_label": predicted_labels[index],
                "similarity_group": row["similarity_group"],
                "fold": int(data["assignment"][index]) + 1,
                "calibrated_dice": float(dice[index]),
                "success_dice_at_least_0_50": int(success[index]),
                "zero_dice": int(dice[index] == 0),
                "full_oof_reliability": float(full_reliability[index]),
                "full_oof_failure_risk": float(1.0 - full_reliability[index]),
                "reduced_oof_reliability": float(reduced_reliability[index]),
                "reduced_oof_failure_risk": float(
                    1.0 - reduced_reliability[index]
                ),
            }
        )
    bootstrap_rows = [
        {
            "resample": index + 1,
            "full_auroc": float(full_bootstrap[index]),
            "reduced_auroc": float(reduced_bootstrap[index]),
            "reduced_minus_full_auroc": float(paired_difference[index]),
        }
        for index in range(len(bootstrap_samples))
    ]
    comparison_rows = []
    for name, summary in (("full_14_feature", full_summary), ("reduced_10_feature", reduced_summary)):
        comparison_rows.append(
            {
                "model": name,
                "failure_detection_auroc": summary["failure_detection_auroc"],
                "bootstrap_ci95_lower": summary["bootstrap"]["ci95_lower"],
                "bootstrap_ci95_upper": summary["bootstrap"]["ci95_upper"],
                "ece": summary["calibration"]["ece"],
                "mce": summary["calibration"]["mce"],
                "brier": summary["calibration"]["brier"],
                "prevalence_baseline_brier": summary["calibration"]["prevalence_baseline_brier"],
                "brier_skill": summary["calibration"]["brier_skill"],
                "zero_dice_captured_top_20_percent": summary[
                    "zero_dice_capture_top_20_percent_risk"
                ]["captured"],
                "zero_dice_total": summary[
                    "zero_dice_capture_top_20_percent_risk"
                ]["total"],
            }
        )
    calibration.write_csv(ARTIFACTS / "oof_model_comparison.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "bootstrap_comparison.csv", bootstrap_rows)
    calibration.write_csv(ARTIFACTS / "model_metrics.csv", comparison_rows)
    calibration.write_csv(
        ARTIFACTS / "reduced_fold_coefficients.csv", coefficients
    )

    result = {
        "schema_version": 1,
        "experiment_id": "r02_reduced_features",
        "status": decision,
        "decision": decision,
        "scope": "Calibrated-E08 internal-validation reliability analysis; not clinical validation",
        "population": {
            "cases": len(success),
            "successes": int(success.sum()),
            "failures": int((~success).sum()),
            "zero_dice_cases": int((dice == 0).sum()),
        },
        "full_14_feature_reproduction": full_summary,
        "reduced_10_feature_model": {
            "features": list(candidates),
            **reduced_summary,
            "fold_coefficients": coefficients,
        },
        "paired_comparison": {
            "reduced_minus_full_auroc": reduced_summary[
                "failure_detection_auroc"
            ]
            - full_summary["failure_detection_auroc"],
            "bootstrap_repetitions": len(paired_difference),
            "bootstrap_mean": float(paired_difference.mean()),
            "ci95_lower": float(np.percentile(paired_difference, 2.5)),
            "ci95_upper": float(np.percentile(paired_difference, 97.5)),
        },
        "gate": {
            **config["gate"],
            "observed_reduced_auroc": reduced_summary[
                "failure_detection_auroc"
            ],
            "status": decision,
        },
        "feature_contract": reduced_contract,
        "verification": {
            "source_hashes": verified_hashes,
            "candidate_list_matches_r01": True,
            "candidate_order_matches_r01": True,
            "c03_changed_pixels": data["changed_pixels"],
            "c03_changed_images": data["changed_images"],
            "population_cases": len(success),
            "fold_count": len(set(data["assignment"])),
            "fold_case_counts": {
                str(fold + 1): int(np.sum(data["assignment"] == fold))
                for fold in range(5)
            },
            "similarity_group_count": len(set(groups)),
            "similarity_group_cross_fold_leakage": False,
            "maximum_full_oof_score_difference_from_r01": maximum_oof_difference,
            "full_auroc_absolute_difference_from_r01": full_auroc_difference,
            "baseline_reproduction_tolerance": tolerance,
            "all_selected_features_inference_time_only": True,
            "reference_mask_feature_leakage": False,
            "classifier_probability_feature_leakage": False,
            "raw_image_feature_leakage": False,
            "reported_scores_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "neural_model_retrained": False,
            "reliability_estimator_refit": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "R02 uses internal BRISC validation data and does not establish external generalization.",
            "The same development population informed R01 feature selection and R02 evaluation; R02 is a registered development comparison, not a new external test.",
            "Subgroup AUROCs are descriptive and can be unstable in groups with few failures.",
            "Coefficient signs describe associations among correlated prediction-derived features, not causal effects.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    config["status"] = "COMPLETE"
    calibration.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    registry_item = next(
        item
        for item in registry["experiments"]
        if item["id"] == "r02_reduced_features"
    )
    if registry_item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R02 registry was not in its registered state")
    registry_item["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)
    (EXPERIMENT / "decision.md").write_text(
        f"""# Reduced calibrated-E08 reliability model decision

**Decision:** {decision}

The fourteen-feature baseline reproduced at AUROC `{full_summary['failure_detection_auroc']:.12f}`, with absolute difference `{full_auroc_difference:.3g}` from committed R01 and maximum per-case OOF score difference `{maximum_oof_difference:.3g}`.

The registered ten-feature model achieved AUROC `{reduced_summary['failure_detection_auroc']:.12f}`. The fixed minimum was `{minimum_auroc:.12f}`; therefore R02 **{decision.lower()}ed** its noninferiority gate. Reduced-minus-full AUROC was `{result['paired_comparison']['reduced_minus_full_auroc']:.12f}`, with paired 95% similarity-group bootstrap interval `[{result['paired_comparison']['ci95_lower']:.12f}, {result['paired_comparison']['ci95_upper']:.12f}]`.

All ten features derive only from the C03 out-of-fold calibrated probability map and threshold-`0.50` prediction. No reference-mask value, classifier probability, raw image, path, identity, or metadata entered the reliability estimator. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
""",
        encoding="utf-8",
    )
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    calibration.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": decision,
            "experiment_id": "r02_reduced_features",
            "registered_decision": decision,
            "test_images_accessed": 0,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": calibration.sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in included
            ],
        },
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "full_auroc": full_summary["failure_detection_auroc"],
                "full_auroc_difference_from_r01": full_auroc_difference,
                "maximum_full_oof_score_difference_from_r01": maximum_oof_difference,
                "reduced": reduced_summary,
                "paired_comparison": result["paired_comparison"],
                "coefficients": coefficients,
                "gate": result["gate"],
                "feature_contract": result["feature_contract"],
                "verification": result["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
