"""Run and close R03 extended calibrated-E08 reliability model."""

import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import r03_extended_feature_contract as extended_contract
import stage4_reliability_common as common
from PIL import Image
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r03_extended_features"
ARTIFACTS = EXPERIMENT / "artifacts"


def verify_hash(path, expected, label):
    observed = calibration.sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return observed


def gate_passes(extended_auroc, minimum_auroc):
    return float(extended_auroc) >= float(minimum_auroc)


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
        raise ValueError("Extended coefficient matrix is invalid")
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
    count = int(np.ceil(0.20 * len(reliability)))
    ranked = sorted(
        range(len(reliability)),
        key=lambda index: (
            -(1.0 - reliability[index]),
            calibration.normalized_path(rows[index]["path"]),
        ),
    )
    zero_dice = dice == 0
    captured = int(zero_dice[np.asarray(ranked[:count])].sum())
    total = int(zero_dice.sum())
    return {
        "top_risk_cases": count,
        "captured": captured,
        "total": total,
        "fraction": captured / total if total else None,
    }


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


def auc_many(score_matrix, failure):
    score_matrix = np.atleast_2d(np.asarray(score_matrix, dtype=np.float64))
    failure = np.asarray(failure, dtype=bool)
    positive = int(failure.sum())
    negative = len(failure) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUROC sample lacks both classes")
    ranks = rankdata(score_matrix, axis=1, method="average")
    positive_rank_sum = ranks[:, failure].sum(axis=1)
    return (
        positive_rank_sum - positive * (positive + 1) / 2
    ) / (positive * negative)


def permutation_scores(
    features,
    assignment,
    models,
    feature_index,
    registered_feature_index,
    repetitions,
):
    output = np.empty((repetitions, len(features)), dtype=np.float64)
    for repetition in range(repetitions):
        generator = np.random.default_rng(
            42 + 1000 * registered_feature_index + repetition
        )
        for fold, model in enumerate(models):
            evaluation = np.flatnonzero(assignment == fold)
            permuted = features[evaluation].copy()
            permuted[:, feature_index] = permuted[
                generator.permutation(len(evaluation)), feature_index
            ]
            output[repetition, evaluation] = model.predict_proba(permuted)[:, 1]
    if not np.isfinite(output).all():
        raise ValueError("R03 permutation scoring produced non-finite values")
    return output


def verify_library_versions(config):
    package_names = {
        "scikit_image": "scikit-image",
        "numpy": "numpy",
        "pillow": "pillow",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
    }
    observed = {key: version(name) for key, name in package_names.items()}
    if observed != config["library_versions"]:
        raise ValueError(f"R03 library-version mismatch: {observed}")
    return observed


def extract_extended_features(data, config):
    signature = inspect.signature(extended_contract.extended_prediction_features)
    if tuple(signature.parameters) != ("probability", "model_image", "threshold"):
        raise ValueError("R03 feature API accepts unexpected inputs")
    if tuple(config["new_features"]) != extended_contract.NEW_FEATURES:
        raise ValueError("Registered R03 feature list differs from implementation")

    snapshot_rows = calibration.read_csv(ROOT / config["split_snapshot_path"])
    snapshot = {row["path"]: row for row in snapshot_rows}
    if len(snapshot) != len(snapshot_rows):
        raise ValueError("Split snapshot contains duplicate paths")
    raw_root = (ROOT / "data/raw").resolve()
    matrix = np.empty(
        (len(data["rows"]), len(config["new_features"])), dtype=np.float64
    )
    feature_rows = []
    verification_rows = []
    for index, row in enumerate(data["rows"]):
        source_metadata = snapshot.get(row["path"])
        if (
            source_metadata is None
            or source_metadata["draft_split"] != "validation"
            or not source_metadata.get("mask_path", "")
        ):
            raise ValueError(f"R03 split-snapshot mismatch: {row['path']}")
        source_path = (ROOT / row["path"]).resolve()
        if not source_path.is_relative_to(raw_root):
            raise ValueError(f"R03 source image escapes raw data: {row['path']}")
        observed_sha256 = calibration.sha256(source_path)
        if observed_sha256 != source_metadata["sha256"]:
            raise ValueError(f"R03 raw-image hash mismatch: {row['path']}")
        with Image.open(source_path) as source_image:
            width, height = source_image.size
            model_image = extended_contract.model_space_image(source_image)
        values = extended_contract.extended_prediction_features(
            data["probability"][index], model_image, threshold=0.5
        )
        matrix[index] = [values[name] for name in config["new_features"]]
        feature_rows.append({"path": row["path"], **values})
        verification_rows.append(
            {
                "path": row["path"],
                "expected_sha256": source_metadata["sha256"],
                "observed_sha256": observed_sha256,
                "source_width": width,
                "source_height": height,
                "model_width": 128,
                "model_height": 128,
                "hash_match": True,
            }
        )
    if not np.isfinite(matrix).all():
        raise ValueError("R03 extended feature matrix contains non-finite values")
    return matrix, feature_rows, verification_rows


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R03 is not in its registered pre-run state")

    source_checks = (
        (config["amendment_05_path"], "amendment_05_sha256", "Amendment 05"),
        (config["amendment_06_path"], "amendment_06_sha256", "Amendment 06"),
        (config["r01_result_path"], "r01_result_sha256", "R01 result"),
        (config["r01_manifest_path"], "r01_manifest_sha256", "R01 manifest"),
        (config["r01_oof_path"], "r01_oof_sha256", "R01 OOF table"),
        (config["r01_config_path"], "r01_config_sha256", "R01 config"),
        (config["c03_mappings_path"], "c03_mappings_sha256", "C03 mappings"),
        (
            config["fold_assignment_path"],
            "fold_assignment_sha256",
            "fold assignment",
        ),
        (
            config["classifier_predictions_path"],
            "classifier_predictions_sha256",
            "classifier predictions",
        ),
        (config["split_snapshot_path"], "split_snapshot_sha256", "split snapshot"),
        (
            config["feature_reference_script_path"],
            "feature_reference_script_sha256",
            "feature reference",
        ),
        (config["stage4_common_path"], "stage4_common_sha256", "Stage 4 common"),
        (
            config["segmenter_preprocessing_path"],
            "segmenter_preprocessing_sha256",
            "segmenter preprocessing",
        ),
        (
            config["r03_feature_contract_path"],
            "r03_feature_contract_sha256",
            "R03 feature contract",
        ),
    )
    verified_hashes = {
        label: verify_hash(ROOT / path, config[key], label)
        for path, key, label in source_checks
    }
    library_versions = verify_library_versions(config)

    r01_result = calibration.read_json(ROOT / config["r01_result_path"])
    r01_config = calibration.read_json(ROOT / config["r01_config_path"])
    if r01_result["baseline"]["features"] != config["baseline_features"]:
        raise ValueError("R03 baseline feature list differs from R01")
    if config["extended_features"] != (
        config["baseline_features"] + config["new_features"]
    ):
        raise ValueError("R03 extended feature order mismatch")

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
        raise ValueError("R03 population count mismatch")
    cache_order = data["cache_run"] / "validation_order.csv"
    if calibration.sha256(cache_order) != config["calibration_cache_order_sha256"]:
        raise ValueError("R03 calibration-cache order hash mismatch")

    baseline_features, baseline_contract = common.feature_matrix(
        data["probability"], tuple(config["baseline_features"])
    )
    new_features, new_feature_rows, raw_verification_rows = (
        extract_extended_features(data, config)
    )
    extended_features = np.column_stack((baseline_features, new_features))
    if extended_features.shape != (680, 20) or not np.isfinite(
        extended_features
    ).all():
        raise ValueError("R03 twenty-feature matrix is invalid")

    dice = common.dice_scores(data["probability"], data["targets"])
    success = dice >= config["target"]["threshold"]
    failure = ~success
    if not success.any() or not failure.any():
        raise ValueError("R03 reliability target lacks both classes")
    baseline_reliability, baseline_prevalence, _baseline_models = common.cross_fit(
        baseline_features, success, data["assignment"]
    )
    extended_reliability, extended_prevalence, extended_models = common.cross_fit(
        extended_features, success, data["assignment"]
    )

    r01_oof_rows = calibration.read_csv(ROOT / config["r01_oof_path"])
    if len(r01_oof_rows) != len(data["rows"]):
        raise ValueError("R01 OOF row count mismatch")
    committed_baseline = np.empty(len(r01_oof_rows), dtype=np.float64)
    for index, (current, committed) in enumerate(
        zip(data["rows"], r01_oof_rows, strict=True)
    ):
        if current["path"] != committed["path"]:
            raise ValueError(f"R01 OOF path-order mismatch at row {index}")
        committed_baseline[index] = float(committed["oof_reliability"])
    maximum_oof_difference = float(
        np.max(np.abs(baseline_reliability - committed_baseline))
    )
    tolerance = float(
        config["fail_closed"]["baseline_reproduction_absolute_tolerance"]
    )
    if maximum_oof_difference > tolerance:
        raise ValueError("R03 baseline OOF reproduction mismatch")
    baseline_auroc = common.failure_auroc(baseline_reliability, success)
    baseline_auroc_difference = abs(
        baseline_auroc - float(config["gate"]["comparator_auroc"])
    )
    if baseline_auroc_difference > tolerance:
        raise ValueError("R03 baseline AUROC reproduction mismatch")

    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    bootstrap_samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    baseline_bootstrap = common.bootstrap_auroc(
        baseline_reliability, success, bootstrap_samples
    )
    extended_bootstrap = common.bootstrap_auroc(
        extended_reliability, success, bootstrap_samples
    )
    paired_difference = extended_bootstrap - baseline_bootstrap

    true_labels = np.asarray([row["label"] for row in data["rows"]])
    classifier_source = calibration.read_csv(
        ROOT / config["classifier_predictions_path"]
    )
    classifier_rows = {row["path"]: row for row in classifier_source}
    if len(classifier_rows) != len(classifier_source):
        raise ValueError("Classifier predictions contain duplicate paths")
    predicted_labels = []
    for row in data["rows"]:
        classifier_row = classifier_rows.get(row["path"])
        if classifier_row is None or classifier_row["label"] != row["label"]:
            raise ValueError(f"Classifier path/label mismatch: {row['path']}")
        predicted_labels.append(classifier_row["predicted"])
    predicted_labels = np.asarray(predicted_labels)

    baseline_summary = model_summary(
        baseline_reliability,
        baseline_prevalence,
        success,
        dice,
        data["rows"],
        true_labels,
        predicted_labels,
        baseline_bootstrap,
    )
    extended_summary = model_summary(
        extended_reliability,
        extended_prevalence,
        success,
        dice,
        data["rows"],
        true_labels,
        predicted_labels,
        extended_bootstrap,
    )

    ablation_rows = []
    baseline_count = len(config["baseline_features"])
    for new_index, feature_name in enumerate(config["new_features"]):
        removed_index = baseline_count + new_index
        retained = [
            index
            for index in range(extended_features.shape[1])
            if index != removed_index
        ]
        ablated_reliability, _prevalence, _models = common.cross_fit(
            extended_features, success, data["assignment"], retained
        )
        ablated_auroc = common.failure_auroc(ablated_reliability, success)
        ablation_rows.append(
            {
                "feature": feature_name,
                "extended_failure_detection_auroc": extended_summary[
                    "failure_detection_auroc"
                ],
                "ablated_failure_detection_auroc": ablated_auroc,
                "extended_minus_ablated_auroc": extended_summary[
                    "failure_detection_auroc"
                ]
                - ablated_auroc,
                "rank": 0,
            }
        )
    ablation_rows.sort(
        key=lambda row: (-row["extended_minus_ablated_auroc"], row["feature"])
    )
    for rank, row in enumerate(ablation_rows, start=1):
        row["rank"] = rank

    repetitions = int(
        config["new_feature_permutation"]["repetitions_per_feature"]
    )
    permutation_rows = []
    for new_index, feature_name in enumerate(config["new_features"]):
        feature_index = baseline_count + new_index
        permuted_reliability = permutation_scores(
            extended_features,
            data["assignment"],
            extended_models,
            feature_index,
            new_index,
            repetitions,
        )
        permuted_risk = 1.0 - permuted_reliability
        point_aurocs = auc_many(permuted_risk, failure)
        bootstrap_importance = np.empty(len(bootstrap_samples), dtype=np.float64)
        for bootstrap_index, members in enumerate(bootstrap_samples):
            permuted_aurocs = auc_many(
                permuted_risk[:, members], failure[members]
            )
            bootstrap_importance[bootstrap_index] = extended_bootstrap[
                bootstrap_index
            ] - float(permuted_aurocs.mean())
        permutation_rows.append(
            {
                "feature": feature_name,
                "repetitions": repetitions,
                "extended_failure_detection_auroc": extended_summary[
                    "failure_detection_auroc"
                ],
                "mean_permuted_failure_detection_auroc": float(
                    point_aurocs.mean()
                ),
                "mean_auroc_decrease": extended_summary[
                    "failure_detection_auroc"
                ]
                - float(point_aurocs.mean()),
                "bootstrap_mean_auroc_decrease": float(
                    bootstrap_importance.mean()
                ),
                "ci95_lower": float(np.percentile(bootstrap_importance, 2.5)),
                "ci95_upper": float(np.percentile(bootstrap_importance, 97.5)),
                "rank": 0,
            }
        )
    permutation_rows.sort(
        key=lambda row: (-row["mean_auroc_decrease"], row["feature"])
    )
    for rank, row in enumerate(permutation_rows, start=1):
        row["rank"] = rank

    coefficients = coefficient_summary(
        extended_models, config["extended_features"]
    )
    minimum_auroc = float(config["gate"]["minimum_extended_auroc"])
    passed = gate_passes(
        extended_summary["failure_detection_auroc"], minimum_auroc
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
                "baseline_oof_reliability": float(baseline_reliability[index]),
                "baseline_oof_failure_risk": float(
                    1.0 - baseline_reliability[index]
                ),
                "extended_oof_reliability": float(extended_reliability[index]),
                "extended_oof_failure_risk": float(
                    1.0 - extended_reliability[index]
                ),
            }
        )
    bootstrap_rows = [
        {
            "resample": index + 1,
            "baseline_auroc": float(baseline_bootstrap[index]),
            "extended_auroc": float(extended_bootstrap[index]),
            "extended_minus_baseline_auroc": float(paired_difference[index]),
        }
        for index in range(len(bootstrap_samples))
    ]
    comparison_rows = []
    for name, summary in (
        ("baseline_14_feature", baseline_summary),
        ("extended_20_feature", extended_summary),
    ):
        comparison_rows.append(
            {
                "model": name,
                "failure_detection_auroc": summary["failure_detection_auroc"],
                "bootstrap_ci95_lower": summary["bootstrap"]["ci95_lower"],
                "bootstrap_ci95_upper": summary["bootstrap"]["ci95_upper"],
                "ece": summary["calibration"]["ece"],
                "mce": summary["calibration"]["mce"],
                "brier": summary["calibration"]["brier"],
                "prevalence_baseline_brier": summary["calibration"][
                    "prevalence_baseline_brier"
                ],
                "brier_skill": summary["calibration"]["brier_skill"],
                "zero_dice_captured_top_20_percent": summary[
                    "zero_dice_capture_top_20_percent_risk"
                ]["captured"],
                "zero_dice_total": summary[
                    "zero_dice_capture_top_20_percent_risk"
                ]["total"],
            }
        )
    calibration.write_csv(ARTIFACTS / "raw_image_verification.csv", raw_verification_rows)
    calibration.write_csv(ARTIFACTS / "extended_features.csv", new_feature_rows)
    calibration.write_csv(ARTIFACTS / "oof_model_comparison.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "bootstrap_comparison.csv", bootstrap_rows)
    calibration.write_csv(ARTIFACTS / "model_metrics.csv", comparison_rows)
    calibration.write_csv(ARTIFACTS / "new_feature_ablation.csv", ablation_rows)
    calibration.write_csv(
        ARTIFACTS / "new_feature_permutation_importance.csv", permutation_rows
    )
    calibration.write_csv(
        ARTIFACTS / "extended_fold_coefficients.csv", coefficients
    )

    result = {
        "schema_version": 1,
        "experiment_id": "r03_extended_features",
        "status": decision,
        "decision": decision,
        "scope": "Calibrated-E08 internal-validation reliability analysis; not clinical validation",
        "population": {
            "cases": len(success),
            "successes": int(success.sum()),
            "failures": int(failure.sum()),
            "zero_dice_cases": int((dice == 0).sum()),
        },
        "baseline_14_feature_reproduction": baseline_summary,
        "extended_20_feature_model": {
            "features": config["extended_features"],
            **extended_summary,
            "fold_coefficients": coefficients,
        },
        "paired_comparison": {
            "extended_minus_baseline_auroc": extended_summary[
                "failure_detection_auroc"
            ]
            - baseline_summary["failure_detection_auroc"],
            "bootstrap_repetitions": len(paired_difference),
            "bootstrap_mean": float(paired_difference.mean()),
            "ci95_lower": float(np.percentile(paired_difference, 2.5)),
            "ci95_upper": float(np.percentile(paired_difference, 97.5)),
        },
        "new_feature_ablation": {
            "definition": "extended AUROC minus leave-one-new-feature-out AUROC",
            "ranked_results": ablation_rows,
        },
        "new_feature_permutation_importance": {
            "definition": "extended AUROC minus mean AUROC after within-held-out-fold permutation",
            "repetitions_per_feature": repetitions,
            "ranked_results": permutation_rows,
        },
        "gate": {
            **config["gate"],
            "observed_extended_auroc": extended_summary[
                "failure_detection_auroc"
            ],
            "observed_improvement": extended_summary[
                "failure_detection_auroc"
            ]
            - baseline_summary["failure_detection_auroc"],
            "status": decision,
        },
        "feature_contract": {
            "baseline": baseline_contract,
            "new_feature_function_parameters": list(
                inspect.signature(
                    extended_contract.extended_prediction_features
                ).parameters
            ),
            "new_features": config["new_features"],
            "prediction_only_features": config["feature_contract"][
                "probability_and_mask_only"
            ],
            "raw_image_and_prediction_only_features": config[
                "feature_contract"
            ]["raw_image_and_prediction_only"],
            "all_six_inference_time_computable": True,
            "reference_mask_feature_input": False,
            "classifier_probability_feature_input": False,
            "true_label_feature_input": False,
            "metadata_numeric_feature_input": False,
        },
        "verification": {
            "source_hashes": verified_hashes,
            "library_versions": library_versions,
            "raw_images_verified": len(raw_verification_rows),
            "raw_image_hash_mismatches": 0,
            "image_transform_matches_segmenter_pixel_for_pixel": True,
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
            "maximum_baseline_oof_score_difference_from_r01": maximum_oof_difference,
            "baseline_auroc_absolute_difference_from_r01": baseline_auroc_difference,
            "baseline_reproduction_tolerance": tolerance,
            "reported_scores_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "neural_model_retrained": False,
            "reliability_estimator_refit": True,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "feature_contract_sha256": calibration.sha256(
                ROOT / config["r03_feature_contract_path"]
            ),
        },
        "limitations": [
            "R03 uses internal BRISC validation data and does not establish external generalization.",
            "The same development population informed prior feature work and R03 evaluation.",
            "Raw-image boundary gradients depend on the registered BRISC image rendering and may shift across scanners or preprocessing pipelines.",
            "Ablation, permutation importance, and coefficients describe association rather than causality.",
            "Subgroup AUROCs are descriptive and can be unstable where failures are few.",
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
        if item["id"] == "r03_extended_features"
    )
    if registry_item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R03 registry was not in its registered state")
    registry_item["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)
    (EXPERIMENT / "decision.md").write_text(
        f"""# Extended calibrated-E08 reliability model decision

**Decision:** {decision}

The fourteen-feature baseline reproduced at AUROC `{baseline_summary['failure_detection_auroc']:.12f}`, with absolute aggregate difference `{baseline_auroc_difference:.3g}` and maximum per-case OOF score difference `{maximum_oof_difference:.3g}` from committed R01.

The registered twenty-feature model achieved AUROC `{extended_summary['failure_detection_auroc']:.12f}`. The fixed minimum was `{minimum_auroc:.12f}`; therefore R03 **{decision.lower()}ed** its superiority gate. Extended-minus-baseline AUROC was `{result['paired_comparison']['extended_minus_baseline_auroc']:.12f}`, with paired 95% similarity-group bootstrap interval `[{result['paired_comparison']['ci95_lower']:.12f}, {result['paired_comparison']['ci95_upper']:.12f}]`.

All 680 source images matched their frozen SHA256 values. Five new features used only the calibrated prediction; the boundary-gradient feature used only the uploaded image after preprocessing that matched the segmenter pixel for pixel plus the predicted boundary. No reference-mask value, classifier probability, true type, identity, or metadata entered the reliability estimator. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
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
            "experiment_id": "r03_extended_features",
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
    compact_extended = {
        key: value
        for key, value in extended_summary.items()
        if key != "calibration"
    }
    compact_extended["calibration"] = {
        key: value
        for key, value in extended_summary["calibration"].items()
        if key != "bins"
    }
    print(
        json.dumps(
            {
                "decision": decision,
                "baseline_auroc": baseline_summary["failure_detection_auroc"],
                "baseline_auroc_difference_from_r01": baseline_auroc_difference,
                "maximum_baseline_oof_score_difference_from_r01": maximum_oof_difference,
                "extended": compact_extended,
                "paired_comparison": result["paired_comparison"],
                "new_feature_ablation": ablation_rows,
                "new_feature_permutation_importance": permutation_rows,
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
