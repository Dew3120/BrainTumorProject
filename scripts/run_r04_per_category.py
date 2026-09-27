"""Run and close R04 per-category calibrated-E08 reliability models."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import stage4_reliability_common as common

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r04_per_category"
ARTIFACTS = EXPERIMENT / "artifacts"
CATEGORIES = ("glioma", "meningioma", "pituitary")


def verify_hash(path, expected, label):
    observed = calibration.sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return observed


def gate_passes(category_aurocs, minimum):
    expected = {f"{version}:{category}" for version in ("A", "B") for category in CATEGORIES}
    if set(category_aurocs) != expected:
        raise ValueError("R04 gate did not receive exactly six category AUROCs")
    return all(float(category_aurocs[key]) >= float(minimum) for key in expected)


def category_cross_fit(
    features,
    success,
    assignment,
    categories,
    expected_categories,
    allowed_excluded=(),
):
    categories = np.asarray(categories)
    expected_categories = tuple(expected_categories)
    if set(categories) - set(expected_categories) - set(allowed_excluded):
        raise ValueError("Category cross-fit received an unregistered category")
    scores = np.full(len(success), np.nan, dtype=np.float64)
    baselines = np.full(len(success), np.nan, dtype=np.float64)
    models = {}
    support = []
    for category in expected_categories:
        category_mask = categories == category
        category_indices = np.flatnonzero(category_mask)
        category_success = success[category_indices]
        if len(category_indices) == 0 or len(np.unique(category_success)) != 2:
            raise ValueError(f"Category {category} lacks both outcomes")
        category_models = []
        for fold in range(5):
            fitting = category_mask & (assignment != fold)
            evaluation = category_mask & (assignment == fold)
            fitting_success = success[fitting]
            if not evaluation.any():
                raise ValueError(f"Category {category} has no cases in fold {fold + 1}")
            if len(np.unique(fitting_success)) != 2:
                raise ValueError(
                    f"Category {category} training partition lacks both outcomes in fold {fold + 1}"
                )
            model = common.make_model()
            model.fit(features[fitting], fitting_success)
            scores[evaluation] = model.predict_proba(features[evaluation])[:, 1]
            baselines[evaluation] = float(fitting_success.mean())
            category_models.append(model)
            support.append(
                {
                    "category": category,
                    "fold": fold + 1,
                    "training_cases": int(fitting.sum()),
                    "training_successes": int(fitting_success.sum()),
                    "training_failures": int((~fitting_success).sum()),
                    "evaluation_cases": int(evaluation.sum()),
                    "evaluation_successes": int(success[evaluation].sum()),
                    "evaluation_failures": int((~success[evaluation]).sum()),
                    "training_prevalence": float(fitting_success.mean()),
                }
            )
        models[category] = category_models
    included = np.isin(categories, expected_categories)
    if not np.isfinite(scores[included]).all() or not np.isfinite(baselines[included]).all():
        raise ValueError("Category OOF scoring is incomplete")
    if np.isfinite(scores[~included]).any() or np.isfinite(baselines[~included]).any():
        raise ValueError("Excluded category unexpectedly received an OOF score")
    return scores, baselines, models, support


def reliability_metrics(reliability, baseline, success):
    ece, mce, bins = calibration.equal_frequency_binary(
        reliability, success.astype(np.float64), bins=15
    )
    brier = float(np.mean(np.square(reliability - success)))
    baseline_brier = float(np.mean(np.square(baseline - success)))
    if baseline_brier <= 0:
        raise ValueError("Category prevalence baseline Brier must be positive")
    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier),
        "bins": bins,
    }


def coefficient_rows(version, models, feature_names):
    rows = []
    for category in CATEGORIES:
        matrix = np.vstack(
            [model.named_steps["logistic"].coef_[0] for model in models[category]]
        )
        if matrix.shape != (5, len(feature_names)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {version} {category} coefficient matrix")
        for feature_index, feature in enumerate(feature_names):
            values = matrix[:, feature_index]
            positive = int(np.sum(values > 0))
            negative = int(np.sum(values < 0))
            zero = int(np.sum(values == 0))
            rows.append(
                {
                    "version": version,
                    "category": category,
                    "feature": feature,
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


def category_bootstrap(
    category_reliability,
    aggregate_reliability,
    success,
    category_labels,
    samples,
    minimum_valid,
):
    output = {}
    rows = []
    category_labels = np.asarray(category_labels)
    for category in CATEGORIES:
        category_values = []
        aggregate_values = []
        for resample, members in enumerate(samples, start=1):
            selected = members[category_labels[members] == category]
            if len(selected) == 0 or len(np.unique(success[selected])) != 2:
                rows.append(
                    {
                        "category": category,
                        "resample": resample,
                        "valid": 0,
                        "category_model_auroc": None,
                        "aggregate_model_auroc": None,
                        "category_minus_aggregate_auroc": None,
                    }
                )
                continue
            category_value = common.failure_auroc(
                category_reliability[selected], success[selected]
            )
            aggregate_value = common.failure_auroc(
                aggregate_reliability[selected], success[selected]
            )
            category_values.append(category_value)
            aggregate_values.append(aggregate_value)
            rows.append(
                {
                    "category": category,
                    "resample": resample,
                    "valid": 1,
                    "category_model_auroc": category_value,
                    "aggregate_model_auroc": aggregate_value,
                    "category_minus_aggregate_auroc": category_value
                    - aggregate_value,
                }
            )
        category_values = np.asarray(category_values, dtype=np.float64)
        aggregate_values = np.asarray(aggregate_values, dtype=np.float64)
        if len(category_values) < minimum_valid:
            raise ValueError(
                f"Category {category} has only {len(category_values)} valid bootstrap replicates"
            )
        differences = category_values - aggregate_values
        output[category] = {
            "repetitions_requested": len(samples),
            "valid_repetitions": len(category_values),
            "invalid_repetitions": len(samples) - len(category_values),
            "category_model_ci95_lower": float(np.percentile(category_values, 2.5)),
            "category_model_ci95_upper": float(np.percentile(category_values, 97.5)),
            "aggregate_model_ci95_lower": float(np.percentile(aggregate_values, 2.5)),
            "aggregate_model_ci95_upper": float(np.percentile(aggregate_values, 97.5)),
            "paired_difference_mean": float(differences.mean()),
            "paired_difference_ci95_lower": float(np.percentile(differences, 2.5)),
            "paired_difference_ci95_upper": float(np.percentile(differences, 97.5)),
        }
    return output, rows


def summarize_version(
    reliability,
    baseline,
    aggregate_reliability,
    success,
    category_labels,
    bootstrap,
    registered_comparators,
):
    output = {}
    for category in CATEGORIES:
        indices = np.flatnonzero(category_labels == category)
        category_auroc = common.failure_auroc(reliability[indices], success[indices])
        aggregate_auroc = common.failure_auroc(
            aggregate_reliability[indices], success[indices]
        )
        registered = float(registered_comparators[category])
        if abs(aggregate_auroc - registered) > 1e-12:
            raise ValueError(f"R02 {category} comparator reproduction mismatch")
        delta = category_auroc - aggregate_auroc
        output[category] = {
            "cases": len(indices),
            "successes": int(success[indices].sum()),
            "failures": int((~success[indices]).sum()),
            "failure_detection_auroc": category_auroc,
            "bootstrap": bootstrap[category],
            "calibration": reliability_metrics(
                reliability[indices], baseline[indices], success[indices]
            ),
            "r02_aggregate_comparator_auroc": aggregate_auroc,
            "category_minus_r02_aggregate_auroc": delta,
            "comparison_direction": (
                "improved" if delta > 0 else "degraded" if delta < 0 else "matched"
            ),
        }
    return output


def routing_summary(true_labels, predicted_labels):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    included = np.isin(predicted_labels, CATEGORIES)
    excluded = predicted_labels == "no_tumor"
    if int(included.sum()) != 679 or int(excluded.sum()) != 1:
        raise ValueError("Version B route coverage differs from registration")
    labels = (*CATEGORIES, "no_tumor")
    matrix = []
    for true_label in CATEGORIES:
        row = {"true_label": true_label}
        for predicted_label in labels:
            row[f"predicted_{predicted_label}"] = int(
                np.sum((true_labels == true_label) & (predicted_labels == predicted_label))
            )
        matrix.append(row)
    return {
        "all_cases": len(true_labels),
        "four_class_correct": int(np.sum(true_labels == predicted_labels)),
        "four_class_accuracy": float(np.mean(true_labels == predicted_labels)),
        "three_tumor_routed_cases": int(included.sum()),
        "three_tumor_route_correct": int(
            np.sum((true_labels == predicted_labels) & included)
        ),
        "three_tumor_route_accuracy": float(
            np.mean(true_labels[included] == predicted_labels[included])
        ),
        "route_coverage": float(included.mean()),
        "excluded_no_tumor_cases": int(excluded.sum()),
        "confusion_matrix": matrix,
    }


def manifest_hash(path, manifest):
    relative = path.relative_to(ROOT).as_posix()
    item = next((item for item in manifest["files"] if item["path"] == relative), None)
    if item is None:
        raise ValueError(f"R02 manifest does not list {relative}")
    return item["sha256"]


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R04 is not in its registered pre-run state")
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    registry_item = next(
        item for item in registry["experiments"] if item["id"] == "r04_per_category"
    )
    if registry_item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R04 registry is not in its registered pre-run state")

    source_checks = (
        (config["amendment_07_path"], "amendment_07_sha256", "Amendment 07"),
        ("audit/preregistration.md", "parent_preregistration_sha256", "preregistration"),
        ("audit/preregistration_amendment_03.md", "stage4_parent_amendment_sha256", "Stage 4 amendment"),
        (config["r02_result_path"], "r02_result_sha256", "R02 result"),
        (config["r02_manifest_path"], "r02_manifest_sha256", "R02 manifest"),
        (config["r02_oof_path"], "r02_oof_sha256", "R02 OOF table"),
        (config["c03_mappings_path"], "c03_mappings_sha256", "C03 mappings"),
        (config["fold_assignment_path"], "fold_assignment_sha256", "fold assignment"),
        (config["classifier_predictions_path"], "classifier_predictions_sha256", "classifier predictions"),
        (config["stage4_common_path"], "stage4_common_sha256", "Stage 4 common"),
    )
    verified_hashes = {
        label: verify_hash(ROOT / path, config[key], label)
        for path, key, label in source_checks
    }

    r02_manifest = calibration.read_json(ROOT / config["r02_manifest_path"])
    r02_config_path = ROOT / "experiments/r02_reduced_features/config.json"
    verify_hash(r02_config_path, manifest_hash(r02_config_path, r02_manifest), "R02 config")
    r02_config = calibration.read_json(r02_config_path)
    r01_config_path = ROOT / "experiments/r01_ablation/config.json"
    verify_hash(
        r01_config_path,
        r02_config["r01_config_sha256"],
        "R01 config",
    )
    r01_config = calibration.read_json(r01_config_path)
    r01_result_path = ROOT / r02_config["r01_result_path"]
    verify_hash(r01_result_path, r02_config["r01_result_sha256"], "R01 result")
    r01_result = calibration.read_json(r01_result_path)

    features_registered = list(config["features"])
    if features_registered != r02_config["candidate_selection"]["retained_features"]:
        raise ValueError("R04 feature list differs from R02")
    all_features = list(r01_result["baseline"]["features"])
    feature_lookup = {feature: index for index, feature in enumerate(all_features)}
    if len(feature_lookup) != len(all_features):
        raise ValueError("R01 feature list contains duplicates")
    try:
        selected = np.asarray(
            [feature_lookup[feature] for feature in features_registered], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"Unknown R04 feature: {error.args[0]}") from error

    loader_config = {
        **r02_config,
        "c03_result_sha256": r01_config["c03_result_sha256"],
        "calibration_cache_manifest_sha256": r01_config[
            "calibration_cache_manifest_sha256"
        ],
        "expected_c03_hard_mask_changes": {
            "pixels": config["fail_closed"]["expected_c03_changed_pixels"],
            "images": config["fail_closed"]["expected_c03_changed_images"],
        },
    }
    data = common.load_calibrated_e08(loader_config)
    if len(data["rows"]) != int(config["population"]["cases"]):
        raise ValueError("R04 population count mismatch")
    all_matrix, feature_contract = common.feature_matrix(
        data["probability"], tuple(all_features)
    )
    features = all_matrix[:, selected]
    if not np.isfinite(features).all():
        raise ValueError("R04 feature matrix contains non-finite values")
    feature_contract = {
        **feature_contract,
        "selected_feature_count": len(features_registered),
        "selected_features": features_registered,
        "all_selected_features_inference_time_only": True,
        "version_a_true_category_role": "descriptive grouping only",
        "version_b_classifier_predicted_category_role": "discrete operational router only",
    }

    dice = common.dice_scores(data["probability"], data["targets"], threshold=0.5)
    success = dice >= 0.5
    aggregate_reliability, _aggregate_baseline, _aggregate_models = common.cross_fit(
        features, success, data["assignment"]
    )
    r02_oof = calibration.read_csv(ROOT / config["r02_oof_path"])
    if len(r02_oof) != len(data["rows"]):
        raise ValueError("R02 OOF row count mismatch")
    committed_aggregate = np.empty(len(r02_oof), dtype=np.float64)
    for index, (current, committed) in enumerate(zip(data["rows"], r02_oof, strict=True)):
        if current["path"] != committed["path"]:
            raise ValueError(f"R02 OOF path-order mismatch at row {index}")
        committed_aggregate[index] = float(committed["reduced_oof_reliability"])
    maximum_oof_difference = float(
        np.max(np.abs(aggregate_reliability - committed_aggregate))
    )
    tolerance = float(config["fail_closed"]["r02_oof_reproduction_absolute_tolerance"])
    if maximum_oof_difference > tolerance:
        raise ValueError("R02 aggregate OOF score reproduction mismatch")

    classifier_source = calibration.read_csv(ROOT / config["classifier_predictions_path"])
    classifier_by_path = {row["path"]: row for row in classifier_source}
    if len(classifier_by_path) != len(classifier_source):
        raise ValueError("Classifier predictions contain duplicate paths")
    true_labels = np.asarray([row["label"] for row in data["rows"]])
    predicted_labels = []
    for row in data["rows"]:
        classifier_row = classifier_by_path.get(row["path"])
        if classifier_row is None or classifier_row["label"] != row["label"]:
            raise ValueError(f"Classifier path/label mismatch: {row['path']}")
        predicted_labels.append(classifier_row["predicted"])
    predicted_labels = np.asarray(predicted_labels)

    expected_true_counts = config["population"]["true_category_counts"]
    expected_predicted_counts = config["population"]["predicted_route_counts"]
    observed_true_counts = {
        category: int(np.sum(true_labels == category)) for category in CATEGORIES
    }
    observed_predicted_counts = {
        category: int(np.sum(predicted_labels == category)) for category in CATEGORIES
    }
    observed_predicted_counts["no_tumor_excluded"] = int(
        np.sum(predicted_labels == "no_tumor")
    )
    if observed_true_counts != expected_true_counts:
        raise ValueError("True-category population differs from registration")
    if observed_predicted_counts != expected_predicted_counts:
        raise ValueError("Predicted-route population differs from registration")

    version_a_scores, version_a_baseline, version_a_models, support_a = category_cross_fit(
        features, success, data["assignment"], true_labels, CATEGORIES
    )
    version_b_scores, version_b_baseline, version_b_models, support_b = category_cross_fit(
        features,
        success,
        data["assignment"],
        predicted_labels,
        CATEGORIES,
        allowed_excluded=("no_tumor",),
    )

    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    minimum_valid = int(config["bootstrap"]["minimum_valid_replicates"])
    bootstrap_a, bootstrap_rows_a = category_bootstrap(
        version_a_scores,
        aggregate_reliability,
        success,
        true_labels,
        samples,
        minimum_valid,
    )
    bootstrap_b, bootstrap_rows_b = category_bootstrap(
        version_b_scores,
        aggregate_reliability,
        success,
        predicted_labels,
        samples,
        minimum_valid,
    )
    version_a = summarize_version(
        version_a_scores,
        version_a_baseline,
        aggregate_reliability,
        success,
        true_labels,
        bootstrap_a,
        config["r02_comparator_aurocs"]["A_true_category"],
    )
    version_b = summarize_version(
        version_b_scores,
        version_b_baseline,
        aggregate_reliability,
        success,
        predicted_labels,
        bootstrap_b,
        config["r02_comparator_aurocs"]["B_predicted_category"],
    )
    coefficients = coefficient_rows("A_descriptive", version_a_models, features_registered)
    coefficients += coefficient_rows("B_operational", version_b_models, features_registered)
    routing = routing_summary(true_labels, predicted_labels)

    minimum_auroc = float(config["gate"]["minimum_auroc"])
    category_aurocs = {
        **{f"A:{category}": version_a[category]["failure_detection_auroc"] for category in CATEGORIES},
        **{f"B:{category}": version_b[category]["failure_detection_auroc"] for category in CATEGORIES},
    }
    checks = {
        key: {
            "observed_auroc": value,
            "minimum_auroc": minimum_auroc,
            "operator": ">=",
            "passed": value >= minimum_auroc,
        }
        for key, value in category_aurocs.items()
    }
    passed = gate_passes(category_aurocs, minimum_auroc)
    decision = "PASS" if passed else "FAIL"

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    oof_rows = []
    for index, row in enumerate(data["rows"]):
        oof_rows.append(
            {
                "path": row["path"],
                "true_label": true_labels[index],
                "classifier_predicted_label": predicted_labels[index],
                "similarity_group": row["similarity_group"],
                "fold": int(data["assignment"][index]) + 1,
                "calibrated_dice": float(dice[index]),
                "success_dice_at_least_0_50": int(success[index]),
                "r02_aggregate_oof_reliability": float(aggregate_reliability[index]),
                "version_a_oof_reliability": float(version_a_scores[index]),
                "version_b_oof_reliability": (
                    float(version_b_scores[index])
                    if np.isfinite(version_b_scores[index])
                    else None
                ),
                "version_b_excluded_no_tumor_route": int(
                    predicted_labels[index] == "no_tumor"
                ),
            }
        )
    metric_rows = []
    for version_name, summary in (("A_descriptive", version_a), ("B_operational", version_b)):
        for category in CATEGORIES:
            item = summary[category]
            metric_rows.append(
                {
                    "version": version_name,
                    "category": category,
                    "cases": item["cases"],
                    "successes": item["successes"],
                    "failures": item["failures"],
                    "failure_detection_auroc": item["failure_detection_auroc"],
                    "bootstrap_ci95_lower": item["bootstrap"]["category_model_ci95_lower"],
                    "bootstrap_ci95_upper": item["bootstrap"]["category_model_ci95_upper"],
                    "valid_bootstrap_repetitions": item["bootstrap"]["valid_repetitions"],
                    "ece": item["calibration"]["ece"],
                    "mce": item["calibration"]["mce"],
                    "brier": item["calibration"]["brier"],
                    "prevalence_baseline_brier": item["calibration"]["prevalence_baseline_brier"],
                    "brier_skill": item["calibration"]["brier_skill"],
                    "r02_aggregate_comparator_auroc": item["r02_aggregate_comparator_auroc"],
                    "category_minus_r02_aggregate_auroc": item["category_minus_r02_aggregate_auroc"],
                    "comparison_direction": item["comparison_direction"],
                    "gate_passed": item["failure_detection_auroc"] >= minimum_auroc,
                }
            )
    bootstrap_rows = [
        {"version": "A_descriptive", **row} for row in bootstrap_rows_a
    ] + [{"version": "B_operational", **row} for row in bootstrap_rows_b]
    calibration.write_csv(ARTIFACTS / "oof_category_scores.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "category_metrics.csv", metric_rows)
    calibration.write_csv(ARTIFACTS / "category_bootstrap_comparisons.csv", bootstrap_rows)
    calibration.write_csv(ARTIFACTS / "category_fold_coefficients.csv", coefficients)
    support_rows = [
        {"version": "A_descriptive", **row} for row in support_a
    ] + [{"version": "B_operational", **row} for row in support_b]
    calibration.write_csv(ARTIFACTS / "category_fold_support.csv", support_rows)
    calibration.write_csv(ARTIFACTS / "routing_confusion_matrix.csv", routing["confusion_matrix"])

    result = {
        "schema_version": 1,
        "experiment_id": "r04_per_category",
        "status": decision,
        "decision": decision,
        "scope": "Calibrated-E08 internal-validation reliability analysis; not clinical validation",
        "population": {
            "cases": len(success),
            "successes": int(success.sum()),
            "failures": int((~success).sum()),
            "true_category_counts": observed_true_counts,
            "predicted_route_counts": observed_predicted_counts,
        },
        "version_a_descriptive_true_category": version_a,
        "version_b_operational_predicted_category": version_b,
        "routing": routing,
        "coefficients": coefficients,
        "gate": {
            **config["gate"],
            "checks": checks,
            "status": decision,
        },
        "feature_contract": feature_contract,
        "verification": {
            "source_hashes": verified_hashes,
            "r02_config_sha256_verified_from_manifest": calibration.sha256(r02_config_path),
            "r02_maximum_per_case_oof_score_difference": maximum_oof_difference,
            "r02_reproduction_tolerance": tolerance,
            "all_six_r02_subgroup_comparators_reproduced": True,
            "c03_changed_pixels": data["changed_pixels"],
            "c03_changed_images": data["changed_images"],
            "population_cases": len(success),
            "fold_count": len(set(data["assignment"])),
            "similarity_group_count": len(set(groups)),
            "similarity_group_cross_fold_leakage": False,
            "every_category_training_partition_has_both_outcomes": True,
            "version_b_included_cases": int(np.isin(predicted_labels, CATEGORIES).sum()),
            "version_b_excluded_no_tumor_cases": int(np.sum(predicted_labels == "no_tumor")),
            "all_selected_features_inference_time_only": True,
            "reference_mask_feature_leakage": False,
            "classifier_probability_feature_leakage": False,
            "true_category_operational_leakage": False,
            "reported_scores_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "neural_model_retrained": False,
            "reliability_estimators_refit": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "R04 uses internal BRISC validation data and does not establish external or clinical generalization.",
            "Version A uses true category and is descriptive only; it cannot be routed operationally without an unavailable ground-truth label.",
            "Version B depends on the frozen classifier route and excludes its one no-tumor prediction rather than inventing a fallback.",
            "Meningioma has few failures, so its AUROC and coefficient estimates can be unstable despite group-bootstrap reporting.",
            "Coefficients are descriptive associations among correlated features, not causal effects.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    config["status"] = "COMPLETE"
    calibration.write_json(config_path, config)
    registry_item["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)

    a_gl = version_a["glioma"]
    b_gl = version_b["glioma"]
    a_pi = version_a["pituitary"]
    b_pi = version_b["pituitary"]
    (EXPERIMENT / "decision.md").write_text(
        f"""# Per-category calibrated-E08 reliability decision

**Decision:** {decision}

R04 required all six true-category and classifier-predicted-category models to achieve failure-detection AUROC at least `{minimum_auroc:.2f}`. The unchanged conjunctive gate **{decision.lower()}ed**.

Version A AUROCs were glioma `{a_gl['failure_detection_auroc']:.6f}`, meningioma `{version_a['meningioma']['failure_detection_auroc']:.6f}`, and pituitary `{a_pi['failure_detection_auroc']:.6f}`. Version B AUROCs were predicted-glioma `{b_gl['failure_detection_auroc']:.6f}`, predicted-meningioma `{version_b['meningioma']['failure_detection_auroc']:.6f}`, and predicted-pituitary `{b_pi['failure_detection_auroc']:.6f}`.

Glioma changed by `{a_gl['category_minus_r02_aggregate_auroc']:+.6f}` in Version A and `{b_gl['category_minus_r02_aggregate_auroc']:+.6f}` in Version B relative to the identical-case R02 aggregate model. Pituitary changed by `{a_pi['category_minus_r02_aggregate_auroc']:+.6f}` in Version A and `{b_pi['category_minus_r02_aggregate_auroc']:+.6f}` in Version B. These subgroup comparisons are internal descriptive findings.

Version B routed `{routing['three_tumor_routed_cases']}/680` cases to a tumor-specific reliability model. The single frozen-classifier `no_tumor` prediction was recorded and excluded without fallback. Four-class routing accuracy was `{routing['four_class_accuracy']:.6f}`; accuracy among the 679 tumor-routed cases was `{routing['three_tumor_route_accuracy']:.6f}`.

All ten model features derive only from the C03 out-of-fold calibrated probability map and threshold-`0.50` prediction. True category entered Version A grouping and routing evaluation only; it did not enter Version B. No reference-mask value or classifier probability entered a reliability model. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
""",
        encoding="utf-8",
    )

    included_files = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    calibration.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": decision,
            "experiment_id": "r04_per_category",
            "registered_decision": decision,
            "test_images_accessed": 0,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": calibration.sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in included_files
            ],
        },
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "version_a": version_a,
                "version_b": version_b,
                "routing": routing,
                "gate": result["gate"],
                "verification": result["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

