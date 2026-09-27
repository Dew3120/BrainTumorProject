"""Run and close R05 category-aware calibrated-E08 reliability model."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import stage4_reliability_common as common

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r05_category_aware"
ARTIFACTS = EXPERIMENT / "artifacts"
CATEGORIES = ("glioma", "meningioma", "pituitary")


def verify_hash(path, expected, label):
    observed = calibration.sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return observed


def manifest_hash(path, manifest, label):
    relative = path.relative_to(ROOT).as_posix()
    item = next((item for item in manifest["files"] if item["path"] == relative), None)
    if item is None:
        raise ValueError(f"{label} manifest does not list {relative}")
    return item["sha256"]


def gate_passes(predicted_glioma_auroc, minimum):
    return float(predicted_glioma_auroc) >= float(minimum)


def one_hot_categories(predicted_labels, categories=CATEGORIES):
    predicted_labels = np.asarray(predicted_labels)
    matrix = np.column_stack(
        [(predicted_labels == category).astype(np.float64) for category in categories]
    )
    included = np.isin(predicted_labels, categories)
    if not np.all(matrix[included].sum(axis=1) == 1.0):
        raise ValueError("Included one-hot rows do not sum to one")
    if np.any(matrix[~included].sum(axis=1) != 0.0):
        raise ValueError("Excluded category unexpectedly activates a one-hot feature")
    return matrix, included


def cross_fit_included(features, success, assignment, included):
    scores = np.full(len(success), np.nan, dtype=np.float64)
    baselines = np.full(len(success), np.nan, dtype=np.float64)
    models = []
    support = []
    for fold in range(5):
        fitting = included & (assignment != fold)
        evaluation = included & (assignment == fold)
        fitting_success = success[fitting]
        if not evaluation.any():
            raise ValueError(f"R05 has no included evaluation cases in fold {fold + 1}")
        if len(np.unique(fitting_success)) != 2:
            raise ValueError(f"R05 training partition lacks both outcomes in fold {fold + 1}")
        model = common.make_model()
        model.fit(features[fitting], fitting_success)
        scores[evaluation] = model.predict_proba(features[evaluation])[:, 1]
        baselines[evaluation] = float(fitting_success.mean())
        models.append(model)
        support.append(
            {
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
    if not np.isfinite(scores[included]).all() or not np.isfinite(baselines[included]).all():
        raise ValueError("R05 included OOF scores are incomplete")
    if np.isfinite(scores[~included]).any() or np.isfinite(baselines[~included]).any():
        raise ValueError("Excluded R05 case received an OOF score")
    return scores, baselines, models, support


def reliability_metrics(reliability, baseline, success):
    ece, mce, bins = calibration.equal_frequency_binary(
        reliability, success.astype(np.float64), bins=15
    )
    brier = float(np.mean(np.square(reliability - success)))
    baseline_brier = float(np.mean(np.square(baseline - success)))
    if baseline_brier <= 0:
        raise ValueError("R05 prevalence baseline Brier must be positive")
    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier),
        "baseline_definition": "single-model fold-training prevalence evaluated on this analysis subset",
        "bins": bins,
    }


def coefficient_summary(models, feature_names):
    matrix = np.vstack(
        [model.named_steps["logistic"].coef_[0] for model in models]
    )
    if matrix.shape != (5, len(feature_names)) or not np.isfinite(matrix).all():
        raise ValueError("R05 coefficient matrix is invalid")
    rows = []
    for index, feature in enumerate(feature_names):
        values = matrix[:, index]
        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
        zero = int(np.sum(values == 0))
        rows.append(
            {
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


def make_analyses(included, true_labels, predicted_labels):
    analyses = {"overall": np.asarray(included, dtype=bool)}
    for category in CATEGORIES:
        analyses[f"true_{category}"] = included & (true_labels == category)
        analyses[f"predicted_{category}"] = predicted_labels == category
    return analyses


def validate_analysis_support(analyses, success):
    for name, mask in analyses.items():
        if not mask.any() or len(np.unique(success[mask])) != 2:
            raise ValueError(f"R05 analysis {name} lacks both outcomes")


def bootstrap_analyses(
    r05_scores,
    comparator_scores,
    success,
    analyses,
    samples,
    minimum_valid_subgroup,
):
    summaries = {}
    rows = []
    for analysis, mask in analyses.items():
        model_values = []
        comparator_values = {name: [] for name in comparator_scores[analysis]}
        invalid = 0
        for resample, members in enumerate(samples, start=1):
            selected = members[mask[members]]
            if len(selected) == 0 or len(np.unique(success[selected])) != 2:
                invalid += 1
                for comparator in comparator_scores[analysis]:
                    rows.append(
                        {
                            "analysis": analysis,
                            "comparator": comparator,
                            "resample": resample,
                            "valid": 0,
                            "r05_auroc": None,
                            "comparator_auroc": None,
                            "r05_minus_comparator_auroc": None,
                        }
                    )
                continue
            r05_value = common.failure_auroc(r05_scores[selected], success[selected])
            model_values.append(r05_value)
            for comparator, scores in comparator_scores[analysis].items():
                comparator_value = common.failure_auroc(scores[selected], success[selected])
                comparator_values[comparator].append(comparator_value)
                rows.append(
                    {
                        "analysis": analysis,
                        "comparator": comparator,
                        "resample": resample,
                        "valid": 1,
                        "r05_auroc": r05_value,
                        "comparator_auroc": comparator_value,
                        "r05_minus_comparator_auroc": r05_value - comparator_value,
                    }
                )
        model_values = np.asarray(model_values, dtype=np.float64)
        required = len(samples) if analysis == "overall" else minimum_valid_subgroup
        if len(model_values) < required:
            raise ValueError(
                f"R05 analysis {analysis} has only {len(model_values)} valid bootstrap replicates"
            )
        comparison_summary = {}
        for comparator, values in comparator_values.items():
            values = np.asarray(values, dtype=np.float64)
            differences = model_values - values
            comparison_summary[comparator] = {
                "comparator_auroc": common.failure_auroc(
                    comparator_scores[analysis][comparator][mask],
                    success[mask],
                ),
                "r05_minus_comparator_auroc": float(
                    common.failure_auroc(r05_scores[mask], success[mask])
                    - common.failure_auroc(
                        comparator_scores[analysis][comparator][mask],
                        success[mask],
                    )
                ),
                "paired_bootstrap_mean_difference": float(differences.mean()),
                "paired_ci95_lower": float(np.percentile(differences, 2.5)),
                "paired_ci95_upper": float(np.percentile(differences, 97.5)),
            }
        summaries[analysis] = {
            "repetitions_requested": len(samples),
            "valid_repetitions": len(model_values),
            "invalid_repetitions": invalid,
            "r05_ci95_lower": float(np.percentile(model_values, 2.5)),
            "r05_ci95_upper": float(np.percentile(model_values, 97.5)),
            "comparisons": comparison_summary,
        }
    return summaries, rows


def analysis_summaries(
    r05_scores,
    baseline,
    success,
    analyses,
    bootstrap,
):
    output = {}
    for name, mask in analyses.items():
        output[name] = {
            "cases": int(mask.sum()),
            "successes": int(success[mask].sum()),
            "failures": int((~success[mask]).sum()),
            "failure_detection_auroc": common.failure_auroc(
                r05_scores[mask], success[mask]
            ),
            "bootstrap": bootstrap[name],
            "calibration": reliability_metrics(
                r05_scores[mask], baseline[mask], success[mask]
            ),
        }
    return output


def verify_comparator(name, observed, expected, tolerance):
    difference = abs(float(observed) - float(expected))
    if difference > tolerance:
        raise ValueError(f"{name} comparator reproduction mismatch")
    return difference


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R05 is not in its registered pre-run state")
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    registry_item = next(
        item for item in registry["experiments"] if item["id"] == "r05_category_aware"
    )
    if registry_item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R05 registry is not in its registered pre-run state")

    source_checks = (
        (config["amendment_08_path"], "amendment_08_sha256", "Amendment 08"),
        ("audit/preregistration.md", "parent_preregistration_sha256", "preregistration"),
        ("audit/preregistration_amendment_03.md", "stage4_parent_amendment_sha256", "Stage 4 amendment"),
        (config["r02_result_path"], "r02_result_sha256", "R02 result"),
        (config["r02_manifest_path"], "r02_manifest_sha256", "R02 manifest"),
        (config["r02_oof_path"], "r02_oof_sha256", "R02 OOF table"),
        (config["r04_result_path"], "r04_result_sha256", "R04 result"),
        (config["r04_manifest_path"], "r04_manifest_sha256", "R04 manifest"),
        (config["r04_oof_path"], "r04_oof_sha256", "R04 OOF table"),
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
    verify_hash(
        r02_config_path,
        manifest_hash(r02_config_path, r02_manifest, "R02"),
        "R02 config",
    )
    r02_config = calibration.read_json(r02_config_path)
    r01_config_path = ROOT / "experiments/r01_ablation/config.json"
    verify_hash(r01_config_path, r02_config["r01_config_sha256"], "R01 config")
    r01_config = calibration.read_json(r01_config_path)
    r01_result_path = ROOT / r02_config["r01_result_path"]
    verify_hash(r01_result_path, r02_config["r01_result_sha256"], "R01 result")
    r01_result = calibration.read_json(r01_result_path)

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
    if len(data["rows"]) != int(config["population"]["source_cases"]):
        raise ValueError("R05 source population count mismatch")

    all_features = list(r01_result["baseline"]["features"])
    numeric_features = list(config["numeric_features"])
    if numeric_features != r02_config["candidate_selection"]["retained_features"]:
        raise ValueError("R05 numeric feature list differs from R02")
    lookup = {feature: index for index, feature in enumerate(all_features)}
    if len(lookup) != len(all_features):
        raise ValueError("R01 feature list contains duplicates")
    selected = np.asarray([lookup[feature] for feature in numeric_features])
    all_matrix, base_contract = common.feature_matrix(
        data["probability"], tuple(all_features)
    )
    numeric_matrix = all_matrix[:, selected]
    if not np.isfinite(numeric_matrix).all():
        raise ValueError("R05 numeric feature matrix contains non-finite values")

    dice = common.dice_scores(data["probability"], data["targets"], threshold=0.5)
    success = dice >= 0.5
    r02_reproduced, _r02_baseline, _r02_models = common.cross_fit(
        numeric_matrix, success, data["assignment"]
    )

    r04_oof = calibration.read_csv(ROOT / config["r04_oof_path"])
    r02_oof = calibration.read_csv(ROOT / config["r02_oof_path"])
    if len(r04_oof) != 680 or len(r02_oof) != 680:
        raise ValueError("Comparator OOF row count mismatch")
    true_labels = np.empty(680, dtype=object)
    predicted_labels = np.empty(680, dtype=object)
    r02_committed = np.empty(680, dtype=np.float64)
    r04_a = np.empty(680, dtype=np.float64)
    r04_b = np.full(680, np.nan, dtype=np.float64)
    for index, (source, row02, row04) in enumerate(
        zip(data["rows"], r02_oof, r04_oof, strict=True)
    ):
        if source["path"] != row02["path"] or source["path"] != row04["path"]:
            raise ValueError(f"Comparator path-order mismatch at row {index}")
        if source["label"] != row04["true_label"]:
            raise ValueError(f"Comparator true-label mismatch at row {index}")
        true_labels[index] = row04["true_label"]
        predicted_labels[index] = row04["classifier_predicted_label"]
        r02_committed[index] = float(row02["reduced_oof_reliability"])
        if abs(r02_committed[index] - float(row04["r02_aggregate_oof_reliability"])) > 1e-15:
            raise ValueError(f"R02/R04 comparator disagreement at row {index}")
        r04_a[index] = float(row04["version_a_oof_reliability"])
        if row04["version_b_oof_reliability"]:
            r04_b[index] = float(row04["version_b_oof_reliability"])

    tolerance = float(config["fail_closed"]["comparator_reproduction_absolute_tolerance"])
    maximum_r02_difference = float(np.max(np.abs(r02_reproduced - r02_committed)))
    if maximum_r02_difference > tolerance:
        raise ValueError("R02 aggregate OOF score reproduction mismatch")

    one_hot, included = one_hot_categories(predicted_labels)
    excluded_indices = np.flatnonzero(~included)
    if len(excluded_indices) != int(config["population"]["predicted_route_counts"]["no_tumor_excluded"]):
        raise ValueError("R05 excluded-case count mismatch")
    excluded_index = int(excluded_indices[0])
    if data["rows"][excluded_index]["path"] != config["population"]["excluded_path"]:
        raise ValueError("R05 excluded path mismatch")
    if int(included.sum()) != int(config["population"]["included_cases"]):
        raise ValueError("R05 included population count mismatch")
    observed_route_counts = {
        category: int(np.sum(predicted_labels == category)) for category in CATEGORIES
    }
    for category in CATEGORIES:
        if observed_route_counts[category] != int(
            config["population"]["predicted_route_counts"][category]
        ):
            raise ValueError(f"R05 {category} route count mismatch")

    feature_names = numeric_features + list(config["category_features"])
    feature_matrix = np.column_stack((numeric_matrix, one_hot))
    if feature_matrix.shape != (680, 13) or not np.isfinite(feature_matrix).all():
        raise ValueError("R05 feature matrix is invalid")
    r05_scores, r05_baseline, models, fold_support = cross_fit_included(
        feature_matrix, success, data["assignment"], included
    )
    coefficients = coefficient_summary(models, feature_names)

    analyses = make_analyses(included, true_labels, predicted_labels)
    validate_analysis_support(analyses, success)
    comparator_scores = {
        "overall": {"r02_aggregate": r02_committed, "r04_version_a": r04_a, "r04_version_b": r04_b},
    }
    for category in CATEGORIES:
        comparator_scores[f"true_{category}"] = {
            "r02_aggregate": r02_committed,
            "r04_version_a": r04_a,
        }
        comparator_scores[f"predicted_{category}"] = {
            "r02_aggregate": r02_committed,
            "r04_version_b": r04_b,
        }

    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    bootstrap, bootstrap_rows = bootstrap_analyses(
        r05_scores,
        comparator_scores,
        success,
        analyses,
        samples,
        int(config["bootstrap"]["minimum_valid_subgroup_replicates"]),
    )
    summaries = analysis_summaries(
        r05_scores, r05_baseline, success, analyses, bootstrap
    )

    comparator_differences = {}
    comparator_differences["r02_matched_overall"] = verify_comparator(
        "R02 matched overall",
        summaries["overall"]["bootstrap"]["comparisons"]["r02_aggregate"]["comparator_auroc"],
        config["comparators"]["r02_matched_overall"],
        tolerance,
    )
    comparator_differences["r04_matched_overall_version_a"] = verify_comparator(
        "R04 matched overall Version A",
        summaries["overall"]["bootstrap"]["comparisons"]["r04_version_a"]["comparator_auroc"],
        config["comparators"]["r04_matched_overall_version_a"],
        tolerance,
    )
    comparator_differences["r04_matched_overall_version_b"] = verify_comparator(
        "R04 matched overall Version B",
        summaries["overall"]["bootstrap"]["comparisons"]["r04_version_b"]["comparator_auroc"],
        config["comparators"]["r04_matched_overall_version_b"],
        tolerance,
    )
    for category in CATEGORIES:
        for prefix, config_key, comparator in (
            ("r02_true", "r02_matched_true_category", "r02_aggregate"),
            ("r04_true", "r04_matched_true_version_a", "r04_version_a"),
        ):
            comparator_differences[f"{prefix}_{category}"] = verify_comparator(
                f"{prefix} {category}",
                summaries[f"true_{category}"]["bootstrap"]["comparisons"][comparator]["comparator_auroc"],
                config["comparators"][config_key][category],
                tolerance,
            )
        for prefix, config_key, comparator in (
            ("r02_predicted", "r02_predicted_category", "r02_aggregate"),
            ("r04_predicted", "r04_predicted_version_b", "r04_version_b"),
        ):
            comparator_differences[f"{prefix}_{category}"] = verify_comparator(
                f"{prefix} {category}",
                summaries[f"predicted_{category}"]["bootstrap"]["comparisons"][comparator]["comparator_auroc"],
                config["comparators"][config_key][category],
                tolerance,
            )

    predicted_glioma_auroc = summaries["predicted_glioma"]["failure_detection_auroc"]
    minimum_auroc = float(config["gate"]["minimum_auroc"])
    passed = gate_passes(predicted_glioma_auroc, minimum_auroc)
    decision = "PASS" if passed else "FAIL"

    feature_contract = {
        **base_contract,
        "numeric_feature_count": len(numeric_features),
        "category_indicator_count": 3,
        "total_feature_count": len(feature_names),
        "feature_names": feature_names,
        "all_included_one_hot_rows_sum_to_one": True,
        "category_source": "frozen classifier discrete predicted label",
        "true_category_model_input": False,
        "classifier_probability_model_input": False,
        "reference_mask_feature_input": False,
        "excluded_no_tumor_case_received_score": False,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    oof_rows = []
    feature_rows = []
    for index, row in enumerate(data["rows"]):
        oof_rows.append(
            {
                "path": row["path"],
                "true_label": true_labels[index],
                "classifier_predicted_label": predicted_labels[index],
                "similarity_group": row["similarity_group"],
                "fold": int(data["assignment"][index]) + 1,
                "included": int(included[index]),
                "calibrated_dice": float(dice[index]),
                "success_dice_at_least_0_50": int(success[index]),
                "r05_oof_reliability": float(r05_scores[index]) if included[index] else None,
                "r02_aggregate_oof_reliability": float(r02_committed[index]),
                "r04_version_a_oof_reliability": float(r04_a[index]),
                "r04_version_b_oof_reliability": float(r04_b[index]) if included[index] else None,
            }
        )
        if included[index]:
            feature_rows.append(
                {
                    "path": row["path"],
                    **{
                        feature: float(feature_matrix[index, feature_index])
                        for feature_index, feature in enumerate(feature_names)
                    },
                }
            )
    metric_rows = []
    for analysis, item in summaries.items():
        metric_rows.append(
            {
                "analysis": analysis,
                "cases": item["cases"],
                "successes": item["successes"],
                "failures": item["failures"],
                "failure_detection_auroc": item["failure_detection_auroc"],
                "bootstrap_ci95_lower": item["bootstrap"]["r05_ci95_lower"],
                "bootstrap_ci95_upper": item["bootstrap"]["r05_ci95_upper"],
                "valid_bootstrap_repetitions": item["bootstrap"]["valid_repetitions"],
                "ece": item["calibration"]["ece"],
                "mce": item["calibration"]["mce"],
                "brier": item["calibration"]["brier"],
                "prevalence_baseline_brier": item["calibration"]["prevalence_baseline_brier"],
                "brier_skill": item["calibration"]["brier_skill"],
            }
        )
    calibration.write_csv(ARTIFACTS / "oof_scores_and_comparators.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "category_aware_features.csv", feature_rows)
    calibration.write_csv(ARTIFACTS / "analysis_metrics.csv", metric_rows)
    calibration.write_csv(ARTIFACTS / "bootstrap_comparisons.csv", bootstrap_rows)
    calibration.write_csv(ARTIFACTS / "fold_coefficients.csv", coefficients)
    calibration.write_csv(ARTIFACTS / "fold_support.csv", fold_support)
    calibration.write_csv(
        ARTIFACTS / "excluded_case.csv",
        [
            {
                "path": data["rows"][excluded_index]["path"],
                "true_label": true_labels[excluded_index],
                "classifier_predicted_label": predicted_labels[excluded_index],
                "reason": "registered classifier-predicted no_tumor exclusion",
                "received_r05_score": 0,
            }
        ],
    )

    result = {
        "schema_version": 1,
        "experiment_id": "r05_category_aware",
        "status": decision,
        "decision": decision,
        "scope": "Calibrated-E08 internal-validation reliability analysis; not clinical validation",
        "population": {
            "source_cases": 680,
            "included_cases": int(included.sum()),
            "excluded_cases": int((~included).sum()),
            "included_successes": int(success[included].sum()),
            "included_failures": int((~success[included]).sum()),
            "excluded_path": data["rows"][excluded_index]["path"],
            "predicted_route_counts": observed_route_counts,
        },
        "analyses": summaries,
        "published_comparators": {
            "r02_true_category": config["comparators"]["r02_published_true_category"],
            "r04_version_a": config["comparators"]["r04_published_version_a"],
        },
        "coefficients": coefficients,
        "gate": {
            **config["gate"],
            "observed_predicted_glioma_auroc": predicted_glioma_auroc,
            "passed": passed,
            "status": decision,
        },
        "feature_contract": feature_contract,
        "verification": {
            "source_hashes": verified_hashes,
            "r02_maximum_per_case_oof_score_difference": maximum_r02_difference,
            "comparator_reproduction_absolute_differences": comparator_differences,
            "comparator_reproduction_tolerance": tolerance,
            "all_comparators_reproduced": True,
            "c03_changed_pixels": data["changed_pixels"],
            "c03_changed_images": data["changed_images"],
            "source_population_cases": len(success),
            "included_population_cases": int(included.sum()),
            "excluded_no_tumor_cases": int((~included).sum()),
            "fold_count": len(set(data["assignment"])),
            "similarity_group_count": len(set(groups)),
            "similarity_group_cross_fold_leakage": False,
            "every_training_partition_has_both_outcomes": True,
            "one_hot_exclusivity_verified": True,
            "true_category_model_leakage": False,
            "reference_mask_feature_leakage": False,
            "classifier_probability_feature_leakage": False,
            "reported_scores_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "neural_model_retrained": False,
            "reliability_estimator_refit": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "R05 uses internal BRISC validation data and does not establish external or clinical generalization.",
            "The same development population informed prior feature selection and the registered R05 comparison.",
            "The single classifier-predicted no-tumor case has no R05 operational score.",
            "Subgroup estimates, especially meningioma with few failures, remain statistically unstable.",
            "One-hot and prediction-feature coefficients are descriptive associations among correlated inputs, not causal effects.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    config["status"] = "COMPLETE"
    calibration.write_json(config_path, config)
    registry_item["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)

    gl = summaries["predicted_glioma"]
    pi = summaries["predicted_pituitary"]
    (EXPERIMENT / "decision.md").write_text(
        f"""# Category-aware calibrated-E08 reliability decision

**Decision:** {decision}

The registered gate required classifier-predicted-glioma failure-detection AUROC at least `{minimum_auroc:.2f}`. R05 achieved `{predicted_glioma_auroc:.6f}` with 95% similarity-group bootstrap interval `[{gl['bootstrap']['r05_ci95_lower']:.6f}, {gl['bootstrap']['r05_ci95_upper']:.6f}]`; therefore R05 **{decision.lower()}ed** its unchanged gate.

Overall AUROC on the 679 included cases was `{summaries['overall']['failure_detection_auroc']:.6f}`. Predicted-pituitary AUROC was `{pi['failure_detection_auroc']:.6f}`. Full true-category, predicted-category, calibration, Brier, coefficient, and paired comparator results are stored in `result.json` and the artifact tables.

The model used the ten R02 prediction-derived features plus three frozen-classifier-predicted-category indicators. True category, classifier probabilities, and reference-mask values did not enter the feature matrix. The registered predicted-no-tumor case was excluded and received no R05 score. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
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
            "experiment_id": "r05_category_aware",
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
                "analyses": summaries,
                "gate": result["gate"],
                "coefficients": coefficients,
                "feature_contract": feature_contract,
                "verification": result["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

