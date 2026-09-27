"""Run and close R06 nonlinear calibrated-E08 reliability comparison."""

import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import stage4_reliability_common as common
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r06_nonlinear_comparison"
ARTIFACTS = EXPERIMENT / "artifacts"
CATEGORIES = ("glioma", "meningioma", "pituitary")
NONLINEAR_MODELS = ("xgboost", "mlp", "random_forest")


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


def model_gate(auroc, minimum):
    return float(auroc) >= float(minimum)


def make_nonlinear_model(name, config):
    parameters = dict(config["models"][name])
    parameters.pop("class")
    parameters.pop("scale_features")
    if name == "xgboost":
        return XGBClassifier(**parameters)
    if name == "mlp":
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", MLPClassifier(**parameters)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(**parameters)
    raise ValueError(f"Unknown R06 model: {name}")


def cross_fit_model(name, features, success, assignment, config):
    scores = np.empty(len(success), dtype=np.float64)
    baseline = np.empty(len(success), dtype=np.float64)
    models = []
    diagnostics = []
    for fold in range(5):
        fitting = assignment != fold
        evaluation = assignment == fold
        if len(np.unique(success[fitting])) != 2 or len(np.unique(success[evaluation])) != 2:
            raise ValueError(f"R06 target lacks both outcomes in fold {fold + 1}")
        model = make_nonlinear_model(name, config)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(features[fitting], success[fitting])
        convergence = [
            str(item.message)
            for item in captured
            if issubclass(item.category, ConvergenceWarning)
        ]
        other_warnings = [
            f"{item.category.__name__}: {item.message}"
            for item in captured
            if not issubclass(item.category, ConvergenceWarning)
        ]
        probability = np.asarray(
            model.predict_proba(features[evaluation])[:, 1], dtype=np.float64
        )
        if not np.isfinite(probability).all() or np.any(
            (probability < 0) | (probability > 1)
        ):
            raise ValueError(f"R06 {name} produced invalid probabilities")
        scores[evaluation] = probability
        baseline[evaluation] = float(success[fitting].mean())
        estimator = model.named_steps["estimator"] if name == "mlp" else model
        diagnostics.append(
            {
                "model": name,
                "fold": fold + 1,
                "training_cases": int(fitting.sum()),
                "training_successes": int(success[fitting].sum()),
                "training_failures": int((~success[fitting]).sum()),
                "evaluation_cases": int(evaluation.sum()),
                "evaluation_successes": int(success[evaluation].sum()),
                "evaluation_failures": int((~success[evaluation]).sum()),
                "training_prevalence": float(success[fitting].mean()),
                "convergence_warning_count": len(convergence),
                "convergence_warnings": " | ".join(convergence),
                "other_warning_count": len(other_warnings),
                "other_warnings": " | ".join(other_warnings),
                "iterations": (
                    int(estimator.n_iter_)
                    if hasattr(estimator, "n_iter_")
                    else int(config["models"][name].get("n_estimators", 0))
                ),
            }
        )
        models.append(model)
    if not np.isfinite(scores).all() or not np.isfinite(baseline).all():
        raise ValueError(f"R06 {name} OOF scoring is incomplete")
    return scores, baseline, models, diagnostics


def reliability_metrics(reliability, baseline, success):
    ece, mce, bins = calibration.equal_frequency_binary(
        reliability, success.astype(np.float64), bins=15
    )
    brier = float(np.mean(np.square(reliability - success)))
    baseline_brier = float(np.mean(np.square(baseline - success)))
    if baseline_brier <= 0:
        raise ValueError("R06 prevalence baseline Brier must be positive")
    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier),
        "bins": bins,
    }


def top_risk_capture(reliability, dice, rows):
    count = int(np.ceil(0.20 * len(reliability)))
    ranked = sorted(
        range(len(reliability)),
        key=lambda index: (
            -(1.0 - reliability[index]),
            calibration.normalized_path(rows[index]["path"]),
        ),
    )
    zero = dice == 0
    captured = int(zero[np.asarray(ranked[:count])].sum())
    total = int(zero.sum())
    return {
        "top_risk_cases": count,
        "captured": captured,
        "total": total,
        "fraction": captured / total if total else None,
    }


def analysis_masks(true_labels, predicted_labels):
    masks = {"overall": np.ones(len(true_labels), dtype=bool)}
    for category in CATEGORIES:
        masks[f"true_{category}"] = true_labels == category
        masks[f"predicted_{category}"] = predicted_labels == category
    return masks


def bootstrap_all(scores, success, masks, samples, minimum_valid_subgroup):
    summaries = {}
    rows = []
    for analysis, mask in masks.items():
        values = {name: [] for name in scores}
        invalid = 0
        for resample, members in enumerate(samples, start=1):
            selected = members[mask[members]]
            if len(selected) == 0 or len(np.unique(success[selected])) != 2:
                invalid += 1
                rows.append(
                    {
                        "analysis": analysis,
                        "resample": resample,
                        "valid": 0,
                        **{f"{name}_auroc": None for name in scores},
                    }
                )
                continue
            row = {"analysis": analysis, "resample": resample, "valid": 1}
            for name, reliability in scores.items():
                value = common.failure_auroc(reliability[selected], success[selected])
                values[name].append(value)
                row[f"{name}_auroc"] = value
            rows.append(row)
        required = len(samples) if analysis == "overall" else minimum_valid_subgroup
        if any(len(model_values) < required for model_values in values.values()):
            raise ValueError(f"R06 {analysis} bootstrap lacks valid replicates")
        arrays = {name: np.asarray(model_values) for name, model_values in values.items()}
        summaries[analysis] = {
            "repetitions_requested": len(samples),
            "valid_repetitions": len(next(iter(arrays.values()))),
            "invalid_repetitions": invalid,
            "models": {
                name: {
                    "ci95_lower": float(np.percentile(array, 2.5)),
                    "ci95_upper": float(np.percentile(array, 97.5)),
                }
                for name, array in arrays.items()
            },
            "paired_vs_r02": {
                name: {
                    "mean_difference": float((array - arrays["r02_logistic"]).mean()),
                    "ci95_lower": float(
                        np.percentile(array - arrays["r02_logistic"], 2.5)
                    ),
                    "ci95_upper": float(
                        np.percentile(array - arrays["r02_logistic"], 97.5)
                    ),
                }
                for name, array in arrays.items()
                if name != "r02_logistic"
            },
        }
    return summaries, rows


def subgroup_summaries(scores, success, masks, bootstrap):
    output = {name: {} for name in scores}
    for model, reliability in scores.items():
        for analysis, mask in masks.items():
            output[model][analysis] = {
                "cases": int(mask.sum()),
                "successes": int(success[mask].sum()),
                "failures": int((~success[mask]).sum()),
                "failure_detection_auroc": common.failure_auroc(
                    reliability[mask], success[mask]
                ),
                "bootstrap": bootstrap[analysis]["models"][model],
            }
    return output


def native_importance(models, feature_names):
    rows = []
    xgb_matrix = []
    for fold, model in enumerate(models["xgboost"], start=1):
        gain = model.get_booster().get_score(importance_type="gain")
        raw = np.asarray(
            [float(gain.get(f"f{index}", 0.0)) for index in range(len(feature_names))]
        )
        normalized = raw / raw.sum() if raw.sum() > 0 else raw
        xgb_matrix.append(normalized)
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "model": "xgboost",
                    "importance_type": "normalized_native_gain",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(normalized[index]),
                    "raw_native_value": float(raw[index]),
                }
            )
    rf_matrix = []
    for fold, model in enumerate(models["random_forest"], start=1):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        rf_matrix.append(values)
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "model": "random_forest",
                    "importance_type": "native_impurity",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(values[index]),
                    "raw_native_value": float(values[index]),
                }
            )
    summary = []
    for model, matrix, importance_type in (
        ("xgboost", np.vstack(xgb_matrix), "normalized_native_gain"),
        ("random_forest", np.vstack(rf_matrix), "native_impurity"),
    ):
        for index, feature in enumerate(feature_names):
            values = matrix[:, index]
            summary.append(
                {
                    "model": model,
                    "importance_type": importance_type,
                    "feature": feature,
                    "mean_importance": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "minimum_fold_importance": float(values.min()),
                    "maximum_fold_importance": float(values.max()),
                }
            )
    return rows, summary


def permutation_importance(
    models,
    baseline_scores,
    features,
    success,
    assignment,
    feature_names,
    repetitions,
):
    rows = []
    summary = []
    for model_index, model_name in enumerate(NONLINEAR_MODELS):
        baseline_auroc = common.failure_auroc(baseline_scores[model_name], success)
        for feature_index, feature in enumerate(feature_names):
            importance_values = []
            for repetition in range(repetitions):
                generator = np.random.default_rng(
                    42 + 1000 * model_index + 100 * feature_index + repetition
                )
                permuted_scores = baseline_scores[model_name].copy()
                for fold in range(5):
                    evaluation = np.flatnonzero(assignment == fold)
                    transformed = features[evaluation].copy()
                    transformed[:, feature_index] = transformed[
                        generator.permutation(len(evaluation)), feature_index
                    ]
                    probability = models[model_name][fold].predict_proba(transformed)[:, 1]
                    if not np.isfinite(probability).all():
                        raise ValueError("Permutation inference produced non-finite values")
                    permuted_scores[evaluation] = probability
                permuted_auroc = common.failure_auroc(permuted_scores, success)
                importance = baseline_auroc - permuted_auroc
                importance_values.append(importance)
                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "repetition": repetition + 1,
                        "seed": 42
                        + 1000 * model_index
                        + 100 * feature_index
                        + repetition,
                        "baseline_auroc": baseline_auroc,
                        "permuted_auroc": permuted_auroc,
                        "importance": importance,
                    }
                )
            values = np.asarray(importance_values)
            summary.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "repetitions": repetitions,
                    "mean_importance": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "percentile_2_5": float(np.percentile(values, 2.5)),
                    "percentile_97_5": float(np.percentile(values, 97.5)),
                }
            )
    return rows, summary


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R06 is not in its registered pre-run state")
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    registry_item = next(
        item
        for item in registry["experiments"]
        if item["id"] == "r06_nonlinear_comparison"
    )
    if registry_item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R06 registry is not in its registered pre-run state")
    if xgboost.__version__ != config["xgboost_dependency"]["version"]:
        raise ValueError("XGBoost version mismatch")

    source_checks = (
        (config["amendment_09_path"], "amendment_09_sha256", "Amendment 09"),
        ("audit/preregistration.md", "parent_preregistration_sha256", "preregistration"),
        ("audit/preregistration_amendment_03.md", "stage4_parent_amendment_sha256", "Stage 4 amendment"),
        (config["r02_result_path"], "r02_result_sha256", "R02 result"),
        (config["r02_manifest_path"], "r02_manifest_sha256", "R02 manifest"),
        (config["r02_oof_path"], "r02_oof_sha256", "R02 OOF table"),
        (config["r05_result_path"], "r05_result_sha256", "R05 result"),
        (config["r05_manifest_path"], "r05_manifest_sha256", "R05 manifest"),
        (config["c03_mappings_path"], "c03_mappings_sha256", "C03 mappings"),
        (config["fold_assignment_path"], "fold_assignment_sha256", "fold assignment"),
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
    if len(data["rows"]) != int(config["population"]["cases"]):
        raise ValueError("R06 population count mismatch")

    all_features = list(r01_result["baseline"]["features"])
    selected_features = list(config["features"])
    if selected_features != r02_config["candidate_selection"]["retained_features"]:
        raise ValueError("R06 feature list differs from R02")
    lookup = {name: index for index, name in enumerate(all_features)}
    selected = np.asarray([lookup[name] for name in selected_features])
    all_matrix, feature_contract = common.feature_matrix(
        data["probability"], tuple(all_features)
    )
    features = all_matrix[:, selected]
    if features.shape != (680, 10) or not np.isfinite(features).all():
        raise ValueError("R06 feature matrix is invalid")

    dice = common.dice_scores(data["probability"], data["targets"], threshold=0.5)
    success = dice >= 0.5
    r02_scores, r02_baseline, _r02_models = common.cross_fit(
        features, success, data["assignment"]
    )
    r02_oof = calibration.read_csv(ROOT / config["r02_oof_path"])
    committed_r02 = np.empty(680, dtype=np.float64)
    for index, (source, row) in enumerate(zip(data["rows"], r02_oof, strict=True)):
        if source["path"] != row["path"]:
            raise ValueError(f"R02 OOF path-order mismatch at row {index}")
        committed_r02[index] = float(row["reduced_oof_reliability"])
    tolerance = float(config["fail_closed"]["r02_reproduction_absolute_tolerance"])
    maximum_r02_difference = float(np.max(np.abs(r02_scores - committed_r02)))
    if maximum_r02_difference > tolerance:
        raise ValueError("R02 per-case OOF reproduction mismatch")
    r02_auroc = common.failure_auroc(r02_scores, success)
    registered_r02 = float(config["comparators"]["r02_full_680_auroc"])
    if abs(r02_auroc - registered_r02) > tolerance:
        raise ValueError("R02 AUROC reproduction mismatch")

    classifier_path = ROOT / r02_config["classifier_predictions_path"]
    verify_hash(
        classifier_path,
        r02_config["classifier_predictions_sha256"],
        "classifier predictions",
    )
    classifier_source = calibration.read_csv(classifier_path)
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

    scores = {"r02_logistic": r02_scores}
    baselines = {"r02_logistic": r02_baseline}
    models = {}
    fold_diagnostics = []
    for name in NONLINEAR_MODELS:
        model_scores, model_baseline, fitted, diagnostics = cross_fit_model(
            name, features, success, data["assignment"], config
        )
        scores[name] = model_scores
        baselines[name] = model_baseline
        models[name] = fitted
        fold_diagnostics.extend(diagnostics)

    masks = analysis_masks(true_labels, predicted_labels)
    for analysis, mask in masks.items():
        if not mask.any() or len(np.unique(success[mask])) != 2:
            raise ValueError(f"R06 analysis {analysis} lacks both outcomes")
    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    bootstrap, bootstrap_rows = bootstrap_all(
        scores,
        success,
        masks,
        samples,
        int(config["bootstrap"]["minimum_valid_subgroup_replicates"]),
    )
    subgroups = subgroup_summaries(scores, success, masks, bootstrap)

    model_summaries = {}
    for name, reliability in scores.items():
        model_summaries[name] = {
            "failure_detection_auroc": subgroups[name]["overall"][
                "failure_detection_auroc"
            ],
            "bootstrap": subgroups[name]["overall"]["bootstrap"],
            "calibration": reliability_metrics(
                reliability, baselines[name], success
            ),
            "zero_dice_capture_top_20_percent_risk": top_risk_capture(
                reliability, dice, data["rows"]
            ),
            "true_category": {
                category: subgroups[name][f"true_{category}"]
                for category in CATEGORIES
            },
            "predicted_category": {
                category: subgroups[name][f"predicted_{category}"]
                for category in CATEGORIES
            },
        }
        if name != "r02_logistic":
            model_summaries[name]["paired_vs_r02"] = bootstrap["overall"][
                "paired_vs_r02"
            ][name]

    native_rows, native_summary = native_importance(models, selected_features)
    permutation_rows, permutation_summary = permutation_importance(
        models,
        scores,
        features,
        success,
        data["assignment"],
        selected_features,
        int(config["permutation_importance"]["repetitions_per_feature"]),
    )

    minimum_auroc = float(config["gate"]["minimum_auroc"])
    model_gates = {
        name: {
            "observed_auroc": model_summaries[name]["failure_detection_auroc"],
            "minimum_auroc": minimum_auroc,
            "operator": ">=",
            "passed": model_gate(
                model_summaries[name]["failure_detection_auroc"], minimum_auroc
            ),
        }
        for name in NONLINEAR_MODELS
    }
    passing = [name for name in NONLINEAR_MODELS if model_gates[name]["passed"]]
    if passing:
        order = {name: index for index, name in enumerate(NONLINEAR_MODELS)}
        winner = min(
            passing,
            key=lambda name: (
                -model_summaries[name]["failure_detection_auroc"],
                model_summaries[name]["calibration"]["ece"],
                model_summaries[name]["calibration"]["brier"],
                order[name],
            ),
        )
        decision = "PASS"
        retained_model = winner
    else:
        winner = None
        decision = "FAIL"
        retained_model = "r02_logistic"

    feature_contract = {
        **feature_contract,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "all_selected_features_inference_time_only": True,
        "true_category_model_input": False,
        "classifier_prediction_model_input": False,
        "classifier_probability_model_input": False,
        "reference_mask_feature_input": False,
        "raw_image_feature_input": False,
    }

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
                **{
                    f"{name}_oof_reliability": float(reliability[index])
                    for name, reliability in scores.items()
                },
            }
        )
    metric_rows = []
    subgroup_rows = []
    for name, summary in model_summaries.items():
        metric_rows.append(
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
                "zero_dice_captured": summary[
                    "zero_dice_capture_top_20_percent_risk"
                ]["captured"],
                "zero_dice_total": summary[
                    "zero_dice_capture_top_20_percent_risk"
                ]["total"],
                "gate_passed": model_gates.get(name, {}).get("passed"),
            }
        )
        for grouping in ("true_category", "predicted_category"):
            for category in CATEGORIES:
                item = summary[grouping][category]
                subgroup_rows.append(
                    {
                        "model": name,
                        "grouping": grouping,
                        "category": category,
                        "cases": item["cases"],
                        "successes": item["successes"],
                        "failures": item["failures"],
                        "failure_detection_auroc": item["failure_detection_auroc"],
                        "bootstrap_ci95_lower": item["bootstrap"]["ci95_lower"],
                        "bootstrap_ci95_upper": item["bootstrap"]["ci95_upper"],
                        "valid_bootstrap_repetitions": bootstrap[
                            f"{'true' if grouping == 'true_category' else 'predicted'}_{category}"
                        ]["valid_repetitions"],
                    }
                )
    comparison_rows = []
    for name in NONLINEAR_MODELS:
        comparison = model_summaries[name]["paired_vs_r02"]
        comparison_rows.append(
            {
                "model": name,
                "model_auroc": model_summaries[name]["failure_detection_auroc"],
                "r02_auroc": r02_auroc,
                "model_minus_r02_auroc": model_summaries[name][
                    "failure_detection_auroc"
                ]
                - r02_auroc,
                "paired_bootstrap_mean_difference": comparison["mean_difference"],
                "paired_ci95_lower": comparison["ci95_lower"],
                "paired_ci95_upper": comparison["ci95_upper"],
                "minimum_passing_auroc": minimum_auroc,
                "gate_passed": model_gates[name]["passed"],
            }
        )
    calibration.write_csv(ARTIFACTS / "oof_model_scores.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "model_metrics.csv", metric_rows)
    calibration.write_csv(ARTIFACTS / "subgroup_aurocs.csv", subgroup_rows)
    calibration.write_csv(ARTIFACTS / "bootstrap_aurocs.csv", bootstrap_rows)
    calibration.write_csv(ARTIFACTS / "paired_comparisons.csv", comparison_rows)
    calibration.write_csv(ARTIFACTS / "fold_diagnostics.csv", fold_diagnostics)
    calibration.write_csv(ARTIFACTS / "native_importance_folds.csv", native_rows)
    calibration.write_csv(ARTIFACTS / "native_importance_summary.csv", native_summary)
    calibration.write_csv(
        ARTIFACTS / "permutation_importance_repetitions.csv", permutation_rows
    )
    calibration.write_csv(
        ARTIFACTS / "permutation_importance_summary.csv", permutation_summary
    )

    environment_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = {
        "schema_version": 1,
        "experiment_id": "r06_nonlinear_comparison",
        "status": decision,
        "decision": decision,
        "scope": "Calibrated-E08 internal-validation reliability analysis; not clinical validation",
        "population": {
            "cases": len(success),
            "successes": int(success.sum()),
            "failures": int((~success).sum()),
            "zero_dice_cases": int((dice == 0).sum()),
        },
        "models": model_summaries,
        "gate": {
            **config["gate"],
            "models": model_gates,
            "passing_models": passing,
            "winner": winner,
            "status": decision,
        },
        "retained_operational_model": retained_model,
        "native_importance": native_summary,
        "permutation_importance": permutation_summary,
        "interpretability": {
            "r02_logistic": "Direct standardized coefficients with stable signs; retained if R06 fails.",
            "xgboost": "Native gain and held-out permutation importance describe associations but do not expose a single global linear rule or causal effects.",
            "mlp": "No native feature coefficients; held-out permutation importance is global and cannot fully explain interactions or individual predictions.",
            "random_forest": "Impurity and permutation importance are available, but the 100-tree ensemble lacks a compact coefficient-level decision rule.",
            "decision": (
                f"{winner} passed the performance gate; explanation remains less direct than R02."
                if winner
                else "No nonlinear model passed; retain R02's coefficient-level interpretability."
            ),
        },
        "feature_contract": feature_contract,
        "verification": {
            "source_hashes": verified_hashes,
            "xgboost_version": xgboost.__version__,
            "pyproject_sha256": calibration.sha256(ROOT / "pyproject.toml"),
            "uv_lock_sha256": calibration.sha256(ROOT / "uv.lock"),
            "environment_commit": environment_commit,
            "r02_maximum_per_case_oof_score_difference": maximum_r02_difference,
            "r02_auroc_absolute_difference": abs(r02_auroc - registered_r02),
            "r02_reproduction_tolerance": tolerance,
            "c03_changed_pixels": data["changed_pixels"],
            "c03_changed_images": data["changed_images"],
            "population_cases": len(success),
            "fold_count": len(set(data["assignment"])),
            "similarity_group_count": len(set(groups)),
            "similarity_group_cross_fold_leakage": False,
            "all_probabilities_finite_and_in_unit_interval": True,
            "all_selected_features_inference_time_only": True,
            "reference_mask_feature_leakage": False,
            "classifier_input_feature_leakage": False,
            "true_category_feature_leakage": False,
            "reported_scores_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "classifier_or_neural_segmenter_retrained": False,
            "reliability_estimators_fitted": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "R06 uses internal BRISC validation data and does not establish external or clinical generalization.",
            "The same development population informed R02 feature selection and R06 model comparison.",
            "Subgroup AUROCs with few failures, especially meningioma, remain unstable.",
            "Native and permutation feature importance describe predictive associations, not causal effects.",
            "The absolute 0.9750 gate was fixed by the user from a rounded matched-cohort comparator; exact change versus full-population R02 is reported separately.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    config["status"] = "COMPLETE"
    config["environment_commit"] = environment_commit
    config["installed_xgboost_version"] = xgboost.__version__
    calibration.write_json(config_path, config)
    registry_item["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)

    lines = [
        "# Nonlinear calibrated-E08 reliability decision",
        "",
        f"**Decision:** {decision}",
        "",
        f"The fixed gate was AUROC at least `{minimum_auroc:.4f}`. R02 reproduced at `{r02_auroc:.6f}`.",
        "",
    ]
    for name in NONLINEAR_MODELS:
        summary = model_summaries[name]
        lines.append(
            f"- **{name}:** AUROC `{summary['failure_detection_auroc']:.6f}` "
            f"(95% CI `{summary['bootstrap']['ci95_lower']:.6f}–{summary['bootstrap']['ci95_upper']:.6f}`), "
            f"change versus R02 `{summary['failure_detection_auroc'] - r02_auroc:+.6f}`, "
            f"gate `{'PASS' if model_gates[name]['passed'] else 'FAIL'}`."
        )
    lines.extend(
        [
            "",
            f"The retained operational reliability model is **{retained_model}**. "
            + result["interpretability"]["decision"],
            "",
            "All ten inputs derive only from the C03 OOF calibrated probability map and threshold-0.50 prediction. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.",
            "",
        ]
    )
    (EXPERIMENT / "decision.md").write_text("\n".join(lines), encoding="utf-8")

    included_files = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    calibration.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": decision,
            "experiment_id": "r06_nonlinear_comparison",
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
                "models": model_summaries,
                "gate": result["gate"],
                "retained_operational_model": retained_model,
                "native_importance": native_summary,
                "permutation_importance": permutation_summary,
                "feature_contract": feature_contract,
                "verification": result["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

