"""Run and close R01 calibrated-E08 feature ablation."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stage4_reliability_common as common
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r01_ablation"
ARTIFACTS = EXPERIMENT / "artifacts"


def reliability_metrics(reliability, baseline_reliability, success):
    ece, mce, bins = calibration.equal_frequency_binary(
        reliability, success.astype(np.float64), bins=15
    )
    brier = float(np.mean(np.square(reliability - success)))
    baseline_brier = float(np.mean(np.square(baseline_reliability - success)))
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
    rows = []
    for index, name in enumerate(feature_names):
        values = matrix[:, index]
        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
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
                "all_folds_same_nonzero_sign": max(positive, negative) == 5,
                "stable_sign": "positive" if positive == 5 else "negative" if negative == 5 else "mixed",
            }
        )
    return rows


def auc_many(score_matrix, failure):
    """Compute row-wise AUROC using average ranks, including tied duplicates."""
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


def permutation_scores(features, success, assignment, models, feature_index, repeats):
    output = np.empty((repeats, len(success)), dtype=np.float64)
    for repetition in range(repeats):
        generator = np.random.default_rng(42 + 1000 * feature_index + repetition)
        for fold, model in enumerate(models):
            evaluation = np.flatnonzero(assignment == fold)
            permuted = features[evaluation].copy()
            permuted[:, feature_index] = permuted[
                generator.permutation(len(evaluation)), feature_index
            ]
            output[repetition, evaluation] = model.predict_proba(permuted)[:, 1]
    if not np.isfinite(output).all():
        raise ValueError("Permutation scoring produced non-finite values")
    return output


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


def render_importance(path, rows, key, title):
    ordered = sorted(rows, key=lambda row: row[key])
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh([row["feature"] for row in ordered], [row[key] for row in ordered])
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("AUROC decrease")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R01 is not in its registered pre-run state")
    if (
        calibration.sha256(ROOT / "audit/preregistration_amendment_03.md")
        != config["amendment_03_sha256"]
    ):
        raise ValueError("Amendment 03 hash mismatch")
    if (
        calibration.sha256(ROOT / "audit/stage3_calibration_report.json")
        != config["stage3_report_sha256"]
    ):
        raise ValueError("Stage 3 report hash mismatch")
    feature_script = ROOT / "scripts/fit_segmenter_case_reliability.py"
    if calibration.sha256(feature_script) != config["feature_reference_script_sha256"]:
        raise ValueError("Feature reference implementation hash mismatch")
    data = common.load_calibrated_e08(config)
    feature_names = tuple(config["features"])
    features, feature_contract = common.feature_matrix(
        data["probability"], feature_names
    )
    dice = common.dice_scores(data["probability"], data["targets"])
    success = dice >= 0.50
    failure = ~success
    if not success.any() or not failure.any():
        raise ValueError("Calibrated-E08 reliability target lacks both classes")
    reliability, prevalence_baseline, models = common.cross_fit(
        features, success, data["assignment"]
    )
    baseline_auroc = common.failure_auroc(reliability, success)
    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    bootstrap_samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    baseline_bootstrap = common.bootstrap_auroc(
        reliability, success, bootstrap_samples
    )
    calibration_metrics = reliability_metrics(
        reliability, prevalence_baseline, success
    )
    coefficients = coefficient_summary(models, feature_names)
    coefficient_by_name = {row["feature"]: row for row in coefficients}
    ablation_rows = []
    for removed_index, removed_feature in enumerate(feature_names):
        retained = [index for index in range(len(feature_names)) if index != removed_index]
        ablated_reliability, _baseline, _models = common.cross_fit(
            features, success, data["assignment"], retained
        )
        ablated_auroc = common.failure_auroc(ablated_reliability, success)
        ablation_rows.append(
            {
                "feature": removed_feature,
                "baseline_failure_detection_auroc": baseline_auroc,
                "ablated_failure_detection_auroc": ablated_auroc,
                "baseline_minus_ablated_auroc": baseline_auroc - ablated_auroc,
                "rank": 0,
            }
        )
    ablation_rows.sort(
        key=lambda row: (-row["baseline_minus_ablated_auroc"], row["feature"])
    )
    for rank, row in enumerate(ablation_rows, start=1):
        row["rank"] = rank
    repeats = int(config["permutation_importance"]["repetitions_per_feature"])
    permutation_rows = []
    for feature_index, feature_name in enumerate(feature_names):
        permuted_reliability = permutation_scores(
            features,
            success,
            data["assignment"],
            models,
            feature_index,
            repeats,
        )
        permuted_risk = 1.0 - permuted_reliability
        point_permuted_aurocs = auc_many(permuted_risk, failure)
        bootstrap_importance = np.empty(len(bootstrap_samples), dtype=np.float64)
        for bootstrap_index, members in enumerate(bootstrap_samples):
            permuted_aurocs = auc_many(
                permuted_risk[:, members], failure[members]
            )
            bootstrap_importance[bootstrap_index] = baseline_bootstrap[
                bootstrap_index
            ] - float(permuted_aurocs.mean())
        permutation_rows.append(
            {
                "feature": feature_name,
                "repetitions": repeats,
                "baseline_failure_detection_auroc": baseline_auroc,
                "mean_permuted_failure_detection_auroc": float(
                    point_permuted_aurocs.mean()
                ),
                "mean_auroc_decrease": baseline_auroc
                - float(point_permuted_aurocs.mean()),
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
    true_labels = np.asarray([row["label"] for row in data["rows"]])
    classifier_path = ROOT / config["classifier_predictions_path"]
    if calibration.sha256(classifier_path) != config["classifier_predictions_sha256"]:
        raise ValueError("Classifier predictions hash mismatch")
    classifier_source_rows = calibration.read_csv(classifier_path)
    classifier_rows = {row["path"]: row for row in classifier_source_rows}
    if len(classifier_rows) != len(classifier_source_rows):
        raise ValueError("Classifier predictions contain duplicate paths")
    for row in data["rows"]:
        classifier_row = classifier_rows.get(row["path"])
        if classifier_row is None:
            raise ValueError(f"Classifier prediction is missing: {row['path']}")
        if classifier_row["label"] != row["label"]:
            raise ValueError(f"Classifier true-label mismatch: {row['path']}")
    predicted_labels = np.asarray(
        [classifier_rows[row["path"]]["predicted"] for row in data["rows"]]
    )
    top_risk_count = int(np.ceil(0.20 * len(success)))
    ranked = sorted(
        range(len(success)),
        key=lambda index: (
            -(1.0 - reliability[index]),
            calibration.normalized_path(data["rows"][index]["path"]),
        ),
    )
    zero_dice = dice == 0
    top_indices = np.asarray(ranked[:top_risk_count])
    zero_dice_capture = int(zero_dice[top_indices].sum())
    ablation_by_name = {row["feature"]: row for row in ablation_rows}
    r02_candidates = []
    for feature_name in feature_names:
        ablation_qualifies = (
            ablation_by_name[feature_name]["baseline_minus_ablated_auroc"] >= 0.005
        )
        sign_qualifies = coefficient_by_name[feature_name][
            "all_folds_same_nonzero_sign"
        ]
        if ablation_qualifies or sign_qualifies:
            r02_candidates.append(
                {
                    "feature": feature_name,
                    "ablation_delta_at_least_0_005": ablation_qualifies,
                    "all_folds_same_nonzero_sign": sign_qualifies,
                }
            )
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
                "zero_dice": int(zero_dice[index]),
                "oof_reliability": float(reliability[index]),
                "oof_failure_risk": float(1.0 - reliability[index]),
                **{
                    feature_name: float(features[index, feature_index])
                    for feature_index, feature_name in enumerate(feature_names)
                },
            }
        )
    calibration.write_csv(ARTIFACTS / "oof_scores_and_features.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "ablation_results.csv", ablation_rows)
    calibration.write_csv(
        ARTIFACTS / "permutation_importance.csv", permutation_rows
    )
    calibration.write_csv(ARTIFACTS / "fold_coefficients.csv", coefficients)
    render_importance(
        ARTIFACTS / "ablation_importance.png",
        ablation_rows,
        "baseline_minus_ablated_auroc",
        "R01 leave-one-feature-out AUROC decrease",
    )
    render_importance(
        ARTIFACTS / "permutation_importance.png",
        permutation_rows,
        "mean_auroc_decrease",
        "R01 held-out permutation AUROC decrease",
    )
    result = {
        "schema_version": 1,
        "experiment_id": "r01_ablation",
        "status": "PASS",
        "decision": "DESCRIPTIVE_COMPLETE",
        "scope": "Calibrated-E08 internal-validation reliability analysis; not clinical validation",
        "population": {
            "cases": len(success),
            "successes": int(success.sum()),
            "failures": int(failure.sum()),
            "zero_dice_cases": int(zero_dice.sum()),
        },
        "baseline": {
            "features": list(feature_names),
            "failure_detection_auroc": baseline_auroc,
            "bootstrap": {
                "repetitions": len(bootstrap_samples),
                "mean": float(baseline_bootstrap.mean()),
                "ci95_lower": float(np.percentile(baseline_bootstrap, 2.5)),
                "ci95_upper": float(np.percentile(baseline_bootstrap, 97.5)),
            },
            "calibration": calibration_metrics,
            "zero_dice_capture_top_20_percent_risk": {
                "top_risk_cases": top_risk_count,
                "captured": zero_dice_capture,
                "total": int(zero_dice.sum()),
                "fraction": (
                    zero_dice_capture / int(zero_dice.sum())
                    if zero_dice.any()
                    else None
                ),
            },
            "true_category_aurocs_descriptive": subgroup_aurocs(
                reliability, success, true_labels
            ),
            "classifier_predicted_category_aurocs_descriptive": subgroup_aurocs(
                reliability, success, predicted_labels
            ),
            "fold_coefficients": coefficients,
        },
        "ablation": {
            "definition": "baseline AUROC minus leave-one-feature-out AUROC",
            "ranked_results": ablation_rows,
        },
        "permutation_importance": {
            "definition": "baseline AUROC minus mean AUROC after within-held-out-fold feature permutation",
            "repetitions_per_feature": repeats,
            "ranked_results": permutation_rows,
        },
        "r02_candidate_screen": {
            "status": "PROVISIONAL_INPUT_TO_R02_REGISTRATION",
            "rule": "ablation delta >=0.005 OR identical non-zero coefficient sign in all five folds",
            "features": r02_candidates,
            "note": "R02 feature selection is not final until its separate preregistration is reviewed and committed.",
        },
        "feature_contract": feature_contract,
        "verification": {
            "c03_mapping_sha256": calibration.sha256(
                ROOT / config["c03_mappings_path"]
            ),
            "c03_changed_pixels": data["changed_pixels"],
            "c03_changed_images": data["changed_images"],
            "fold_assignment_sha256": calibration.sha256(calibration.FOLDS_PATH),
            "fold_count": len(set(data["assignment"])),
            "fold_case_counts": {
                str(fold + 1): int(np.sum(data["assignment"] == fold))
                for fold in range(5)
            },
            "similarity_group_count": len(set(groups)),
            "similarity_group_cross_fold_leakage": False,
            "feature_reference_script_sha256": calibration.sha256(feature_script),
            "features_receive_probability_and_threshold_only": True,
            "reference_mask_feature_leakage": False,
            "classifier_probability_feature_leakage": False,
            "raw_image_feature_leakage": False,
            "reported_scores_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "neural_model_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "gate": {
            "type": "DESCRIPTIVE",
            "status": "NOT_APPLICABLE",
            "selectable": False,
        },
        "limitations": [
            "R01 uses internal BRISC validation data and does not establish external generalization.",
            "Ablation and permutation importance describe predictive association, not causality.",
            "Correlated features can share or mask importance.",
            "True-category subgroup AUROCs are descriptive and unavailable as routing inputs.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    config["status"] = "COMPLETE"
    calibration.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    next(
        item for item in registry["experiments"] if item["id"] == "r01_ablation"
    )["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)
    candidate_names = [item["feature"] for item in r02_candidates]
    (EXPERIMENT / "decision.md").write_text(
        f"""# Calibrated-E08 reliability feature ablation decision

**Decision:** DESCRIPTIVE COMPLETE

R01 has no performance gate and cannot be selected. The full fourteen-feature calibrated-E08 baseline achieved failure-detection AUROC `{baseline_auroc:.6f}` with 95% similarity-group bootstrap CI `[{np.percentile(baseline_bootstrap, 2.5):.6f}, {np.percentile(baseline_bootstrap, 97.5):.6f}]`.

Provisional R02 candidates under the user-specified screen (ablation decrease at least `0.005` or identical non-zero coefficient sign across all five folds) are: {', '.join(candidate_names) if candidate_names else 'none'}. This screen becomes binding only when R02 is separately registered before fitting.

All features receive only a C03-calibrated probability map and the unchanged `0.50` threshold. Reference masks created targets and evaluation metrics only. No classifier probability, true category, path, metadata, or raw image enters a feature. No neural model was retrained and no sealed-test image was accessed.
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
            "status": "PASS",
            "experiment_id": "r01_ablation",
            "registered_decision": "DESCRIPTIVE_COMPLETE",
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
                "status": result["decision"],
                "cases": result["population"],
                "baseline_auroc": baseline_auroc,
                "baseline_ci95": [
                    float(np.percentile(baseline_bootstrap, 2.5)),
                    float(np.percentile(baseline_bootstrap, 97.5)),
                ],
                "ece": calibration_metrics["ece"],
                "brier": calibration_metrics["brier"],
                "brier_skill": calibration_metrics["brier_skill"],
                "zero_dice_capture": result["baseline"][
                    "zero_dice_capture_top_20_percent_risk"
                ],
                "top_ablation": ablation_rows[:5],
                "top_permutation": permutation_rows[:5],
                "r02_candidate_features": candidate_names,
                "verification": result["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
