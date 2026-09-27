"""Run and close R09 alternative-fold sensitivity analysis."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import stage4_reliability_common as common
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r09_fold_sensitivity"
ARTIFACTS = EXPERIMENT / "artifacts"
AUDIT_REPORT = ROOT / "audit/fold_sensitivity.md"
CATEGORIES = ("glioma", "meningioma", "pituitary")


def verify_hash(path, expected, label):
    observed = calibration.sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return observed


def reliability_metrics(reliability, baseline, success):
    ece, mce, bins = calibration.equal_frequency_binary(
        reliability, success.astype(np.float64), bins=15
    )
    brier = float(np.mean(np.square(reliability - success)))
    baseline_brier = float(np.mean(np.square(baseline - success)))
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


def generate_assignment(success, groups, seed):
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=int(seed)
    )
    assignment = np.full(len(success), -1, dtype=np.int64)
    fold_support = []
    for fold, (fitting, evaluation) in enumerate(
        splitter.split(np.zeros((len(success), 1)), success.astype(int), groups)
    ):
        if np.any(assignment[evaluation] != -1):
            raise ValueError(f"Seed {seed} assigns a case twice")
        if len(np.unique(success[fitting])) != 2 or len(np.unique(success[evaluation])) != 2:
            raise ValueError(f"Seed {seed} fold {fold + 1} lacks both outcomes")
        assignment[evaluation] = fold
        train_groups = set(groups[fitting])
        evaluation_groups = set(groups[evaluation])
        if train_groups.intersection(evaluation_groups):
            raise ValueError(f"Seed {seed} fold {fold + 1} leaks a group")
        fold_support.append(
            {
                "seed": seed,
                "fold": fold + 1,
                "training_cases": len(fitting),
                "evaluation_cases": len(evaluation),
                "training_successes": int(success[fitting].sum()),
                "training_failures": int((~success[fitting]).sum()),
                "evaluation_successes": int(success[evaluation].sum()),
                "evaluation_failures": int((~success[evaluation]).sum()),
                "training_groups": len(train_groups),
                "evaluation_groups": len(evaluation_groups),
            }
        )
    if np.any(assignment < 0) or set(assignment) != set(range(5)):
        raise ValueError(f"Seed {seed} assignment is incomplete")
    group_fold_count = {}
    for group, fold in zip(groups, assignment, strict=True):
        group_fold_count.setdefault(group, set()).add(int(fold))
    if any(len(folds) != 1 for folds in group_fold_count.values()):
        raise ValueError(f"Seed {seed} group isolation failed")
    return assignment, fold_support


def category_aurocs(reliability, success, labels):
    output = {}
    for category in CATEGORIES:
        selected = labels == category
        if len(np.unique(success[selected])) != 2:
            raise ValueError(f"Category {category} lacks both outcomes")
        output[category] = {
            "cases": int(selected.sum()),
            "successes": int(success[selected].sum()),
            "failures": int((~success[selected]).sum()),
            "failure_detection_auroc": common.failure_auroc(
                reliability[selected], success[selected]
            ),
        }
    return output


def dispersion_label(auroc_values, original_auroc, rules):
    values = np.asarray(auroc_values, dtype=np.float64)
    sample_sd = float(values.std(ddof=1))
    maximum_difference = float(np.max(np.abs(values - original_auroc)))
    narrow = rules["narrow"]
    moderate = rules["moderate"]
    if (
        sample_sd <= float(narrow["maximum_auroc_sample_sd"])
        and maximum_difference
        <= float(narrow["maximum_absolute_original_difference"])
    ):
        label = "NARROW"
    elif (
        sample_sd <= float(moderate["maximum_auroc_sample_sd"])
        and maximum_difference
        <= float(moderate["maximum_absolute_original_difference"])
    ):
        label = "MODERATE"
    else:
        label = "MATERIAL"
    return {
        "label": label,
        "auroc_sample_sd": sample_sd,
        "maximum_absolute_alternative_minus_original_auroc": maximum_difference,
    }


def markdown_report(result):
    lines = [
        "# Reliability fold sensitivity",
        "",
        "**Status:** DESCRIPTIVE_COMPLETE",
        "",
        "R09 regenerated three complete five-fold partitions while preserving similarity-group isolation. It is descriptive and cannot change the selected model.",
        "",
        "| Seed | AUROC | Difference from original | ECE | Brier skill |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in result["alternative_splits"]:
        lines.append(
            f"| {item['seed']} | {item['failure_detection_auroc']:.4f} | "
            f"{item['comparison_to_original']['point_difference']:+.4f} | "
            f"{item['calibration']['ece']:.4f} | {item['calibration']['brier_skill']:.4f} |"
        )
    aggregate = result["across_alternative_splits"]
    sensitivity = result["descriptive_stability"]
    lines.extend(
        [
            "",
            f"Alternative-split AUROC mean was {aggregate['failure_detection_auroc']['mean']:.4f} with sample SD {aggregate['failure_detection_auroc']['sample_sd']:.4f}. The maximum absolute difference from original R02 was {sensitivity['maximum_absolute_alternative_minus_original_auroc']:.4f}.",
            "",
            f"The preregistered descriptive label is **{sensitivity['label']}**. This label is not a performance gate and does not alter R02 selection.",
            "",
            "## True-category AUROCs",
            "",
            "| Seed | Glioma | Meningioma | Pituitary |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in result["alternative_splits"]:
        groups = item["true_category_aurocs"]
        lines.append(
            f"| {item['seed']} | {groups['glioma']['failure_detection_auroc']:.4f} | "
            f"{groups['meningioma']['failure_detection_auroc']:.4f} | "
            f"{groups['pituitary']['failure_detection_auroc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "No test image was accessed, the segmentation threshold remained 0.50, and no classifier or neural segmenter was retrained.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R09 is not in its registered pre-run state")
    source_checks = (
        (config["amendment_12_path"], "amendment_12_sha256", "Amendment 12"),
        (config["r02_result_path"], "r02_result_sha256", "R02 result"),
        (config["r02_manifest_path"], "r02_manifest_sha256", "R02 manifest"),
        (config["r02_oof_path"], "r02_oof_sha256", "R02 OOF"),
        (config["r08_result_path"], "r08_result_sha256", "R08 result"),
        (config["r08_manifest_path"], "r08_manifest_sha256", "R08 manifest"),
        (config["c03_mappings_path"], "c03_mappings_sha256", "C03 mappings"),
        (
            config["fold_assignment_path"],
            "fold_assignment_sha256",
            "original fold assignment",
        ),
        (config["stage4_common_path"], "stage4_common_sha256", "Stage 4 common"),
    )
    verified_hashes = {
        label: verify_hash(ROOT / path, config[key], label)
        for path, key, label in source_checks
    }
    r01_config = calibration.read_json(ROOT / "experiments/r01_ablation/config.json")
    r02_config_path = ROOT / "experiments/r02_reduced_features/config.json"
    r02_manifest = calibration.read_json(ROOT / config["r02_manifest_path"])
    r02_config_entry = next(
        item
        for item in r02_manifest["files"]
        if item["path"] == "experiments/r02_reduced_features/config.json"
    )
    verified_hashes["R02 config via manifest"] = verify_hash(
        r02_config_path, r02_config_entry["sha256"], "R02 config"
    )
    r02_config = calibration.read_json(r02_config_path)
    verified_hashes["feature reference"] = verify_hash(
        ROOT / r02_config["feature_reference_script_path"],
        r02_config["feature_reference_script_sha256"],
        "feature reference",
    )
    loader_config = {
        **r02_config,
        "c03_result_sha256": r01_config["c03_result_sha256"],
        "expected_c03_hard_mask_changes": {
            "pixels": config["fail_closed"]["expected_c03_changed_pixels"],
            "images": config["fail_closed"]["expected_c03_changed_images"],
        },
    }
    data = common.load_calibrated_e08(loader_config)
    if len(data["rows"]) != config["population"]["cases"]:
        raise ValueError("R09 population count mismatch")
    r01_result = calibration.read_json(ROOT / "experiments/r01_ablation/result.json")
    all_features = tuple(r01_result["baseline"]["features"])
    selected = tuple(config["features"])
    all_matrix, feature_contract = common.feature_matrix(
        data["probability"], all_features
    )
    lookup = {feature: index for index, feature in enumerate(all_features)}
    features = all_matrix[:, [lookup[feature] for feature in selected]]
    dice = common.dice_scores(data["probability"], data["targets"])
    success = dice >= float(config["target"]["threshold"])
    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    labels = np.asarray([row["label"] for row in data["rows"]])
    if len(set(groups)) != config["population"]["similarity_groups"]:
        raise ValueError("R09 group-count mismatch")

    original_reliability, _original_baseline, _models = common.cross_fit(
        features, success, data["assignment"]
    )
    committed_rows = calibration.read_csv(ROOT / config["r02_oof_path"])
    committed = np.empty(len(committed_rows), dtype=np.float64)
    for index, (current, row) in enumerate(
        zip(data["rows"], committed_rows, strict=True)
    ):
        if current["path"] != row["path"]:
            raise ValueError(f"R02 path mismatch at row {index}")
        committed[index] = float(row["reduced_oof_reliability"])
    tolerance = float(config["fail_closed"]["r02_reproduction_absolute_tolerance"])
    maximum_original_difference = float(
        np.max(np.abs(original_reliability - committed))
    )
    if maximum_original_difference > tolerance:
        raise ValueError("Original R02 OOF vector mismatch")
    original_auroc = common.failure_auroc(original_reliability, success)
    if abs(original_auroc - config["fail_closed"]["expected_r02_aggregate_auroc"]) > tolerance:
        raise ValueError("Original R02 AUROC mismatch")

    samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    original_bootstrap = common.bootstrap_auroc(
        original_reliability, success, samples
    )
    assignments = []
    fold_support_rows = []
    split_results = []
    oof_rows = []
    assignment_rows = []
    subgroup_rows = []
    paired_rows = []
    seen_assignments = {tuple(data["assignment"].tolist())}
    for seed in config["alternative_folds"]["seeds"]:
        assignment, support = generate_assignment(success, groups, seed)
        signature = tuple(assignment.tolist())
        if signature in seen_assignments:
            raise ValueError(f"Seed {seed} assignment is not distinct")
        seen_assignments.add(signature)
        assignments.append(assignment)
        fold_support_rows.extend(support)
        reliability, baseline, _models = common.cross_fit(
            features, success, assignment
        )
        metrics = reliability_metrics(reliability, baseline, success)
        auroc = common.failure_auroc(reliability, success)
        categories = category_aurocs(reliability, success, labels)
        bootstrap = common.bootstrap_auroc(reliability, success, samples)
        difference = bootstrap - original_bootstrap
        comparison = {
            "original_r02_auroc": original_auroc,
            "point_difference": auroc - original_auroc,
            "paired_bootstrap_mean_difference": float(difference.mean()),
            "paired_ci95_lower": float(np.percentile(difference, 2.5)),
            "paired_ci95_upper": float(np.percentile(difference, 97.5)),
        }
        split_results.append(
            {
                "seed": seed,
                "failure_detection_auroc": auroc,
                "calibration": metrics,
                "true_category_aurocs": categories,
                "comparison_to_original": comparison,
            }
        )
        for index, row in enumerate(data["rows"]):
            assignment_rows.append(
                {
                    "seed": seed,
                    "path": row["path"],
                    "similarity_group": row["similarity_group"],
                    "fold": int(assignment[index]) + 1,
                }
            )
            oof_rows.append(
                {
                    "seed": seed,
                    "path": row["path"],
                    "true_label": row["label"],
                    "similarity_group": row["similarity_group"],
                    "fold": int(assignment[index]) + 1,
                    "success_dice_at_least_0_50": int(success[index]),
                    "oof_reliability": float(reliability[index]),
                    "oof_failure_risk": float(1.0 - reliability[index]),
                    "fold_training_prevalence_baseline": float(baseline[index]),
                }
            )
        for category in CATEGORIES:
            subgroup_rows.append(
                {"seed": seed, "category": category, **categories[category]}
            )
        for repetition in range(len(samples)):
            paired_rows.append(
                {
                    "seed": seed,
                    "resample": repetition + 1,
                    "original_r02_auroc": float(original_bootstrap[repetition]),
                    "alternative_auroc": float(bootstrap[repetition]),
                    "alternative_minus_original": float(difference[repetition]),
                }
            )

    metric_keys = ("failure_detection_auroc", "ece", "brier", "brier_skill")
    aggregate = {}
    for key in metric_keys:
        values = np.asarray(
            [
                item["failure_detection_auroc"]
                if key == "failure_detection_auroc"
                else item["calibration"][key]
                for item in split_results
            ]
        )
        aggregate[key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
        }
    stability = dispersion_label(
        [item["failure_detection_auroc"] for item in split_results],
        original_auroc,
        config["descriptive_label"],
    )
    comparison_rows = [
        {"seed": item["seed"], **item["comparison_to_original"]}
        for item in split_results
    ]
    metric_rows = [
        {
            "seed": item["seed"],
            "failure_detection_auroc": item["failure_detection_auroc"],
            "ece": item["calibration"]["ece"],
            "mce": item["calibration"]["mce"],
            "brier": item["calibration"]["brier"],
            "prevalence_baseline_brier": item["calibration"][
                "prevalence_baseline_brier"
            ],
            "brier_skill": item["calibration"]["brier_skill"],
        }
        for item in split_results
    ]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    calibration.write_csv(ARTIFACTS / "alternative_fold_assignments.csv", assignment_rows)
    calibration.write_csv(ARTIFACTS / "fold_support.csv", fold_support_rows)
    calibration.write_csv(ARTIFACTS / "oof_scores.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "split_metrics.csv", metric_rows)
    calibration.write_csv(ARTIFACTS / "subgroup_aurocs.csv", subgroup_rows)
    calibration.write_csv(ARTIFACTS / "paired_bootstrap_comparisons.csv", paired_rows)
    calibration.write_csv(ARTIFACTS / "comparisons_to_original.csv", comparison_rows)

    verification = {
        "source_hashes": verified_hashes,
        "population_cases": len(success),
        "similarity_group_count": len(set(groups)),
        "alternative_seeds": config["alternative_folds"]["seeds"],
        "all_assignments_distinct": True,
        "all_assignments_cover_every_case_once": True,
        "all_assignments_preserve_group_isolation": True,
        "all_fold_partitions_have_both_outcomes": True,
        "maximum_original_r02_oof_score_difference": maximum_original_difference,
        "original_r02_auroc_absolute_difference": abs(
            original_auroc
            - float(config["fail_closed"]["expected_r02_aggregate_auroc"])
        ),
        "c03_changed_pixels": data["changed_pixels"],
        "c03_changed_images": data["changed_images"],
        "all_selected_features_inference_time_only": True,
        "reference_mask_feature_leakage": False,
        "classifier_probability_feature_leakage": False,
        "raw_image_feature_leakage": False,
        "reported_metrics_are_out_of_fold": True,
        "test_images_accessed": 0,
        "threshold_changed": False,
        "classifier_or_neural_segmenter_retrained": False,
        "reliability_estimators_fitted": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result = {
        "schema_version": 1,
        "experiment_id": "r09_fold_sensitivity",
        "status": "DESCRIPTIVE_COMPLETE",
        "decision": "DESCRIPTIVE_COMPLETE",
        "scope": "Internal BRISC alternative-fold reliability stability; not clinical validation",
        "original_r02": {"failure_detection_auroc": original_auroc},
        "alternative_splits": split_results,
        "across_alternative_splits": aggregate,
        "descriptive_stability": stability,
        "feature_contract": {
            **feature_contract,
            "selected_feature_count": len(selected),
            "selected_features": list(selected),
            "reference_mask_role": "target creation and evaluation only",
        },
        "gate": config["gate"],
        "verification": verification,
        "limitations": [
            "R09 is descriptive and cannot change the selected operational model.",
            "Only three preregistered alternative split seeds were studied.",
            "BRISC patient independence remains unverified even though known similarity groups are isolated.",
            "Category AUROCs with few failures, especially meningioma, are unstable.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    AUDIT_REPORT.write_text(markdown_report(result), encoding="utf-8")
    (EXPERIMENT / "decision.md").write_text(
        f"""# Fold-sensitivity decision

**Decision:** DESCRIPTIVE_COMPLETE

The three preregistered alternative-split AUROCs had mean `{aggregate['failure_detection_auroc']['mean']:.6f}` and sample SD `{aggregate['failure_detection_auroc']['sample_sd']:.6f}`. The maximum absolute change from original R02 was `{stability['maximum_absolute_alternative_minus_original_auroc']:.6f}`, producing the preregistered descriptive label **{stability['label']}**.

R09 has no performance gate and cannot alter R02 selection. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
""",
        encoding="utf-8",
    )
    config["status"] = "COMPLETE"
    calibration.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    item = next(
        entry
        for entry in registry["experiments"]
        if entry["id"] == "r09_fold_sensitivity"
    )
    if item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R09 registry was not in its registered state")
    item["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    calibration.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": "DESCRIPTIVE_COMPLETE",
            "experiment_id": "r09_fold_sensitivity",
            "registered_decision": "DESCRIPTIVE_ONLY",
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
                "decision": "DESCRIPTIVE_COMPLETE",
                "original_r02_auroc": original_auroc,
                "alternative_splits": split_results,
                "across_alternative_splits": aggregate,
                "descriptive_stability": stability,
                "feature_contract": result["feature_contract"],
                "verification": verification,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
