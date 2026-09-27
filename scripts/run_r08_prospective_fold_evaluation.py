"""Run and close R08 prospective five-fold reliability evaluation."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import stage4_reliability_common as common

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r08_prospective_fold_evaluation"
ARTIFACTS = EXPERIMENT / "artifacts"
AUDIT_REPORT = ROOT / "audit/prospective_reliability_evaluation.md"
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


def summarize_fold_metrics(rows):
    keys = (
        "failure_detection_auroc",
        "ece",
        "mce",
        "brier",
        "prevalence_baseline_brier",
        "brier_skill",
    )
    output = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid five-fold metric: {key}")
        output[key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
        }
    return output


def subgroup_bootstrap(reliability, success, labels, samples, minimum_valid):
    summaries = {}
    long_rows = []
    for category in CATEGORIES:
        mask = labels == category
        cases = int(mask.sum())
        failures = int((~success[mask]).sum())
        successes = int(success[mask].sum())
        if not failures or not successes:
            raise ValueError(f"Category {category} lacks both outcomes")
        observed = common.failure_auroc(reliability[mask], success[mask])
        values = []
        invalid = 0
        for repetition, members in enumerate(samples, start=1):
            selected = members[labels[members] == category]
            if len(selected) == 0 or len(np.unique(success[selected])) != 2:
                invalid += 1
                continue
            value = common.failure_auroc(reliability[selected], success[selected])
            values.append(value)
            long_rows.append(
                {
                    "category": category,
                    "resample": repetition,
                    "failure_detection_auroc": value,
                }
            )
        if len(values) < minimum_valid:
            raise ValueError(
                f"Category {category} has only {len(values)} valid bootstraps"
            )
        values = np.asarray(values, dtype=np.float64)
        summaries[category] = {
            "cases": cases,
            "successes": successes,
            "failures": failures,
            "failure_detection_auroc": observed,
            "bootstrap": {
                "valid_repetitions": len(values),
                "invalid_repetitions": invalid,
                "mean": float(values.mean()),
                "ci95_lower": float(np.percentile(values, 2.5)),
                "ci95_upper": float(np.percentile(values, 97.5)),
            },
        }
    return summaries, long_rows


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


def gate_passes(mean_fold_auroc, minimum):
    return float(mean_fold_auroc) >= float(minimum)


def report_markdown(result):
    lines = [
        "# Prospective reliability evaluation",
        "",
        f"**Decision:** {result['decision']}",
        "",
        "Each case was scored only by a standardized logistic model fitted on the other four similarity-aware folds. The controlling statistic is the unweighted mean of the five held-out-fold AUROCs.",
        "",
        "| Fold | Cases | Failures | AUROC | ECE | Brier skill |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["fold_metrics"]:
        lines.append(
            f"| {row['fold']} | {row['cases']} | {row['failures']} | "
            f"{row['failure_detection_auroc']:.4f} | {row['ece']:.4f} | "
            f"{row['brier_skill']:.4f} |"
        )
    mean = result["fold_summary"]
    lines.extend(
        [
            "",
            (
                f"Mean fold AUROC was **{mean['failure_detection_auroc']['mean']:.4f}** "
                f"(SD {mean['failure_detection_auroc']['sample_sd']:.4f}). The registered minimum was `0.941`, so R08 **{result['decision'].lower()}ed**."
            ),
            "",
            (
                f"Aggregate OOF AUROC was {result['overall_oof']['failure_detection_auroc']:.4f} "
                f"(95% group-bootstrap CI {result['overall_oof']['bootstrap']['ci95_lower']:.4f}–{result['overall_oof']['bootstrap']['ci95_upper']:.4f})."
            ),
            "",
            "## True-category AUROC",
            "",
            "| Category | Cases | Failures | AUROC | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category in CATEGORIES:
        item = result["true_category_aurocs"][category]
        lines.append(
            f"| {category} | {item['cases']} | {item['failures']} | "
            f"{item['failure_detection_auroc']:.4f} | "
            f"[{item['bootstrap']['ci95_lower']:.4f}, {item['bootstrap']['ci95_upper']:.4f}] |"
        )
    capture = result["zero_dice_capture_top_20_percent_risk"]
    lines.extend(
        [
            "",
            f"The top {capture['top_risk_cases']} risks captured {capture['captured']}/{capture['total']} zero-Dice cases.",
            "",
            "All features are prediction-derived. No test image was accessed, the threshold remained 0.50, and no classifier or neural segmenter was retrained.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R08 is not in its registered pre-run state")

    source_checks = (
        (config["amendment_11_path"], "amendment_11_sha256", "Amendment 11"),
        (config["r02_result_path"], "r02_result_sha256", "R02 result"),
        (config["r02_manifest_path"], "r02_manifest_sha256", "R02 manifest"),
        (config["r02_oof_path"], "r02_oof_sha256", "R02 OOF"),
        (config["r07_result_path"], "r07_result_sha256", "R07 result"),
        (config["r07_manifest_path"], "r07_manifest_sha256", "R07 manifest"),
        (config["c03_mappings_path"], "c03_mappings_sha256", "C03 mappings"),
        (
            config["fold_assignment_path"],
            "fold_assignment_sha256",
            "fold assignment",
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
        raise ValueError("R08 population count mismatch")

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
    assignment = data["assignment"]
    if len(set(groups)) != config["population"]["similarity_groups"]:
        raise ValueError("R08 group-count mismatch")
    group_folds = {}
    for group, fold in zip(groups, assignment, strict=True):
        group_folds.setdefault(group, set()).add(int(fold))
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("Similarity group crosses R08 folds")

    reliability, baseline, _models = common.cross_fit(
        features, success, assignment
    )
    committed_rows = calibration.read_csv(ROOT / config["r02_oof_path"])
    committed = np.empty(len(committed_rows), dtype=np.float64)
    for index, (current, row) in enumerate(
        zip(data["rows"], committed_rows, strict=True)
    ):
        if current["path"] != row["path"]:
            raise ValueError(f"R02 OOF path mismatch at row {index}")
        committed[index] = float(row["reduced_oof_reliability"])
    maximum_difference = float(np.max(np.abs(reliability - committed)))
    tolerance = float(config["fail_closed"]["r02_reproduction_absolute_tolerance"])
    if maximum_difference > tolerance:
        raise ValueError("R02 OOF reliability reproduction mismatch")
    overall_auroc = common.failure_auroc(reliability, success)
    expected_auroc = float(config["fail_closed"]["expected_r02_aggregate_auroc"])
    if abs(overall_auroc - expected_auroc) > tolerance:
        raise ValueError("R02 aggregate AUROC reproduction mismatch")

    fold_rows = []
    for fold in range(5):
        evaluation = assignment == fold
        fitting = ~evaluation
        if len(np.unique(success[evaluation])) != 2 or len(np.unique(success[fitting])) != 2:
            raise ValueError(f"Fold {fold + 1} lacks both outcomes")
        metrics = reliability_metrics(
            reliability[evaluation], baseline[evaluation], success[evaluation]
        )
        fold_rows.append(
            {
                "fold": fold + 1,
                "cases": int(evaluation.sum()),
                "successes": int(success[evaluation].sum()),
                "failures": int((~success[evaluation]).sum()),
                "training_success_prevalence": float(success[fitting].mean()),
                "failure_detection_auroc": common.failure_auroc(
                    reliability[evaluation], success[evaluation]
                ),
                "ece": metrics["ece"],
                "mce": metrics["mce"],
                "brier": metrics["brier"],
                "prevalence_baseline_brier": metrics[
                    "prevalence_baseline_brier"
                ],
                "brier_skill": metrics["brier_skill"],
            }
        )
    fold_summary = summarize_fold_metrics(fold_rows)

    samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    overall_bootstrap = common.bootstrap_auroc(reliability, success, samples)
    subgroup_summary, subgroup_rows = subgroup_bootstrap(
        reliability,
        success,
        labels,
        samples,
        int(config["bootstrap"]["minimum_valid_subgroup_repetitions"]),
    )
    capture = top_risk_capture(reliability, dice, data["rows"])
    minimum = float(config["gate"]["minimum_mean_fold_auroc"])
    passed = gate_passes(
        fold_summary["failure_detection_auroc"]["mean"], minimum
    )
    decision = "PASS" if passed else "FAIL"

    overall_metrics = reliability_metrics(reliability, baseline, success)
    overall = {
        "failure_detection_auroc": overall_auroc,
        "bootstrap": {
            "repetitions": len(overall_bootstrap),
            "mean": float(overall_bootstrap.mean()),
            "ci95_lower": float(np.percentile(overall_bootstrap, 2.5)),
            "ci95_upper": float(np.percentile(overall_bootstrap, 97.5)),
        },
        "calibration": overall_metrics,
    }
    oof_rows = [
        {
            "path": row["path"],
            "true_label": row["label"],
            "similarity_group": row["similarity_group"],
            "fold": int(assignment[index]) + 1,
            "calibrated_dice": float(dice[index]),
            "success_dice_at_least_0_50": int(success[index]),
            "zero_dice": int(dice[index] == 0),
            "oof_reliability": float(reliability[index]),
            "oof_failure_risk": float(1.0 - reliability[index]),
            "fold_training_prevalence_baseline": float(baseline[index]),
        }
        for index, row in enumerate(data["rows"])
    ]
    overall_bootstrap_rows = [
        {"resample": index + 1, "failure_detection_auroc": float(value)}
        for index, value in enumerate(overall_bootstrap)
    ]
    subgroup_metric_rows = [
        {"category": category, **subgroup_summary[category]}
        for category in CATEGORIES
    ]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    calibration.write_csv(ARTIFACTS / "oof_scores.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "fold_metrics.csv", fold_rows)
    calibration.write_csv(
        ARTIFACTS / "overall_bootstrap_aurocs.csv", overall_bootstrap_rows
    )
    calibration.write_csv(ARTIFACTS / "subgroup_aurocs.csv", subgroup_metric_rows)
    calibration.write_csv(
        ARTIFACTS / "subgroup_bootstrap_aurocs.csv", subgroup_rows
    )

    verification = {
        "source_hashes": verified_hashes,
        "population_cases": len(success),
        "population_successes": int(success.sum()),
        "population_failures": int((~success).sum()),
        "similarity_group_count": len(set(groups)),
        "similarity_group_cross_fold_leakage": False,
        "fold_count": len(set(assignment)),
        "maximum_r02_oof_score_difference": maximum_difference,
        "r02_aggregate_auroc_absolute_difference": abs(overall_auroc - expected_auroc),
        "reproduction_tolerance": tolerance,
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
        "experiment_id": "r08_prospective_fold_evaluation",
        "status": decision,
        "decision": decision,
        "scope": "Internal BRISC prospective-style OOF reliability evaluation; not clinical validation",
        "fold_metrics": fold_rows,
        "fold_summary": fold_summary,
        "overall_oof": overall,
        "true_category_aurocs": subgroup_summary,
        "zero_dice_capture_top_20_percent_risk": capture,
        "gate": {
            **config["gate"],
            "observed_mean_fold_auroc": fold_summary[
                "failure_detection_auroc"
            ]["mean"],
            "status": decision,
        },
        "feature_contract": {
            **feature_contract,
            "selected_feature_count": len(selected),
            "selected_features": list(selected),
            "reference_mask_role": "target creation and evaluation only",
        },
        "verification": verification,
        "limitations": [
            "R08 uses the internal BRISC development population and does not establish external or clinical generalization.",
            "The folds are prospective-style held-out partitions but are not independent patients because BRISC patient independence is unverified.",
            "Subgroup intervals, especially meningioma with few failures, are wide and descriptive.",
            "The mean-fold gate weights folds equally rather than weighting by case count, as preregistered.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    AUDIT_REPORT.write_text(report_markdown(result), encoding="utf-8")
    (EXPERIMENT / "decision.md").write_text(
        f"""# Prospective reliability fold decision

**Decision:** {decision}

The five held-out-fold AUROCs had an unweighted mean of `{fold_summary['failure_detection_auroc']['mean']:.6f}` (sample SD `{fold_summary['failure_detection_auroc']['sample_sd']:.6f}`). The unchanged H7 minimum was `{minimum:.3f}`; equality passes.

Aggregate OOF AUROC was `{overall_auroc:.6f}` with 95% similarity-group bootstrap interval `[{overall['bootstrap']['ci95_lower']:.6f}, {overall['bootstrap']['ci95_upper']:.6f}]`. Reported metrics use only held-out-fold predictions. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
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
        if entry["id"] == "r08_prospective_fold_evaluation"
    )
    if item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R08 registry was not in its registered state")
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
            "status": decision,
            "experiment_id": "r08_prospective_fold_evaluation",
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
                "fold_metrics": fold_rows,
                "fold_summary": fold_summary,
                "overall_oof": overall,
                "true_category_aurocs": subgroup_summary,
                "zero_dice_capture": capture,
                "gate": result["gate"],
                "feature_contract": result["feature_contract"],
                "verification": verification,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
