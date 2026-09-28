"""Run and close R10 logistic-estimator seed stability analysis."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
import stage4_reliability_common as common
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r10_seed_stability"
ARTIFACTS = EXPERIMENT / "artifacts"
AUDIT_REPORT = ROOT / "audit/seed_stability.md"


def verify_hash(path, expected, label):
    observed = calibration.sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return observed


def make_seeded_model(seed):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    class_weight=None,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def cross_fit_seed(features, success, assignment, seed):
    reliability = np.empty(len(success), dtype=np.float64)
    baseline = np.empty(len(success), dtype=np.float64)
    for fold in range(5):
        fitting = assignment != fold
        evaluation = assignment == fold
        if len(np.unique(success[fitting])) != 2 or len(np.unique(success[evaluation])) != 2:
            raise ValueError(f"Seed {seed} fold {fold + 1} lacks both outcomes")
        model = make_seeded_model(seed)
        model.fit(features[fitting], success[fitting])
        reliability[evaluation] = model.predict_proba(features[evaluation])[:, 1]
        baseline[evaluation] = float(success[fitting].mean())
    if not np.isfinite(reliability).all() or not np.isfinite(baseline).all():
        raise ValueError(f"Seed {seed} produced non-finite OOF scores")
    return reliability, baseline


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


def stability_label(maximum_score_difference, auroc_values, r02_auroc, rules):
    values = np.asarray(auroc_values, dtype=np.float64)
    sample_sd = float(values.std(ddof=1))
    maximum_auroc_difference = float(np.max(np.abs(values - r02_auroc)))
    if maximum_score_difference <= float(rules["exact_probability_tolerance"]):
        label = "EXACT"
    elif (
        sample_sd
        <= float(rules["numerically_stable_maximum_auroc_sample_sd"])
        and maximum_auroc_difference
        <= float(
            rules["numerically_stable_maximum_absolute_r02_auroc_difference"]
        )
    ):
        label = "NUMERICALLY_STABLE"
    else:
        label = "SEED_SENSITIVE"
    return {
        "label": label,
        "maximum_absolute_oof_probability_difference_from_seed_42": float(
            maximum_score_difference
        ),
        "auroc_sample_sd": sample_sd,
        "maximum_absolute_auroc_difference_from_r02": maximum_auroc_difference,
    }


def report_markdown(result):
    lines = [
        "# Reliability estimator seed stability",
        "",
        "**Status:** DESCRIPTIVE_COMPLETE",
        "",
        "R10 varied only the logistic estimator seed while preserving every case, fold, feature, target, scaler, and other estimator parameter.",
        "",
        "| Seed | AUROC | Difference from R02 | ECE | Brier skill | Maximum OOF probability difference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["seeds"]:
        lines.append(
            f"| {item['seed']} | {item['failure_detection_auroc']:.6f} | "
            f"{item['comparison_to_r02']['point_auroc_difference']:+.6f} | "
            f"{item['calibration']['ece']:.6f} | {item['calibration']['brier_skill']:.6f} | "
            f"{item['comparison_to_seed_42']['maximum_absolute_oof_probability_difference']:.3g} |"
        )
    summary = result["across_seeds"]
    stability = result["descriptive_stability"]
    lines.extend(
        [
            "",
            f"Mean AUROC was {summary['failure_detection_auroc']['mean']:.6f} with sample SD {summary['failure_detection_auroc']['sample_sd']:.6f}.",
            "",
            f"The preregistered descriptive stability label is **{stability['label']}**. R10 has no gate and cannot change model selection.",
            "",
            "All ten features are prediction-derived. No test image was accessed, threshold 0.50 was unchanged, and no classifier or neural segmenter was retrained.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R10 is not in its registered pre-run state")
    source_checks = (
        (
            config["stage4_parent_amendment_path"],
            "stage4_parent_amendment_sha256",
            "Stage 4 parent amendment",
        ),
        (config["r02_result_path"], "r02_result_sha256", "R02 result"),
        (config["r02_manifest_path"], "r02_manifest_sha256", "R02 manifest"),
        (config["r02_oof_path"], "r02_oof_sha256", "R02 OOF"),
        (config["r09_result_path"], "r09_result_sha256", "R09 result"),
        (config["r09_manifest_path"], "r09_manifest_sha256", "R09 manifest"),
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
    parent_text = (ROOT / config["stage4_parent_amendment_path"]).read_text(
        encoding="utf-8"
    )
    if "Estimator seeds 7, 21, 42, 84, and 168" not in parent_text:
        raise ValueError("Parent R10 seed definition not found")

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
        raise ValueError("R10 population count mismatch")
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
    assignment = data["assignment"]
    if len(set(groups)) != config["population"]["similarity_groups"]:
        raise ValueError("R10 group-count mismatch")

    committed_rows = calibration.read_csv(ROOT / config["r02_oof_path"])
    committed = np.empty(len(committed_rows), dtype=np.float64)
    for index, (current, row) in enumerate(
        zip(data["rows"], committed_rows, strict=True)
    ):
        if current["path"] != row["path"]:
            raise ValueError(f"R02 OOF path mismatch at row {index}")
        committed[index] = float(row["reduced_oof_reliability"])
    r02_auroc = float(config["fail_closed"]["expected_r02_aggregate_auroc"])
    samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    r02_bootstrap = common.bootstrap_auroc(committed, success, samples)

    seed_outputs = {}
    for seed in config["estimator"]["seeds"]:
        reliability, baseline = cross_fit_seed(
            features, success, assignment, seed
        )
        seed_outputs[int(seed)] = (reliability, baseline)
    seed_42 = seed_outputs[42][0]
    tolerance = float(config["fail_closed"]["r02_reproduction_absolute_tolerance"])
    maximum_seed42_difference = float(np.max(np.abs(seed_42 - committed)))
    if maximum_seed42_difference > tolerance:
        raise ValueError("Seed-42 R02 OOF vector mismatch")
    if abs(common.failure_auroc(seed_42, success) - r02_auroc) > tolerance:
        raise ValueError("Seed-42 R02 AUROC mismatch")

    seed_results = []
    metric_rows = []
    oof_rows = []
    paired_rows = []
    maximum_probability_difference = 0.0
    for seed in config["estimator"]["seeds"]:
        reliability, baseline = seed_outputs[int(seed)]
        metrics = reliability_metrics(reliability, baseline, success)
        auroc = common.failure_auroc(reliability, success)
        bootstrap = common.bootstrap_auroc(reliability, success, samples)
        paired = bootstrap - r02_bootstrap
        score_difference = float(np.max(np.abs(reliability - seed_42)))
        maximum_probability_difference = max(
            maximum_probability_difference, score_difference
        )
        item = {
            "seed": seed,
            "failure_detection_auroc": auroc,
            "calibration": metrics,
            "comparison_to_r02": {
                "r02_auroc": r02_auroc,
                "point_auroc_difference": auroc - r02_auroc,
                "paired_bootstrap_mean_difference": float(paired.mean()),
                "paired_ci95_lower": float(np.percentile(paired, 2.5)),
                "paired_ci95_upper": float(np.percentile(paired, 97.5)),
            },
            "comparison_to_seed_42": {
                "maximum_absolute_oof_probability_difference": score_difference
            },
        }
        seed_results.append(item)
        metric_rows.append(
            {
                "seed": seed,
                "failure_detection_auroc": auroc,
                "ece": metrics["ece"],
                "mce": metrics["mce"],
                "brier": metrics["brier"],
                "prevalence_baseline_brier": metrics[
                    "prevalence_baseline_brier"
                ],
                "brier_skill": metrics["brier_skill"],
                "maximum_absolute_oof_probability_difference_from_seed_42": score_difference,
            }
        )
        for index, row in enumerate(data["rows"]):
            oof_rows.append(
                {
                    "seed": seed,
                    "path": row["path"],
                    "similarity_group": row["similarity_group"],
                    "fold": int(assignment[index]) + 1,
                    "success_dice_at_least_0_50": int(success[index]),
                    "oof_reliability": float(reliability[index]),
                    "oof_failure_risk": float(1.0 - reliability[index]),
                }
            )
        for repetition in range(len(samples)):
            paired_rows.append(
                {
                    "seed": seed,
                    "resample": repetition + 1,
                    "r02_auroc": float(r02_bootstrap[repetition]),
                    "seed_auroc": float(bootstrap[repetition]),
                    "seed_minus_r02_auroc": float(paired[repetition]),
                }
            )

    aggregate = {}
    for key in ("failure_detection_auroc", "ece", "brier", "brier_skill"):
        values = np.asarray(
            [
                item["failure_detection_auroc"]
                if key == "failure_detection_auroc"
                else item["calibration"][key]
                for item in seed_results
            ]
        )
        aggregate[key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
        }
    stability = stability_label(
        maximum_probability_difference,
        [item["failure_detection_auroc"] for item in seed_results],
        r02_auroc,
        config["descriptive_label"],
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    calibration.write_csv(ARTIFACTS / "seed_metrics.csv", metric_rows)
    calibration.write_csv(ARTIFACTS / "oof_scores.csv", oof_rows)
    calibration.write_csv(ARTIFACTS / "paired_bootstrap_comparisons.csv", paired_rows)
    verification = {
        "source_hashes": verified_hashes,
        "registered_seeds": config["estimator"]["seeds"],
        "parent_seed_definition_verified": True,
        "population_cases": len(success),
        "similarity_group_count": len(set(groups)),
        "fold_count": len(set(assignment)),
        "maximum_seed42_r02_oof_score_difference": maximum_seed42_difference,
        "seed42_r02_auroc_absolute_difference": abs(
            common.failure_auroc(seed_42, success) - r02_auroc
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
        "experiment_id": "r10_seed_stability",
        "status": "DESCRIPTIVE_COMPLETE",
        "decision": "DESCRIPTIVE_COMPLETE",
        "scope": "Internal BRISC estimator-seed stability; not clinical validation",
        "r02_aggregate_auroc": r02_auroc,
        "seeds": seed_results,
        "across_seeds": aggregate,
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
            "R10 is descriptive and cannot change model selection.",
            "Only the five preregistered estimator seeds were evaluated.",
            "The lbfgs logistic solver may be deterministic for this problem, so exact equality does not imply all estimator families are seed-invariant.",
            "This internal BRISC result does not establish external or clinical generalization.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    AUDIT_REPORT.write_text(report_markdown(result), encoding="utf-8")
    (EXPERIMENT / "decision.md").write_text(
        f"""# Estimator-seed stability decision

**Decision:** DESCRIPTIVE_COMPLETE

The five preregistered seed AUROCs had mean `{aggregate['failure_detection_auroc']['mean']:.6f}` and sample SD `{aggregate['failure_detection_auroc']['sample_sd']:.6f}`. The preregistered descriptive label is **{stability['label']}**; maximum OOF probability difference from seed 42 was `{stability['maximum_absolute_oof_probability_difference_from_seed_42']:.3g}`.

R10 has no performance gate and cannot alter R02 selection. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
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
        if entry["id"] == "r10_seed_stability"
    )
    if item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R10 registry was not in its registered state")
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
            "experiment_id": "r10_seed_stability",
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
                "seeds": seed_results,
                "across_seeds": aggregate,
                "descriptive_stability": stability,
                "feature_contract": result["feature_contract"],
                "verification": verification,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
