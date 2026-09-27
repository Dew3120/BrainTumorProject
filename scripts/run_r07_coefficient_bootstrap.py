"""Run and close R07 coefficient-bootstrap stability analysis."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as calibration
import matplotlib.pyplot as plt
import numpy as np
import stage4_reliability_common as common
from fit_segmenter_case_reliability import make_model

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/r07_coefficient_bootstrap"
ARTIFACTS = EXPERIMENT / "artifacts"
AUDIT_REPORT = ROOT / "audit/reliability_coefficients_with_ci.md"


def verify_hash(path, expected, label):
    observed = calibration.sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return observed


def coefficient_matrix(models, expected_features):
    matrix = np.vstack(
        [model.named_steps["logistic"].coef_[0] for model in models]
    )
    expected_shape = (5, len(expected_features))
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise ValueError("Five-fold coefficient matrix is invalid")
    return matrix


def read_committed_coefficients(path, feature_names):
    rows = calibration.read_csv(path)
    by_feature = {row["feature"]: row for row in rows}
    if len(by_feature) != len(rows):
        raise ValueError(f"Duplicate feature in {path}")
    matrix = np.empty((5, len(feature_names)), dtype=np.float64)
    for column, feature in enumerate(feature_names):
        if feature not in by_feature:
            raise ValueError(f"Missing committed coefficient: {feature}")
        for fold in range(5):
            matrix[fold, column] = float(
                by_feature[feature][
                    f"fold_{fold + 1}_standardized_coefficient"
                ]
            )
    if not np.isfinite(matrix).all():
        raise ValueError("Committed coefficient matrix is non-finite")
    return matrix


def sign(values):
    return np.sign(np.asarray(values, dtype=np.float64)).astype(np.int8)


def coefficient_summary(
    feature_names, fold_matrix, bootstrap_matrix, full_coefficients
):
    rows = []
    for column, feature in enumerate(feature_names):
        folds = fold_matrix[:, column]
        boot = bootstrap_matrix[:, column]
        reference_sign = int(sign([full_coefficients[column]])[0])
        rows.append(
            {
                "feature": feature,
                "full_population_standardized_coefficient": float(
                    full_coefficients[column]
                ),
                "fold_mean_standardized_coefficient": float(folds.mean()),
                "fold_sample_sd": float(folds.std(ddof=1)),
                "five_fold_sign_agreement_fraction": float(
                    np.mean(sign(folds) == reference_sign)
                ),
                "five_fold_sign_agreement_count": int(
                    np.sum(sign(folds) == reference_sign)
                ),
                "bootstrap_mean_standardized_coefficient": float(boot.mean()),
                "bootstrap_sample_sd": float(boot.std(ddof=1)),
                "bootstrap_ci95_lower": float(np.percentile(boot, 2.5)),
                "bootstrap_ci95_upper": float(np.percentile(boot, 97.5)),
                "bootstrap_sign_agreement_fraction": float(
                    np.mean(sign(boot) == reference_sign)
                ),
                "bootstrap_sign_agreement_count": int(
                    np.sum(sign(boot) == reference_sign)
                ),
                "bootstrap_repetitions": len(boot),
                "reference_sign": (
                    "positive"
                    if reference_sign > 0
                    else "negative"
                    if reference_sign < 0
                    else "zero"
                ),
                "bootstrap_interval_excludes_zero": bool(
                    np.percentile(boot, 2.5) > 0
                    or np.percentile(boot, 97.5) < 0
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -abs(row["bootstrap_mean_standardized_coefficient"]),
            row["feature"],
        ),
    )


def compare_r01(feature_names, r01_matrix, r02_matrix, full_coefficients):
    rows = []
    for column, feature in enumerate(feature_names):
        r01_values = r01_matrix[:, column]
        r02_values = r02_matrix[:, column]
        r01_mean = float(r01_values.mean())
        r02_mean = float(r02_values.mean())
        rows.append(
            {
                "feature": feature,
                "r01_14_feature_fold_mean_coefficient": r01_mean,
                "r02_10_feature_fold_mean_coefficient": r02_mean,
                "r02_minus_r01_signed_change": r02_mean - r01_mean,
                "r02_minus_r01_absolute_magnitude_change": abs(r02_mean)
                - abs(r01_mean),
                "r01_positive_folds": int(np.sum(r01_values > 0)),
                "r01_negative_folds": int(np.sum(r01_values < 0)),
                "r02_positive_folds": int(np.sum(r02_values > 0)),
                "r02_negative_folds": int(np.sum(r02_values < 0)),
                "r01_five_fold_sign_agreement_fraction": float(
                    np.mean(sign(r01_values) == sign([r01_mean])[0])
                ),
                "r02_five_fold_sign_agreement_fraction": float(
                    np.mean(
                        sign(r02_values)
                        == sign([full_coefficients[column]])[0]
                    )
                ),
                "mean_sign_changed": bool(
                    sign([r01_mean])[0] != sign([r02_mean])[0]
                ),
            }
        )
    return rows


def plot_distributions(summary_rows, bootstrap_matrix, feature_names, output):
    index = {name: position for position, name in enumerate(feature_names)}
    figure, axes = plt.subplots(5, 2, figsize=(14, 19), constrained_layout=True)
    for axis, row in zip(axes.flat, summary_rows, strict=True):
        values = bootstrap_matrix[:, index[row["feature"]]]
        axis.hist(values, bins=40, color="#3569a8", alpha=0.82, edgecolor="white")
        axis.axvline(0.0, color="black", linewidth=1.2, label="zero")
        axis.axvline(
            row["bootstrap_mean_standardized_coefficient"],
            color="#d1495b",
            linewidth=1.8,
            label="mean",
        )
        axis.axvline(
            row["bootstrap_ci95_lower"],
            color="#2a9d8f",
            linestyle="--",
            linewidth=1.4,
            label="95% limits",
        )
        axis.axvline(
            row["bootstrap_ci95_upper"],
            color="#2a9d8f",
            linestyle="--",
            linewidth=1.4,
        )
        axis.set_title(row["feature"].replace("_", " "))
        axis.set_xlabel("Standardized logistic coefficient")
        axis.set_ylabel("Bootstrap count")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "R07 similarity-group bootstrap coefficient distributions (n=2,000)",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def markdown_report(summary_rows, comparison_rows, verification):
    lines = [
        "# Reliability coefficients with bootstrap confidence intervals",
        "",
        "**Status:** DESCRIPTIVE_COMPLETE",
        "",
        "R07 refit the accepted R02 standardized logistic reliability model on 2,000 similarity-group bootstrap resamples. This is a development-set stability analysis, not external or clinical validation. Coefficients are conditional associations among correlated prediction-derived features and are not causal effects.",
        "",
        "## Coefficient stability",
        "",
        "| Rank | Feature | Full fit | Bootstrap mean | SD | 95% CI | Fold sign | Bootstrap sign |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary_rows, start=1):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['full_population_standardized_coefficient']:.4f} | "
            f"{row['bootstrap_mean_standardized_coefficient']:.4f} | {row['bootstrap_sample_sd']:.4f} | "
            f"[{row['bootstrap_ci95_lower']:.4f}, {row['bootstrap_ci95_upper']:.4f}] | "
            f"{row['five_fold_sign_agreement_count']}/5 | "
            f"{row['bootstrap_sign_agreement_count']}/2,000 |"
        )
    lines.extend(
        [
            "",
            "The ordering above is fixed by absolute bootstrap-mean magnitude. An interval excluding zero is evidence of coefficient-direction stability in this internal development population; it does not prove causal importance.",
            "",
            "## R01 fourteen-feature comparison",
            "",
            "| Feature | R01 mean | R02 mean | Signed change | Magnitude change | Mean sign changed |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| `{row['feature']}` | {row['r01_14_feature_fold_mean_coefficient']:.4f} | "
            f"{row['r02_10_feature_fold_mean_coefficient']:.4f} | "
            f"{row['r02_minus_r01_signed_change']:+.4f} | "
            f"{row['r02_minus_r01_absolute_magnitude_change']:+.4f} | "
            f"{'yes' if row['mean_sign_changed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Cases: {verification['population_cases']}",
            f"- Similarity groups: {verification['similarity_group_count']}",
            f"- Valid bootstrap refits: {verification['valid_bootstrap_repetitions']}",
            f"- Maximum R02 fold-coefficient reproduction difference: `{verification['maximum_r02_fold_coefficient_difference']:.3g}`",
            f"- C03 changed pixels/images reproduced: {verification['c03_changed_pixels']}/{verification['c03_changed_images']}",
            "- All ten features are available at inference time and derive only from the calibrated prediction.",
            "- Reference masks were used only to define the reliability target and evaluate it.",
            "- Test images accessed: 0; threshold changes: 0; classifier/neural-segmenter retraining: 0.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config_path = EXPERIMENT / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R07 is not in its registered pre-run state")

    source_checks = (
        (config["amendment_10_path"], "amendment_10_sha256", "Amendment 10"),
        (config["r01_result_path"], "r01_result_sha256", "R01 result"),
        (config["r01_manifest_path"], "r01_manifest_sha256", "R01 manifest"),
        (
            config["r01_fold_coefficients_path"],
            "r01_fold_coefficients_sha256",
            "R01 fold coefficients",
        ),
        (config["r02_result_path"], "r02_result_sha256", "R02 result"),
        (config["r02_manifest_path"], "r02_manifest_sha256", "R02 manifest"),
        (config["r02_oof_path"], "r02_oof_sha256", "R02 OOF"),
        (
            config["r02_fold_coefficients_path"],
            "r02_fold_coefficients_sha256",
            "R02 fold coefficients",
        ),
        (config["r06_result_path"], "r06_result_sha256", "R06 result"),
        (config["r06_manifest_path"], "r06_manifest_sha256", "R06 manifest"),
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
    manifest_r02_config = next(
        item
        for item in r02_manifest["files"]
        if item["path"] == "experiments/r02_reduced_features/config.json"
    )
    verified_hashes["R02 config via manifest"] = verify_hash(
        r02_config_path, manifest_r02_config["sha256"], "R02 config"
    )
    r02_config = calibration.read_json(r02_config_path)
    verify_hash(
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
        raise ValueError("R07 population count mismatch")

    r01_result = calibration.read_json(ROOT / config["r01_result_path"])
    all_features = tuple(r01_result["baseline"]["features"])
    selected = tuple(config["features"])
    all_matrix, feature_contract = common.feature_matrix(
        data["probability"], all_features
    )
    lookup = {feature: index for index, feature in enumerate(all_features)}
    selected_indices = np.asarray([lookup[feature] for feature in selected])
    features = all_matrix[:, selected_indices]
    if features.shape != (680, 10) or not np.isfinite(features).all():
        raise ValueError("R07 selected feature matrix is invalid")

    dice = common.dice_scores(data["probability"], data["targets"])
    success = dice >= float(config["target"]["threshold"])
    groups = np.asarray([row["similarity_group"] for row in data["rows"]])
    if len(set(groups)) != config["population"]["similarity_groups"]:
        raise ValueError("R07 similarity-group count mismatch")

    _scores, _baseline, fold_models = common.cross_fit(
        features, success, data["assignment"]
    )
    folds = coefficient_matrix(fold_models, selected)
    committed_folds = read_committed_coefficients(
        ROOT / config["r02_fold_coefficients_path"], selected
    )
    maximum_fold_difference = float(np.max(np.abs(folds - committed_folds)))
    tolerance = float(
        config["fold_coefficients"]["reproduction_absolute_tolerance"]
    )
    if maximum_fold_difference > tolerance:
        raise ValueError("R02 fold coefficient reproduction mismatch")

    full_model = make_model()
    full_model.fit(features, success)
    full_coefficients = full_model.named_steps["logistic"].coef_[0]
    if full_coefficients.shape != (10,) or not np.isfinite(full_coefficients).all():
        raise ValueError("Full-population coefficient vector is invalid")

    samples = common.bootstrap_indices(
        groups,
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    if len(samples) != 2000:
        raise ValueError("R07 bootstrap count mismatch")
    bootstraps = np.empty((len(samples), len(selected)), dtype=np.float64)
    sample_diagnostics = []
    for repetition, members in enumerate(samples):
        outcome_count = len(np.unique(success[members]))
        if outcome_count != 2:
            raise ValueError(
                f"Bootstrap resample {repetition + 1} lacks both outcomes"
            )
        model = make_model()
        model.fit(features[members], success[members])
        values = model.named_steps["logistic"].coef_[0]
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError(
                f"Bootstrap resample {repetition + 1} has invalid coefficients"
            )
        bootstraps[repetition] = values
        sample_diagnostics.append(
            {
                "resample": repetition + 1,
                "sampled_rows": len(members),
                "sampled_successes": int(success[members].sum()),
                "sampled_failures": int((~success[members]).sum()),
                "unique_groups_represented": len(set(groups[members])),
            }
        )

    summaries = coefficient_summary(
        selected, folds, bootstraps, full_coefficients
    )
    r01_folds = read_committed_coefficients(
        ROOT / config["r01_fold_coefficients_path"], selected
    )
    comparisons = compare_r01(selected, r01_folds, folds, full_coefficients)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    long_rows = [
        {
            "resample": repetition + 1,
            "feature": feature,
            "standardized_coefficient": float(bootstraps[repetition, column]),
        }
        for repetition in range(len(samples))
        for column, feature in enumerate(selected)
    ]
    full_rows = [
        {
            "feature": feature,
            "full_population_standardized_coefficient": float(
                full_coefficients[column]
            ),
        }
        for column, feature in enumerate(selected)
    ]
    calibration.write_csv(ARTIFACTS / "coefficient_summary.csv", summaries)
    calibration.write_csv(ARTIFACTS / "bootstrap_coefficients.csv", long_rows)
    calibration.write_csv(ARTIFACTS / "bootstrap_diagnostics.csv", sample_diagnostics)
    calibration.write_csv(ARTIFACTS / "full_population_coefficients.csv", full_rows)
    calibration.write_csv(ARTIFACTS / "r01_r02_comparison.csv", comparisons)
    plot_distributions(
        summaries,
        bootstraps,
        selected,
        ARTIFACTS / "coefficient_bootstrap_distributions.png",
    )

    verification = {
        "source_hashes": verified_hashes,
        "population_cases": len(success),
        "population_successes": int(success.sum()),
        "population_failures": int((~success).sum()),
        "similarity_group_count": len(set(groups)),
        "fold_count": len(set(data["assignment"])),
        "maximum_r02_fold_coefficient_difference": maximum_fold_difference,
        "r02_coefficient_reproduction_tolerance": tolerance,
        "valid_bootstrap_repetitions": len(samples),
        "all_bootstrap_draws_have_both_outcomes": True,
        "all_bootstrap_coefficients_finite": True,
        "c03_changed_pixels": data["changed_pixels"],
        "c03_changed_images": data["changed_images"],
        "all_selected_features_inference_time_only": True,
        "reference_mask_feature_leakage": False,
        "classifier_probability_feature_leakage": False,
        "raw_image_feature_leakage": False,
        "full_population_fit_is_descriptive_only": True,
        "test_images_accessed": 0,
        "threshold_changed": False,
        "classifier_or_neural_segmenter_retrained": False,
        "reliability_estimators_fitted": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result = {
        "schema_version": 1,
        "experiment_id": "r07_coefficient_bootstrap",
        "status": "DESCRIPTIVE_COMPLETE",
        "decision": "DESCRIPTIVE_COMPLETE",
        "scope": "Internal BRISC development coefficient stability; not clinical validation",
        "population": {
            "cases": len(success),
            "successes": int(success.sum()),
            "failures": int((~success).sum()),
            "similarity_groups": len(set(groups)),
        },
        "coefficient_summary_ranked_by_absolute_bootstrap_mean": summaries,
        "r01_fourteen_feature_comparison": comparisons,
        "feature_contract": {
            **feature_contract,
            "selected_feature_count": len(selected),
            "selected_features": list(selected),
            "reference_mask_role": "target creation and evaluation only",
        },
        "gate": config["gate"],
        "verification": verification,
        "limitations": [
            "R07 is descriptive and cannot change the registered R02 operational model.",
            "Intervals quantify similarity-group resampling stability in the internal BRISC development population, not external generalization.",
            "Standardized coefficients are conditional associations among correlated predictors and are not causal effects.",
            "The full-population fit defines the reference sign and is not reported as prospective performance evidence.",
        ],
    }
    calibration.write_json(EXPERIMENT / "result.json", result)
    AUDIT_REPORT.write_text(
        markdown_report(summaries, comparisons, verification), encoding="utf-8"
    )
    (EXPERIMENT / "decision.md").write_text(
        """# Coefficient-bootstrap stability decision

**Decision:** DESCRIPTIVE_COMPLETE

All 2,000 registered similarity-group bootstrap samples produced valid finite standardized coefficients for the accepted R02 ten-feature logistic reliability model. R07 has no performance gate and does not change model selection.

The coefficient table, full-population reference coefficients, R01 comparison, and ten coefficient distributions are preserved in `artifacts/`. Coefficients are descriptive associations, not causal effects. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
""",
        encoding="utf-8",
    )
    config["status"] = "COMPLETE"
    calibration.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    registry_item = next(
        item
        for item in registry["experiments"]
        if item["id"] == "r07_coefficient_bootstrap"
    )
    if registry_item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("R07 registry was not in its registered state")
    registry_item["status"] = "COMPLETE"
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
            "experiment_id": "r07_coefficient_bootstrap",
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
                "coefficient_summary": summaries,
                "comparison": comparisons,
                "feature_contract": result["feature_contract"],
                "verification": verification,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
