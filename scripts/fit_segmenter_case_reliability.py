"""Cross-fit a prediction-only case reliability model for the frozen segmenter."""

import json
from collections import Counter
from pathlib import Path

import calibrate_classifier_confidence as classifier_calibration
import calibrate_segmenter_pixels as pixel_calibration
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import torch
import train_segmenter as segmenter
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = pixel_calibration.RUN_NAME
RUN = pixel_calibration.RUN
CHECKPOINT = pixel_calibration.CHECKPOINT
PIXEL_CALIBRATION = pixel_calibration.OUTPUT_JSON
CALIBRATION_FINDINGS = pixel_calibration.CALIBRATION_FINDINGS
OUTPUT_JSON = ROOT / "audit" / "segmenter_case_reliability.json"
OUTPUT_MD = ROOT / "audit" / "segmenter_case_reliability.md"
OOF_PATH = RUN / "segmenter_case_reliability_oof.csv"
COEFFICIENT_PATH = RUN / "segmenter_case_reliability_coefficients.csv"
TOP_RISK_PATH = RUN / "segmenter_case_reliability_top20_risk.csv"
DIAGRAM_PATH = RUN / "segmenter_case_reliability_diagram.png"

FOLDS = 5
SEED = 42
BOOTSTRAP_REPETITIONS = 2000
FAILURE_DICE_THRESHOLD = 0.50
TOP_RISK_CASES = 136
AUROC_REQUIRED = 0.75
ECE_REQUIRED = 0.05
EPSILON = 1e-12
FEATURES = (
    "empty_prediction",
    "predicted_area_fraction",
    "whole_probability_mean",
    "whole_probability_sd",
    "predicted_region_probability_mean",
    "predicted_region_entropy_mean",
    "boundary_entropy_mean",
    "high_confidence_foreground_fraction",
    "uncertain_pixel_fraction",
    "log1p_component_count",
    "largest_component_fraction",
    "bounding_box_fill_ratio",
    "normalized_perimeter",
    "border_touch_fraction",
)

read_json = pixel_calibration.read_json
sha256 = pixel_calibration.sha256
write_csv = pixel_calibration.write_csv
normalized_path = pixel_calibration.normalized_path


def binary_entropy(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1 - EPSILON)
    return -probability * np.log(probability) - (1 - probability) * np.log(1 - probability)


def prediction_features(probability, threshold=0.5):
    """Extract features from one probability map; no reference input is accepted."""
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (128, 128) or not np.isfinite(probability).all():
        raise ValueError("Expected one finite 128x128 probability map")
    hard = probability >= threshold
    area = int(hard.sum())
    total = hard.size
    entropy = binary_entropy(probability)
    uncertain = (probability >= 0.40) & (probability <= 0.60)
    if area == 0:
        return {
            "empty_prediction": 1.0,
            "predicted_area_fraction": 0.0,
            "whole_probability_mean": float(probability.mean()),
            "whole_probability_sd": float(probability.std()),
            "predicted_region_probability_mean": 0.0,
            "predicted_region_entropy_mean": 0.0,
            "boundary_entropy_mean": 0.0,
            "high_confidence_foreground_fraction": 0.0,
            "uncertain_pixel_fraction": float(uncertain.mean()),
            "log1p_component_count": 0.0,
            "largest_component_fraction": 0.0,
            "bounding_box_fill_ratio": 0.0,
            "normalized_perimeter": 0.0,
            "border_touch_fraction": 0.0,
        }

    structure = np.ones((3, 3), dtype=bool)
    components, component_count = label(hard, structure=structure)
    component_areas = np.bincount(components.reshape(-1))[1:]
    largest_component_fraction = float(component_areas.max() / area)
    eroded = binary_erosion(hard, structure=structure, iterations=2, border_value=0)
    dilated = binary_dilation(hard, structure=structure, iterations=2, border_value=0)
    boundary_band = dilated ^ eroded
    boundary_entropy = float(entropy[boundary_band].mean())
    one_pixel_eroded = binary_erosion(
        hard, structure=structure, iterations=1, border_value=0
    )
    perimeter = int((hard & ~one_pixel_eroded).sum())
    y, x = np.where(hard)
    box_area = (int(y.max()) - int(y.min()) + 1) * (int(x.max()) - int(x.min()) + 1)
    border = np.zeros_like(hard)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    predicted_probability = probability[hard]
    return {
        "empty_prediction": 0.0,
        "predicted_area_fraction": area / total,
        "whole_probability_mean": float(probability.mean()),
        "whole_probability_sd": float(probability.std()),
        "predicted_region_probability_mean": float(predicted_probability.mean()),
        "predicted_region_entropy_mean": float(entropy[hard].mean()),
        "boundary_entropy_mean": boundary_entropy,
        "high_confidence_foreground_fraction": float(
            (predicted_probability >= 0.90).mean()
        ),
        "uncertain_pixel_fraction": float(uncertain.mean()),
        "log1p_component_count": float(np.log1p(component_count)),
        "largest_component_fraction": largest_component_fraction,
        "bounding_box_fill_ratio": float(area / box_area),
        "normalized_perimeter": float(perimeter / np.sqrt(area)),
        "border_touch_fraction": float((hard & border).sum() / area),
    }


def make_model():
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
                    random_state=SEED,
                ),
            ),
        ]
    )


def cross_fit(features, success, assignment, rows):
    scores = np.empty(len(rows), dtype=np.float64)
    baseline_scores = np.empty(len(rows), dtype=np.float64)
    coefficients = []
    folds = []
    groups = np.asarray([row["similarity_group"] for row in rows])
    for fold in range(FOLDS):
        evaluation = assignment == fold
        fitting = ~evaluation
        if len(np.unique(success[fitting])) != 2 or len(np.unique(success[evaluation])) != 2:
            raise ValueError(f"Fold {fold + 1} lacks reliability target diversity")
        model = make_model()
        model.fit(features[fitting], success[fitting])
        scores[evaluation] = model.predict_proba(features[evaluation])[:, 1]
        fitting_failure_prevalence = float((~success[fitting]).mean())
        baseline_scores[evaluation] = 1.0 - fitting_failure_prevalence
        coefficient = model.named_steps["logistic"].coef_[0]
        if not np.isfinite(coefficient).all():
            raise ValueError("Non-finite logistic regression coefficient")
        coefficients.append(coefficient)
        folds.append(
            {
                "fold": fold + 1,
                "fit_cases": int(fitting.sum()),
                "evaluation_cases": int(evaluation.sum()),
                "fit_similarity_groups": len(set(groups[fitting])),
                "evaluation_similarity_groups": len(set(groups[evaluation])),
                "fit_failure_prevalence": fitting_failure_prevalence,
                "evaluation_failures": int((~success[evaluation]).sum()),
                "evaluation_zero_dice": 0,
                "intercept": float(model.named_steps["logistic"].intercept_[0]),
                "iterations": int(model.named_steps["logistic"].n_iter_[0]),
            }
        )
    if not np.isfinite(scores).all() or not np.isfinite(baseline_scores).all():
        raise ValueError("Incomplete OOF reliability or baseline scores")
    return scores, baseline_scores, np.asarray(coefficients), folds


def coefficient_summary(coefficient_matrix):
    output = {}
    for index, name in enumerate(FEATURES):
        values = coefficient_matrix[:, index]
        signs = np.sign(values)
        positive = int((signs > 0).sum())
        negative = int((signs < 0).sum())
        zero = int((signs == 0).sum())
        dominant_count = max(positive, negative, zero)
        if positive == dominant_count:
            dominant = "positive"
        elif negative == dominant_count:
            dominant = "negative"
        else:
            dominant = "zero"
        output[name] = {
            "fold_coefficients": [float(value) for value in values],
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "positive_folds": positive,
            "negative_folds": negative,
            "zero_folds": zero,
            "dominant_sign": dominant,
            "sign_agreement_fraction": dominant_count / FOLDS,
            "all_folds_same_sign": dominant_count == FOLDS,
        }
    return output


def reliability_metrics(reliability, baseline_reliability, success):
    reliability = np.asarray(reliability, dtype=np.float64)
    baseline_reliability = np.asarray(baseline_reliability, dtype=np.float64)
    success_float = np.asarray(success, dtype=np.float64)
    bins = classifier_calibration.equal_frequency_bins(
        reliability, success_float, bins=10
    )
    ece, mce = classifier_calibration.calibration_error(bins)
    brier = float(np.mean(np.square(reliability - success_float)))
    baseline_brier = float(
        np.mean(np.square(baseline_reliability - success_float))
    )
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


def bootstrap_metrics(risk, baseline_risk, failure, groups):
    groups = np.asarray(groups)
    unique_groups = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    auroc_samples = []
    brier_skill_samples = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        target = failure[indices]
        if len(np.unique(target)) != 2:
            raise ValueError("A group-bootstrap sample lacks both reliability targets")
        auroc_samples.append(float(roc_auc_score(target, risk[indices])))
        model_brier = float(np.mean(np.square(risk[indices] - target)))
        baseline_brier = float(
            np.mean(np.square(baseline_risk[indices] - target))
        )
        if baseline_brier <= 0:
            raise ValueError("Bootstrap baseline Brier is non-positive")
        brier_skill_samples.append(1.0 - model_brier / baseline_brier)
    return {
        "auroc": {
            "bootstrap_mean": float(np.mean(auroc_samples)),
            "ci95_lower": float(np.percentile(auroc_samples, 2.5)),
            "ci95_upper": float(np.percentile(auroc_samples, 97.5)),
        },
        "brier_skill": {
            "bootstrap_mean": float(np.mean(brier_skill_samples)),
            "ci95_lower": float(np.percentile(brier_skill_samples, 2.5)),
            "ci95_upper": float(np.percentile(brier_skill_samples, 97.5)),
        },
    }


def top_risk_indices(risk, rows, count=TOP_RISK_CASES):
    return np.asarray(
        sorted(
            range(len(rows)),
            key=lambda index: (-risk[index], normalized_path(rows[index]["path"])),
        )[:count],
        dtype=np.int64,
    )


def per_category_metrics(rows, reliability, baseline, success, dice, top_indices):
    labels = np.asarray([row["label"] for row in rows])
    in_top = np.zeros(len(rows), dtype=bool)
    in_top[top_indices] = True
    output = {}
    for category in segmenter.LABELS:
        indices = np.flatnonzero(labels == category)
        failure = ~success[indices]
        risk = 1.0 - reliability[indices]
        metrics = reliability_metrics(
            reliability[indices], baseline[indices], success[indices]
        )
        output[category] = {
            "count": len(indices),
            "failures": int(failure.sum()),
            "zero_dice": int((dice[indices] == 0.0).sum()),
            "failure_detection_auroc": float(roc_auc_score(failure, risk)),
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "brier": metrics["brier"],
            "prevalence_baseline_brier": metrics["prevalence_baseline_brier"],
            "brier_skill": metrics["brier_skill"],
            "cases_in_global_top20_risk": int(in_top[indices].sum()),
            "failures_in_global_top20_risk": int((in_top[indices] & failure).sum()),
            "zero_dice_in_global_top20_risk": int(
                (in_top[indices] & (dice[indices] == 0.0)).sum()
            ),
        }
    return output


def render_diagram(path, metrics):
    bins = metrics["bins"]
    x = [item["mean_confidence"] for item in bins]
    y = [item["observed_accuracy"] for item in bins]
    sizes = [max(35, item["count"] * 1.1) for item in bins]
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "--", color="#555555", label="ideal")
    axis.plot(x, y, color="#4c72b0", linewidth=1.7)
    axis.scatter(x, y, s=sizes, color="#4c72b0", edgecolor="white", linewidth=0.7)
    axis.set_xlim(-0.005, 1.005)
    axis.set_ylim(-0.005, 1.005)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Mean predicted reliability")
    axis.set_ylabel("Observed Dice ≥ 0.50 frequency")
    axis.set_title(
        "OOF case-level segmentation reliability\n"
        f"10 equal-frequency bins; ECE={metrics['ece']:.4f}, "
        f"Brier={metrics['brier']:.4f}"
    )
    axis.grid(alpha=0.22)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_report(result):
    metrics = result["metrics"]
    lines = [
        "# Segmenter case-level reliability",
        "",
        "An L2-regularized logistic regression was evaluated through five-fold similarity-aware cross-fitting. Features use only the raw segmenter probability map and unchanged 0.50 hard mask. Reference masks were used only to create and evaluate the Dice-based target.",
        "",
        "## Performance",
        "",
        f"- Failure-detection AUROC: `{metrics['failure_detection_auroc']:.6f}` (95% group-bootstrap interval `[{metrics['bootstrap']['auroc']['ci95_lower']:.6f}, {metrics['bootstrap']['auroc']['ci95_upper']:.6f}]`)",
        f"- Zero-Dice cases captured in top 20% risk: `{metrics['top20_risk']['zero_dice_captured']}/{metrics['top20_risk']['zero_dice_total']}`",
        f"- All Dice-below-0.50 failures captured in top 20% risk: `{metrics['top20_risk']['failures_captured']}/{metrics['top20_risk']['failures_total']}`",
        f"- Reliability ECE: `{metrics['reliability']['ece']:.6f}`",
        f"- Reliability Brier: `{metrics['reliability']['brier']:.6f}`",
        f"- Fold-prevalence baseline Brier: `{metrics['reliability']['prevalence_baseline_brier']:.6f}`",
        f"- Brier skill: `{metrics['reliability']['brier_skill']:.6f}` (95% group-bootstrap interval `[{metrics['bootstrap']['brier_skill']['ci95_lower']:.6f}, {metrics['bootstrap']['brier_skill']['ci95_upper']:.6f}]`)",
        "",
        "## Standardized coefficients",
        "",
        "Positive coefficients increase predicted reliability; negative coefficients increase predicted failure risk.",
        "",
        "| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | SD | Sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in FEATURES:
        item = result["coefficients"]["features"][feature]
        folds = item["fold_coefficients"]
        lines.append(
            f"| {feature} | {folds[0]:+.4f} | {folds[1]:+.4f} | "
            f"{folds[2]:+.4f} | {folds[3]:+.4f} | {folds[4]:+.4f} | "
            f"{item['mean']:+.4f} | {item['sample_sd']:.4f} | "
            f"{item['dominant_sign']} {item['sign_agreement_fraction']:.0%} |"
        )
    lines.extend(["", "## Gate", "", f"**{result['gate']['status']}**", ""])
    for name, passed in result["gate"]["checks"].items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            f"Reliability diagram: `{result['artifacts']['reliability_diagram']}`",
            "",
            "No base model was changed during reliability fitting, no test image or test prediction was accessed, the segmentation threshold remained 0.50, and no application integration was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config = read_json(RUN / "config.json")
    result = read_json(RUN / "result.json")
    findings = read_json(pixel_calibration.FINDINGS)
    pixel_result = read_json(PIXEL_CALIBRATION)
    if pixel_result["gate"]["status"] not in {"PASS", "FAIL"}:
        raise ValueError("Pixel calibration lacks a final gate decision")
    split = RUN / "split_snapshot.csv"
    if sha256(split) != config["split_sha256"]:
        raise ValueError("Segmenter split SHA256 mismatch")
    expected_checkpoint = findings["top3_checkpoints"][0]
    if sha256(CHECKPOINT).upper() != expected_checkpoint["sha256"]:
        raise ValueError("Accepted segmenter checkpoint SHA256 mismatch")
    parts = segmenter.partition_masked_rows(smoke.read_csv(split))
    rows = parts["validation"]
    if len(rows) != 680:
        raise ValueError("Expected 680 segmentation-validation cases")
    verified_sources = pixel_calibration.verify_validation_sources(rows)
    assignment = pixel_calibration.load_fold_assignment(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != result["best_epoch"] or checkpoint["config"] != config:
        raise ValueError("Checkpoint metadata differs from accepted run")
    model.load_state_dict(checkpoint["model_state"])
    logits, targets, dice, iou, _labels, recomputed = pixel_calibration.infer(
        model, rows, device
    )
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_metric_difference = max(
        abs(recomputed[key] - result["best_validation_metrics"][key])
        for key in result["best_validation_metrics"]
        if key != "val_loss"
    )
    if maximum_metric_difference > tolerance:
        raise ValueError(f"Recomputed metric mismatch: {maximum_metric_difference}")
    maximum_per_image_difference = pixel_calibration.verify_prior_predictions(
        rows, dice, iou, targets, logits, tolerance
    )
    probabilities = expit(logits.astype(np.float64))
    feature_rows = [prediction_features(probability) for probability in probabilities]
    if any(tuple(row) != FEATURES for row in feature_rows):
        raise ValueError("Feature order differs from the predeclared feature list")
    feature_matrix = np.asarray(
        [[row[name] for name in FEATURES] for row in feature_rows], dtype=np.float64
    )
    if feature_matrix.shape != (680, len(FEATURES)) or not np.isfinite(feature_matrix).all():
        raise ValueError("Invalid case reliability feature matrix")

    success = dice >= FAILURE_DICE_THRESHOLD
    failure = ~success
    zero_dice = dice == 0.0
    if not failure.any() or not success.any() or not zero_dice.any():
        raise ValueError("Reliability target lacks required outcome diversity")
    required_zero_capture = int(zero_dice.sum()) // 2 + 1
    reliability, baseline_reliability, coefficient_matrix, folds = cross_fit(
        feature_matrix, success, assignment, rows
    )
    for fold in folds:
        member = assignment == fold["fold"] - 1
        fold["evaluation_zero_dice"] = int(zero_dice[member].sum())
    risk = 1.0 - reliability
    baseline_risk = 1.0 - baseline_reliability
    auroc = float(roc_auc_score(failure, risk))
    calibration = reliability_metrics(reliability, baseline_reliability, success)
    bootstrap = bootstrap_metrics(
        risk, baseline_risk, failure, [row["similarity_group"] for row in rows]
    )
    top_indices = top_risk_indices(risk, rows)
    zero_captured = int(zero_dice[top_indices].sum())
    failures_captured = int(failure[top_indices].sum())
    coefficients = coefficient_summary(coefficient_matrix)
    per_category = per_category_metrics(
        rows, reliability, baseline_reliability, success, dice, top_indices
    )

    checks = {
        "failure_detection_auroc_at_least_0_75": auroc >= AUROC_REQUIRED,
        "top20_risk_captures_more_than_half_of_zero_dice": (
            zero_captured >= required_zero_capture
        ),
        "reliability_ece_at_most_0_05": calibration["ece"] <= ECE_REQUIRED,
        "brier_skill_positive": calibration["brier_skill"] > 0.0,
        "brier_skill_bootstrap_ci95_lower_above_zero": (
            bootstrap["brier_skill"]["ci95_lower"] > 0.0
        ),
    }
    gate_passed = all(checks.values())
    render_diagram(DIAGRAM_PATH, calibration)

    oof_rows = []
    for index, row in enumerate(rows):
        record = {
            "path": row["path"],
            "label": row["label"],
            "similarity_group": row["similarity_group"],
            "fold": int(assignment[index]) + 1,
            "dice": float(dice[index]),
            "success_dice_at_least_0_50": int(success[index]),
            "failure_dice_below_0_50": int(failure[index]),
            "zero_dice": int(zero_dice[index]),
            "oof_reliability_score": float(reliability[index]),
            "oof_risk_score": float(risk[index]),
            "fold_training_success_prevalence_baseline": float(
                baseline_reliability[index]
            ),
            **feature_rows[index],
        }
        oof_rows.append(record)
    write_csv(OOF_PATH, oof_rows)

    coefficient_rows = []
    for feature_index, feature in enumerate(FEATURES):
        summary = coefficients[feature]
        coefficient_rows.append(
            {
                "feature": feature,
                **{
                    f"fold_{fold + 1}_standardized_coefficient": coefficient_matrix[
                        fold, feature_index
                    ]
                    for fold in range(FOLDS)
                },
                "mean": summary["mean"],
                "sample_sd": summary["sample_sd"],
                "dominant_sign": summary["dominant_sign"],
                "sign_agreement_fraction": summary["sign_agreement_fraction"],
                "all_folds_same_sign": summary["all_folds_same_sign"],
            }
        )
    write_csv(COEFFICIENT_PATH, coefficient_rows)
    ranked = sorted(
        top_indices,
        key=lambda index: (-risk[index], normalized_path(rows[index]["path"])),
    )
    write_csv(
        TOP_RISK_PATH,
        [
            {
                "risk_rank": rank,
                "path": rows[index]["path"],
                "label": rows[index]["label"],
                "dice": float(dice[index]),
                "zero_dice": int(zero_dice[index]),
                "failure_dice_below_0_50": int(failure[index]),
                "oof_reliability_score": float(reliability[index]),
                "oof_risk_score": float(risk[index]),
            }
            for rank, index in enumerate(ranked, start=1)
        ],
    )

    final_model = make_model()
    final_model.fit(feature_matrix, success)
    final_parameters = {
        "status": "proposed operational model only",
        "used_for_reported_metrics": False,
        "eligible_for_application_integration": gate_passed,
        "standardizer_mean": {
            name: float(value)
            for name, value in zip(
                FEATURES, final_model.named_steps["scale"].mean_, strict=True
            )
        },
        "standardizer_scale": {
            name: float(value)
            for name, value in zip(
                FEATURES, final_model.named_steps["scale"].scale_, strict=True
            )
        },
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(
                FEATURES,
                final_model.named_steps["logistic"].coef_[0],
                strict=True,
            )
        },
        "intercept": float(final_model.named_steps["logistic"].intercept_[0]),
    }
    group_counts = Counter(row["similarity_group"] for row in rows)
    output = {
        "schema_version": 1,
        "status": "SEGMENTER_CASE_RELIABILITY_COMPLETE",
        "scope": "Internal BRISC validation reliability experiment; not clinical validation",
        "target": {
            "model_output": "P(Dice >= 0.50)",
            "success_definition": "hard-mask Dice >= 0.50",
            "failure_definition": "hard-mask Dice < 0.50",
            "success_cases": int(success.sum()),
            "failure_cases": int(failure.sum()),
            "zero_dice_cases": int(zero_dice.sum()),
            "reference_mask_use": (
                "target creation and evaluation only; no reference-derived model feature"
            ),
        },
        "model": {
            "type": "L2-regularized logistic regression",
            "C": 1.0,
            "class_weight": None,
            "solver": "lbfgs",
            "feature_standardization": "fit independently inside each outer training fold",
            "hyperparameter_search": False,
            "post_hoc_probability_calibration": False,
            "feature_source": "raw segmenter probability map and unchanged 0.50 hard mask only",
            "excluded_sources": [
                "reference-mask geometry or pixels",
                "Dice or IoU",
                "true tumor type",
                "patient or similarity-group identity",
                "filename or path",
                "dataset metadata",
                "input-image intensity",
                "classifier probabilities",
                "rejected pixel-calibrated segmenter probabilities",
            ],
            "features": list(FEATURES),
        },
        "cross_fitting": {
            "folds": folds,
            "assignment": (
                "exact five-fold path-to-fold mapping reused from the similarity-aware "
                "classifier calibration"
            ),
            "reported_scores": "out-of-fold only",
        },
        "validation_cases": len(rows),
        "similarity_groups": len(group_counts),
        "multi_image_similarity_groups": sum(value > 1 for value in group_counts.values()),
        "largest_similarity_group": max(group_counts.values()),
        "metrics": {
            "failure_detection_auroc": auroc,
            "reliability": calibration,
            "bootstrap": bootstrap,
            "top20_risk": {
                "case_count": TOP_RISK_CASES,
                "population_fraction": TOP_RISK_CASES / len(rows),
                "tie_break": "descending risk, then normalized path ascending",
                "zero_dice_captured": zero_captured,
                "zero_dice_total": int(zero_dice.sum()),
                "failures_captured": failures_captured,
                "failures_total": int(failure.sum()),
            },
            "per_category": per_category,
        },
        "coefficients": {
            "meaning": (
                "standardized coefficients; positive increases predicted reliability, "
                "negative increases predicted failure risk"
            ),
            "features": coefficients,
            "fold_intercepts": [fold["intercept"] for fold in folds],
        },
        "final_all_validation_model": final_parameters,
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "All checks must pass; there is no metric tie-break.",
            "on_failure": "Document and close; do not attempt model variants.",
        },
        "verification": {
            "checkpoint_sha256": sha256(CHECKPOINT),
            "split_sha256": sha256(split),
            "reused_fold_assignments_sha256": sha256(
                pixel_calibration.CLASSIFIER_FOLDS
            ),
            "raw_validation_predictions_sha256": sha256(
                RUN / "validation_predictions.csv"
            ),
            "pixel_calibration_sha256": sha256(PIXEL_CALIBRATION),
            "source_images_and_masks_verified_against_manifest": verified_sources,
            "maximum_recomputed_metric_difference": maximum_metric_difference,
            "maximum_recomputed_per_image_difference": maximum_per_image_difference,
            "feature_matrix_shape": list(feature_matrix.shape),
            "feature_matrix_all_finite": True,
            "inference_device": str(device),
            "base_model_is_retrained_candidate": "segmentation_v2" in str(RUN),
            "case_reliability_model_fitted": True,
            "test_images_accessed": 0,
            "test_predictions_accessed": 0,
            "threshold": 0.5,
            "threshold_changed": False,
            "application_integration_performed": False,
            "script_sha256": sha256(Path(__file__)),
        },
        "artifacts": {
            "oof_scores": str(OOF_PATH),
            "coefficient_table": str(COEFFICIENT_PATH),
            "top20_risk_cases": str(TOP_RISK_PATH),
            "reliability_diagram": str(DIAGRAM_PATH),
        },
        "limitations": [
            "The base segmenter checkpoint was selected using this validation partition.",
            "Similarity groups are not verified patient identifiers.",
            "Patient independence and cross-dataset generalization remain unverified.",
            "Reference masks are required to create the development target but are not model features.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(markdown_report(output), encoding="utf-8")

    consolidated = read_json(CALIBRATION_FINDINGS)
    consolidated["segmenter_case_reliability"] = {
        "status": output["gate"]["status"],
        "result_path": str(OUTPUT_JSON),
        "failure_detection_auroc": auroc,
        "zero_dice_captured_top20_risk": zero_captured,
        "zero_dice_total": int(zero_dice.sum()),
        "reliability_ece": calibration["ece"],
        "brier_skill": calibration["brier_skill"],
        "accepted_for_application": gate_passed,
        "on_failure": "No variants; subsection closed.",
    }
    CALIBRATION_FINDINGS.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=config["mlflow_run_id"]):
        mlflow.log_params(
            {
                "case_reliability_model": "l2_logistic_regression",
                "case_reliability_features": len(FEATURES),
                "case_reliability_folds": FOLDS,
                "case_reliability_target_dice": FAILURE_DICE_THRESHOLD,
            }
        )
        mlflow.log_metrics(
            {
                "case_reliability_auroc": auroc,
                "case_reliability_ece": calibration["ece"],
                "case_reliability_brier": calibration["brier"],
                "case_reliability_brier_skill": calibration["brier_skill"],
                "case_reliability_zero_dice_top20": zero_captured,
                "case_reliability_gate_pass": float(gate_passed),
            }
        )
        for artifact in (
            OUTPUT_JSON,
            OUTPUT_MD,
            CALIBRATION_FINDINGS,
            OOF_PATH,
            COEFFICIENT_PATH,
            TOP_RISK_PATH,
            DIAGRAM_PATH,
        ):
            mlflow.log_artifact(str(artifact))

    print(f"AUROC: {auroc:.8f}")
    print(
        "AUROC bootstrap 95% CI: "
        f"[{bootstrap['auroc']['ci95_lower']:.8f}, "
        f"{bootstrap['auroc']['ci95_upper']:.8f}]"
    )
    print(f"Zero-Dice top-20% capture: {zero_captured}/{int(zero_dice.sum())}")
    print(
        f"ECE {calibration['ece']:.8f}; Brier {calibration['brier']:.8f}; "
        f"baseline {calibration['prevalence_baseline_brier']:.8f}; "
        f"skill {calibration['brier_skill']:.8f}"
    )
    print(
        "Brier skill bootstrap 95% CI: "
        f"[{bootstrap['brier_skill']['ci95_lower']:.8f}, "
        f"{bootstrap['brier_skill']['ci95_upper']:.8f}]"
    )
    print(f"Gate: {output['gate']['status']}")


if __name__ == "__main__":
    main()
