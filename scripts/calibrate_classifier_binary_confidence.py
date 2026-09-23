"""Cross-fit binary tumor/no-tumor temperature scaling for the frozen classifier."""

import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import calibrate_classifier_confidence as multiclass_calibration
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import train_classifier as classifier
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "20260921-170607-243133-baseline"
RUN = ROOT / "runs" / "classifier" / RUN_NAME
FINDINGS_PATH = ROOT / "audit" / "classifier_training_findings.json"
MULTICLASS_CALIBRATION_PATH = ROOT / "audit" / "classifier_calibration.json"
AUDIT_INVENTORY_PATH = (
    ROOT / "audit" / "runs" / "20260921-182736-884368" / "inventory.csv"
)
SIMILARITY_CANDIDATES_PATH = (
    ROOT
    / "audit"
    / "runs"
    / "20260921-182736-884368"
    / "similarity_candidates.csv"
)
EXISTING_FOLDS_PATH = RUN / "classifier_calibration_folds.csv"
OUTPUT_JSON = ROOT / "audit" / "classifier_binary_calibration.json"
OUTPUT_MARKDOWN = ROOT / "audit" / "classifier_binary_calibration.md"
DIAGRAM_PATH = RUN / "classifier_binary_reliability_diagram.png"
FOLD_RESULTS_PATH = RUN / "classifier_binary_calibration_fold_results.csv"
OOF_PATH = RUN / "classifier_binary_calibration_oof_predictions.csv"
CLASSES = multiclass_calibration.CLASSES
NO_TUMOR_INDEX = CLASSES.index("no_tumor")
FOLDS = multiclass_calibration.FOLDS
SEED = multiclass_calibration.SEED
BOOTSTRAP_REPETITIONS = multiclass_calibration.BOOTSTRAP_REPETITIONS


def binary_tumor_margin(logits):
    """Collapse four logits into log-odds for tumor versus no tumor."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(CLASSES):
        raise ValueError("Expected an N x 4 logit matrix")
    margin = logsumexp(logits[:, :NO_TUMOR_INDEX], axis=1) - logits[:, NO_TUMOR_INDEX]
    if not np.isfinite(margin).all():
        raise ValueError("Non-finite binary tumor margin")
    return margin


def binary_temperature_probabilities(margins, temperature):
    """Return P(tumor) after scaling the collapsed binary margin."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be finite and positive")
    probabilities = expit(np.asarray(margins, dtype=np.float64) / temperature)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite calibrated binary probability")
    return probabilities


def binary_nll(probabilities, targets):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-15, 1.0 - 1e-15
    )
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != targets.shape:
        raise ValueError("Binary probability and target shapes differ")
    return float(
        -np.mean(
            targets * np.log(probabilities)
            + (1.0 - targets) * np.log(1.0 - probabilities)
        )
    )


def fit_binary_temperature(margins, targets):
    """Fit one positive temperature using binary tumor/no-tumor NLL only."""

    def objective(log_temperature):
        probabilities = binary_temperature_probabilities(
            margins, np.exp(log_temperature)
        )
        return binary_nll(probabilities, targets)

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Binary temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    if not 0.05 <= temperature <= 20.0:
        raise ValueError("Fitted binary temperature left its predeclared bounds")
    return temperature


def load_existing_fold_assignment(records):
    """Reuse and validate the accepted multiclass calibration fold assignment."""
    rows = multiclass_calibration.load_csv(EXISTING_FOLDS_PATH)
    if len(rows) != len(records):
        raise ValueError("Existing calibration fold count differs from validation count")
    by_path = {}
    for row in rows:
        if row["path"] in by_path:
            raise ValueError(f"Duplicate existing fold path: {row['path']}")
        by_path[row["path"]] = row
    assignment = np.full(len(records), -1, dtype=np.int64)
    for index, record in enumerate(records):
        existing = by_path.get(record["path"])
        if existing is None:
            raise ValueError(f"Validation path missing from existing folds: {record['path']}")
        if (
            existing["label"] != record["label"]
            or existing["similarity_group"] != record["similarity_group"]
        ):
            raise ValueError(f"Existing fold metadata mismatch: {record['path']}")
        fold = int(existing["fold"]) - 1
        if not 0 <= fold < FOLDS:
            raise ValueError(f"Invalid existing fold number: {existing['fold']}")
        assignment[index] = fold
    if set(by_path) != {record["path"] for record in records}:
        raise ValueError("Existing folds contain paths outside frozen validation")
    group_folds = {}
    for record, fold in zip(records, assignment, strict=True):
        group_folds.setdefault(record["similarity_group"], set()).add(int(fold))
    leaked = {group: folds for group, folds in group_folds.items() if len(folds) != 1}
    if leaked:
        raise ValueError(f"Similarity groups cross reused folds: {leaked}")
    if set(assignment) != set(range(FOLDS)):
        raise ValueError("Reused fold assignment does not contain all five folds")
    return assignment


def cross_fit_binary(margins, targets, records, fold_assignment):
    calibrated = np.empty(len(margins), dtype=np.float64)
    results = []
    groups = np.asarray([record["similarity_group"] for record in records])
    labels = np.asarray([record["label"] for record in records])
    for fold in range(FOLDS):
        evaluation = fold_assignment == fold
        fitting = ~evaluation
        if set(targets[evaluation]) != {False, True}:
            raise ValueError(f"Fold {fold + 1} does not contain both binary classes")
        temperature = fit_binary_temperature(margins[fitting], targets[fitting])
        calibrated[evaluation] = binary_temperature_probabilities(
            margins[evaluation], temperature
        )
        results.append(
            {
                "fold": fold + 1,
                "temperature": temperature,
                "fit_cases": int(fitting.sum()),
                "evaluation_cases": int(evaluation.sum()),
                "fit_similarity_groups": len(set(groups[fitting])),
                "evaluation_similarity_groups": len(set(groups[evaluation])),
                "evaluation_tumor_cases": int(targets[evaluation].sum()),
                "evaluation_no_tumor_cases": int((~targets[evaluation]).sum()),
                "evaluation_class_counts": dict(
                    sorted(Counter(labels[evaluation]).items())
                ),
            }
        )
    if not np.isfinite(calibrated).all():
        raise ValueError("Cross-fitted binary probabilities are incomplete")
    return calibrated, results


def bootstrap_metric_deltas(raw, calibrated, targets, groups):
    groups = np.asarray(groups)
    unique_groups = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in ("nll", "brier", "ece", "mce")}
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled_groups])
        raw_metrics = multiclass_calibration.binary_metrics(raw[indices], targets[indices])
        calibrated_metrics = multiclass_calibration.binary_metrics(
            calibrated[indices], targets[indices]
        )
        for metric, samples in values.items():
            samples.append(calibrated_metrics[metric] - raw_metrics[metric])
    return {
        metric: {
            "point_estimate": float(
                multiclass_calibration.binary_metrics(calibrated, targets)[metric]
                - multiclass_calibration.binary_metrics(raw, targets)[metric]
            ),
            "bootstrap_mean": float(np.mean(samples)),
            "ci95_lower": float(np.percentile(samples, 2.5)),
            "ci95_upper": float(np.percentile(samples, 97.5)),
        }
        for metric, samples in values.items()
    }


def render_reliability_diagram(path, raw_metrics, calibrated_metrics):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, title, metrics, color in (
        (axes[0], "Raw binary probability", raw_metrics, "#c44e52"),
        (
            axes[1],
            "Cross-fitted binary temperature scaling",
            calibrated_metrics,
            "#4c72b0",
        ),
    ):
        bins = metrics["bins"]
        x = [item["mean_confidence"] for item in bins]
        y = [item["observed_accuracy"] for item in bins]
        sizes = [max(28, item["count"] * 0.8) for item in bins]
        axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="ideal")
        axis.plot(x, y, color=color, linewidth=1.5)
        axis.scatter(x, y, s=sizes, color=color, edgecolor="white", linewidth=0.7)
        axis.set_xlim(-0.005, 1.005)
        axis.set_ylim(-0.005, 1.005)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Mean predicted tumor probability")
        axis.set_ylabel("Observed tumor frequency")
        axis.set_title(
            f"{title}\nECE={metrics['ece']:.4f}, NLL={metrics['nll']:.4f}"
        )
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Tumor/no-tumor reliability on 893 internal-validation images\n"
        "Equal-frequency bins; calibrated panel uses only out-of-fold probabilities",
        fontsize=13,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_report(result):
    raw = result["metrics"]["raw"]
    calibrated = result["metrics"]["cross_fitted_temperature_scaled"]
    bootstrap = result["metrics"]["bootstrap_paired_deltas"]
    lines = [
        "# Classifier binary tumor/no-tumor calibration",
        "",
        "One temperature was fitted to the collapsed binary tumor margin using binary tumor/no-tumor NLL. Four-class NLL was not used. Reported calibrated metrics come only from held-out folds.",
        "",
        "## Fitted temperatures",
        "",
        "| Fold | Temperature | Fit cases | Evaluation cases | Evaluation groups |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in result["folds"]:
        lines.append(
            f"| {item['fold']} | {item['temperature']:.8f} | {item['fit_cases']} | "
            f"{item['evaluation_cases']} | {item['evaluation_similarity_groups']} |"
        )
    lines.extend(
        [
            "",
            (
                "Proposed all-validation operational temperature: "
                f"`{result['final_temperature_all_validation']:.8f}`. This value was fitted "
                "on all 893 validation cases and was **not** used for any reported metric."
            ),
            "",
            "## Binary calibration metrics",
            "",
            "| Metric | Raw | Cross-fitted calibrated | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("nll", "Binary NLL"),
        ("brier", "Binary Brier"),
        ("ece", "Binary ECE"),
        ("mce", "Binary MCE"),
    ):
        lines.append(
            f"| {label} | {raw[key]:.8f} | {calibrated[key]:.8f} | "
            f"{calibrated[key] - raw[key]:+.8f} |"
        )
    lines.extend(
        [
            "",
            "## Paired similarity-group bootstrap",
            "",
            "Calibrated-minus-raw changes from 2,000 paired resamples:",
            "",
            "| Metric | Point change | Bootstrap mean | 95% interval |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("nll", "Binary NLL"),
        ("brier", "Binary Brier"),
        ("ece", "Binary ECE"),
        ("mce", "Binary MCE"),
    ):
        item = bootstrap[key]
        lines.append(
            f"| {label} | {item['point_estimate']:+.8f} | "
            f"{item['bootstrap_mean']:+.8f} | "
            f"[{item['ci95_lower']:+.8f}, {item['ci95_upper']:+.8f}] |"
        )
    lines.extend(
        [
            "",
            "## Gate result",
            "",
            f"**{result['gate']['status']}**",
            "",
        ]
    )
    for name, value in result["gate"]["checks"].items():
        lines.append(f"- `{name}`: **{'PASS' if value else 'FAIL'}**")
    if result["gate"]["status"] == "PASS":
        lines.extend(
            [
                "",
                "The proposed all-validation binary temperature is eligible for later application integration. Four-class probabilities remain raw because multiclass temperature scaling failed its gate.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The binary temperature is rejected. The classifier remains closed with raw probabilities for both four-class and tumor/no-tumor outputs.",
            ]
        )
    lines.extend(
        [
            "",
            f"Reliability diagram: `{result['artifacts']['reliability_diagram']}`",
            "",
            "## Limits",
            "",
            "This is post-hoc evaluation on BRISC internal validation. The checkpoint was previously selected with this partition, the grouping is similarity-aware rather than patient-aware, and patient independence and cross-dataset calibration remain unverified.",
            "",
        ]
    )
    return "\n".join(lines)


def recompute_logits(model, rows, device):
    loader = DataLoader(
        classifier.BrainImages(rows, training=False),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    labels = []
    logits = []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                output = model(images)
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite classifier logit")
            logits.append(output.float().cpu().numpy())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels, dtype=np.int64), np.concatenate(logits).astype(np.float64)


def main():
    config_path = RUN / "config.json"
    split_path = RUN / "split_snapshot.csv"
    predictions_path = RUN / "validation_predictions.csv"
    checkpoint_path = RUN / "best_model.pt"
    config = multiclass_calibration.load_json(config_path)
    findings = multiclass_calibration.load_json(FINDINGS_PATH)
    prior_calibration = multiclass_calibration.load_json(MULTICLASS_CALIBRATION_PATH)

    if tuple(config["classes"]) != CLASSES:
        raise ValueError("Classifier class order changed")
    if config["test_evaluated"] or findings["test_evaluated"]:
        raise ValueError("A source artifact reports test evaluation")
    if multiclass_calibration.sha256(split_path) != findings["split_sha256"]:
        raise ValueError("Classifier split SHA256 mismatch")
    if multiclass_calibration.sha256(AUDIT_INVENTORY_PATH) != multiclass_calibration.sha256(
        split_path
    ):
        raise ValueError("Frozen split is not byte-identical to accepted audit inventory")
    if multiclass_calibration.sha256(checkpoint_path) != findings["checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint SHA256 mismatch")
    if prior_calibration["gate"]["status"] != "FAIL":
        raise ValueError("Expected the accepted multiclass calibration gate to be failed")
    if prior_calibration["decision_rules"]["temperature_accepted_for_application"]:
        raise ValueError("Multiclass temperature was unexpectedly accepted")

    split_rows = multiclass_calibration.load_csv(split_path)
    validation_metadata = {
        row["path"]: row for row in split_rows if row["draft_split"] == "validation"
    }
    prediction_rows = multiclass_calibration.load_csv(predictions_path)
    if len(validation_metadata) != 893 or len(prediction_rows) != 893:
        raise ValueError("Expected exactly 893 classifier-validation cases")
    if len({row["path"] for row in prediction_rows}) != 893:
        raise ValueError("Duplicate classifier-validation prediction path")

    records = []
    stored_probabilities = []
    labels = []
    ordered_metadata = []
    for prediction in sorted(prediction_rows, key=lambda row: row["path"].casefold()):
        path = prediction["path"]
        if path not in validation_metadata:
            raise ValueError(f"Prediction is not in frozen validation metadata: {path}")
        metadata = validation_metadata[path]
        if prediction["label"] != metadata["label"]:
            raise ValueError(f"Prediction/split label mismatch: {path}")
        probability = np.asarray(
            [float(prediction[f"score_{name}"]) for name in CLASSES], dtype=np.float64
        )
        if prediction["predicted"] != CLASSES[int(probability.argmax())]:
            raise ValueError(f"Stored predicted class mismatch: {path}")
        stored_probabilities.append(probability)
        labels.append(CLASSES.index(prediction["label"]))
        ordered_metadata.append(metadata)
        records.append(
            {
                "path": path,
                "label": prediction["label"],
                "similarity_group": metadata["similarity_group"],
            }
        )

    for metadata in ordered_metadata:
        image = classifier.image_path(metadata)
        if multiclass_calibration.sha256(image) != metadata["sha256"]:
            raise ValueError(
                f"Classifier validation image SHA256 mismatch: {metadata['path']}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["epoch"] != findings["best_epoch"]
        or tuple(checkpoint["config"]["classes"]) != CLASSES
        or checkpoint["config"]["script_sha256"] != config["script_sha256"]
    ):
        raise ValueError("Classifier checkpoint metadata mismatch")
    model = classifier.resnet18(weights=None, num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    recomputed_labels, logits = recompute_logits(model, ordered_metadata, device)
    label_array = np.asarray(labels, dtype=np.int64)
    if not np.array_equal(recomputed_labels, label_array):
        raise ValueError("Recomputed classifier label order mismatch")

    raw_four_class = multiclass_calibration.temperature_probabilities(logits, 1.0)
    stored_probability_array = np.asarray(stored_probabilities, dtype=np.float64)
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_probability_difference = float(
        np.max(np.abs(raw_four_class - stored_probability_array))
    )
    if maximum_probability_difference > tolerance:
        raise ValueError(
            "Recomputed/stored classifier probability mismatch: "
            f"{maximum_probability_difference}"
        )
    recomputed_metrics, _ = classifier.metrics_for(label_array, raw_four_class)
    expected_metrics = multiclass_calibration.load_json(RUN / "validation_metrics.json")[
        "validation"
    ]
    maximum_metric_difference = max(
        abs(recomputed_metrics[name] - expected_metrics[name])
        for name in expected_metrics
    )
    if maximum_metric_difference > tolerance:
        raise ValueError(
            f"Recomputed classifier metric mismatch: {maximum_metric_difference}"
        )

    margins = binary_tumor_margin(logits)
    raw_binary = binary_temperature_probabilities(margins, 1.0)
    raw_from_four_class = raw_four_class[:, :NO_TUMOR_INDEX].sum(axis=1)
    maximum_binary_collapse_difference = float(
        np.max(np.abs(raw_binary - raw_from_four_class))
    )
    if maximum_binary_collapse_difference > 1e-12:
        raise ValueError(
            "Collapsed binary margin disagrees with summed tumor probability: "
            f"{maximum_binary_collapse_difference}"
        )
    targets = label_array != NO_TUMOR_INDEX
    raw_metrics = multiclass_calibration.binary_metrics(raw_binary, targets)
    prior_raw = prior_calibration["metrics"]["raw"]["tumor_binary"]
    maximum_prior_binary_metric_difference = max(
        abs(raw_metrics[name] - prior_raw[name])
        for name in ("nll", "brier", "ece", "mce")
    )
    if maximum_prior_binary_metric_difference > tolerance:
        raise ValueError(
            "Raw binary metrics disagree with accepted calibration artifact: "
            f"{maximum_prior_binary_metric_difference}"
        )

    fold_assignment = load_existing_fold_assignment(records)
    calibrated_binary, fold_results = cross_fit_binary(
        margins, targets, records, fold_assignment
    )
    final_temperature = fit_binary_temperature(margins, targets)
    calibrated_metrics = multiclass_calibration.binary_metrics(
        calibrated_binary, targets
    )
    groups = [record["similarity_group"] for record in records]
    bootstrap = bootstrap_metric_deltas(
        raw_binary, calibrated_binary, targets, groups
    )

    raw_decisions = raw_binary >= 0.5
    calibrated_decisions = calibrated_binary >= 0.5
    decisions_identical = bool(np.array_equal(raw_decisions, calibrated_decisions))
    changed_decisions = int(np.count_nonzero(raw_decisions != calibrated_decisions))
    checks = {
        "binary_nll_improved": calibrated_metrics["nll"] < raw_metrics["nll"],
        "binary_brier_improved": calibrated_metrics["brier"] < raw_metrics["brier"],
        "binary_ece_at_most_0_05": calibrated_metrics["ece"] <= 0.05,
        "brier_delta_bootstrap_ci95_upper_below_zero": bootstrap["brier"][
            "ci95_upper"
        ]
        < 0.0,
        "binary_threshold_decisions_identical": decisions_identical,
    }
    gate_passed = all(checks.values())

    render_reliability_diagram(DIAGRAM_PATH, raw_metrics, calibrated_metrics)
    multiclass_calibration.write_csv(
        FOLD_RESULTS_PATH,
        fold_results,
        [
            "fold",
            "temperature",
            "fit_cases",
            "evaluation_cases",
            "fit_similarity_groups",
            "evaluation_similarity_groups",
            "evaluation_tumor_cases",
            "evaluation_no_tumor_cases",
            "evaluation_class_counts",
        ],
    )
    oof_rows = []
    for index, record in enumerate(records):
        oof_rows.append(
            {
                **record,
                "fold": int(fold_assignment[index]) + 1,
                "target_tumor": int(targets[index]),
                "raw_tumor_probability": float(raw_binary[index]),
                "oof_calibrated_tumor_probability": float(calibrated_binary[index]),
                "raw_tumor_decision_at_0_50": int(raw_decisions[index]),
                "oof_calibrated_tumor_decision_at_0_50": int(
                    calibrated_decisions[index]
                ),
            }
        )
    multiclass_calibration.write_csv(OOF_PATH, oof_rows, list(oof_rows[0]))

    group_counts = Counter(groups)
    result = {
        "schema_version": 1,
        "status": "CLASSIFIER_BINARY_TEMPERATURE_SCALING_COMPLETE",
        "scope": "Post-hoc BRISC internal-validation calibration; not clinical validation",
        "method": {
            "name": "binary scalar temperature scaling",
            "binary_margin": "logsumexp(glioma, meningioma, pituitary logits) - no_tumor logit",
            "fit_objective": "binary tumor/no-tumor negative log-likelihood",
            "four_class_nll_used_for_fitting": False,
            "case_weighting": "every validation image weighted equally",
            "temperature_bounds": [0.05, 20.0],
            "cross_fitting": (
                "five held-out folds; exact similarity-aware fold assignment reused from "
                "the accepted multiclass calibration"
            ),
            "reported_metrics_probability_source": (
                "out-of-fold probabilities only; each case was calibrated by a temperature "
                "fitted without that case or its similarity group"
            ),
        },
        "decision_rules": {
            "all_checks_must_pass": True,
            "metric_tie_break": None,
            "temperature_accepted_for_application": gate_passed,
            "on_failure": (
                "Use raw classifier probabilities for both four-class and binary outputs"
            ),
        },
        "validation_cases": len(records),
        "tumor_cases": int(targets.sum()),
        "no_tumor_cases": int((~targets).sum()),
        "similarity_groups": len(set(groups)),
        "multi_member_similarity_groups": sum(
            count > 1 for count in group_counts.values()
        ),
        "largest_similarity_group": max(group_counts.values()),
        "folds": fold_results,
        "final_temperature_all_validation": final_temperature,
        "final_temperature_role": {
            "status": "proposed operational value only",
            "used_for_reported_metrics": False,
            "eligible_for_later_application_integration": gate_passed,
        },
        "metrics": {
            "raw": raw_metrics,
            "cross_fitted_temperature_scaled": calibrated_metrics,
            "bootstrap_paired_deltas": bootstrap,
        },
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "All checks must pass; there is no metric tie-break.",
        },
        "verification": {
            "classifier_run": RUN_NAME,
            "checkpoint_sha256": multiclass_calibration.sha256(checkpoint_path),
            "split_sha256": multiclass_calibration.sha256(split_path),
            "accepted_audit_inventory_sha256": multiclass_calibration.sha256(
                AUDIT_INVENTORY_PATH
            ),
            "similarity_candidates_sha256": multiclass_calibration.sha256(
                SIMILARITY_CANDIDATES_PATH
            ),
            "validation_predictions_sha256": multiclass_calibration.sha256(
                predictions_path
            ),
            "reused_fold_assignments_sha256": multiclass_calibration.sha256(
                EXISTING_FOLDS_PATH
            ),
            "prior_multiclass_calibration_sha256": multiclass_calibration.sha256(
                MULTICLASS_CALIBRATION_PATH
            ),
            "script_sha256": multiclass_calibration.sha256(Path(__file__)),
            "inference_device": str(device),
            "maximum_stored_probability_difference": maximum_probability_difference,
            "maximum_stored_metric_difference": maximum_metric_difference,
            "maximum_binary_collapse_difference": maximum_binary_collapse_difference,
            "maximum_prior_binary_metric_difference": (
                maximum_prior_binary_metric_difference
            ),
            "fold_groups_disjoint": True,
            "binary_decisions_changed_at_0_50": changed_decisions,
            "test_images_accessed": 0,
            "test_predictions_accessed": 0,
            "model_retrained": False,
            "threshold_changed": False,
        },
        "artifacts": {
            "reliability_diagram": str(DIAGRAM_PATH),
            "fold_results": str(FOLD_RESULTS_PATH),
            "out_of_fold_predictions": str(OOF_PATH),
        },
        "limitations": [
            "The classifier checkpoint was selected using this validation partition before calibration.",
            "The proposed all-validation temperature is not used for reported metrics.",
            "Similarity groups are not verified patient identifiers.",
            "Patient independence and cross-dataset generalization remain unverified.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUTPUT_MARKDOWN.write_text(markdown_report(result), encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=config["mlflow_run_id"]):
        mlflow.log_params(
            {
                "binary_calibration_method": "collapsed_margin_temperature_scaling",
                "binary_calibration_objective": "binary_nll",
                "binary_calibration_folds": FOLDS,
            }
        )
        mlflow.log_metrics(
            {
                "binary_calibration_proposed_temperature": final_temperature,
                "binary_calibration_raw_nll": raw_metrics["nll"],
                "binary_calibration_oof_nll": calibrated_metrics["nll"],
                "binary_calibration_raw_brier": raw_metrics["brier"],
                "binary_calibration_oof_brier": calibrated_metrics["brier"],
                "binary_calibration_raw_ece": raw_metrics["ece"],
                "binary_calibration_oof_ece": calibrated_metrics["ece"],
                "binary_calibration_gate_pass": float(gate_passed),
            }
        )
        for artifact in (
            OUTPUT_JSON,
            OUTPUT_MARKDOWN,
            DIAGRAM_PATH,
            FOLD_RESULTS_PATH,
            OOF_PATH,
        ):
            mlflow.log_artifact(str(artifact))

    print("Per-fold temperatures:")
    for item in fold_results:
        print(f"  Fold {item['fold']}: {item['temperature']:.8f}")
    print(f"Proposed all-validation temperature: {final_temperature:.8f}")
    print(
        "Raw -> OOF calibrated: "
        f"NLL {raw_metrics['nll']:.8f} -> {calibrated_metrics['nll']:.8f}; "
        f"Brier {raw_metrics['brier']:.8f} -> {calibrated_metrics['brier']:.8f}; "
        f"ECE {raw_metrics['ece']:.8f} -> {calibrated_metrics['ece']:.8f}; "
        f"MCE {raw_metrics['mce']:.8f} -> {calibrated_metrics['mce']:.8f}"
    )
    print(f"Gate: {result['gate']['status']}")


if __name__ == "__main__":
    main()
