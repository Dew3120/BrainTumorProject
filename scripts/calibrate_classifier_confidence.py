"""Cross-fit scalar temperature scaling for the frozen BRISC classifier."""

import csv
import hashlib
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import train_classifier as classifier
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "20260921-170607-243133-baseline"
RUN = ROOT / "runs" / "classifier" / RUN_NAME
FINDINGS_PATH = ROOT / "audit" / "classifier_training_findings.json"
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
OUTPUT_JSON = ROOT / "audit" / "classifier_calibration.json"
OUTPUT_MARKDOWN = ROOT / "audit" / "classifier_calibration.md"
CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")
FOLDS = 5
SEED = 42
ECE_BINS = 10
BOOTSTRAP_REPETITIONS = 2000


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def probabilities_to_logits(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASSES):
        raise ValueError("Expected an N x 4 probability matrix")
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("Stored probabilities must be strictly between zero and one")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("Stored class probabilities do not sum to one")
    normalized = probabilities / sums[:, None]
    return np.log(normalized)


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


def temperature_probabilities(logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be finite and positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled)
    return exponent / exponent.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities, labels):
    chosen = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(chosen, 1e-15, 1.0)).mean())


def multiclass_brier(probabilities, labels):
    targets = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.square(probabilities - targets).sum(axis=1).mean())


def fit_temperature(logits, labels):
    def objective(log_temperature):
        probabilities = temperature_probabilities(logits, np.exp(log_temperature))
        return negative_log_likelihood(probabilities, labels)

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    if not 0.05 <= temperature <= 20.0:
        raise ValueError("Fitted temperature left its predeclared bounds")
    return temperature


def equal_frequency_bins(confidence, correctness, bins=ECE_BINS):
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if len(confidence) != len(correctness) or len(confidence) < bins:
        raise ValueError("Insufficient or mismatched values for reliability bins")
    order = np.lexsort((np.arange(len(confidence)), confidence))
    output = []
    for index, members in enumerate(np.array_split(order, bins)):
        mean_confidence = float(confidence[members].mean())
        mean_accuracy = float(correctness[members].mean())
        count = len(members)
        output.append(
            {
                "bin": index + 1,
                "count": count,
                "minimum_confidence": float(confidence[members].min()),
                "maximum_confidence": float(confidence[members].max()),
                "mean_confidence": mean_confidence,
                "observed_accuracy": mean_accuracy,
                "absolute_gap": abs(mean_accuracy - mean_confidence),
            }
        )
    return output


def calibration_error(bins):
    total = sum(item["count"] for item in bins)
    ece = sum(item["count"] * item["absolute_gap"] for item in bins) / total
    mce = max(item["absolute_gap"] for item in bins)
    return float(ece), float(mce)


def classwise_calibration(probabilities, labels):
    output = {}
    for index, name in enumerate(CLASSES):
        bins = equal_frequency_bins(
            probabilities[:, index], (labels == index).astype(np.float64)
        )
        ece, mce = calibration_error(bins)
        output[name] = {"ece": ece, "mce": mce, "bins": bins}
    return output


def binary_metrics(probabilities, targets):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1 - 1e-15)
    targets = np.asarray(targets, dtype=np.float64)
    bins = equal_frequency_bins(probabilities, targets)
    ece, mce = calibration_error(bins)
    nll = -np.mean(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities))
    brier = np.mean(np.square(probabilities - targets))
    return {
        "nll": float(nll),
        "brier": float(brier),
        "ece": ece,
        "mce": mce,
        "bins": bins,
    }


def summarize(probabilities, labels):
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predicted == labels
    top_bins = equal_frequency_bins(confidence, correctness.astype(np.float64))
    top_ece, top_mce = calibration_error(top_bins)
    classwise = classwise_calibration(probabilities, labels)
    tumor_probability = probabilities[:, :3].sum(axis=1)
    tumor_target = labels != CLASSES.index("no_tumor")
    return {
        "multiclass_nll": negative_log_likelihood(probabilities, labels),
        "multiclass_brier": multiclass_brier(probabilities, labels),
        "accuracy": float(correctness.mean()),
        "top_label_ece": top_ece,
        "top_label_mce": top_mce,
        "top_label_bins": top_bins,
        "classwise": classwise,
        "tumor_binary": binary_metrics(tumor_probability, tumor_target),
    }


def assign_similarity_aware_folds(rows, labels, folds=FOLDS, seed=SEED):
    groups = np.asarray([row["similarity_group"] for row in rows])
    if np.any(groups == ""):
        raise ValueError("Every calibration case requires a similarity group")
    assignment = np.full(len(rows), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, evaluation) in enumerate(
        splitter.split(np.zeros((len(rows), 1)), labels, groups)
    ):
        assignment[evaluation] = fold
    if np.any(assignment < 0):
        raise ValueError("At least one calibration case received no fold")
    group_folds = {}
    for group, fold in zip(groups, assignment, strict=True):
        group_folds.setdefault(group, set()).add(int(fold))
    leaked = {group: value for group, value in group_folds.items() if len(value) != 1}
    if leaked:
        raise ValueError(f"Similarity groups cross calibration folds: {leaked}")
    return assignment


def cross_fit(logits, labels, rows):
    fold_assignment = assign_similarity_aware_folds(rows, labels)
    calibrated = np.empty_like(logits)
    fold_results = []
    groups = np.asarray([row["similarity_group"] for row in rows])
    for fold in range(FOLDS):
        evaluation = fold_assignment == fold
        fitting = ~evaluation
        temperature = fit_temperature(logits[fitting], labels[fitting])
        calibrated[evaluation] = temperature_probabilities(
            logits[evaluation], temperature
        )
        fold_results.append(
            {
                "fold": fold + 1,
                "temperature": temperature,
                "fit_cases": int(fitting.sum()),
                "evaluation_cases": int(evaluation.sum()),
                "fit_similarity_groups": len(set(groups[fitting])),
                "evaluation_similarity_groups": len(set(groups[evaluation])),
                "evaluation_class_counts": dict(
                    sorted(Counter(rows[i]["label"] for i in np.flatnonzero(evaluation)).items())
                ),
            }
        )
    return fold_assignment, calibrated, fold_results


def bootstrap_metric_deltas(raw, calibrated, labels, groups, repetitions=BOOTSTRAP_REPETITIONS):
    groups = np.asarray(groups)
    unique_groups = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    values = {"multiclass_nll": [], "multiclass_brier": [], "top_label_ece": []}
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled_groups])
        raw_summary = summarize(raw[indices], labels[indices])
        calibrated_summary = summarize(calibrated[indices], labels[indices])
        for metric, samples in values.items():
            samples.append(calibrated_summary[metric] - raw_summary[metric])
    return {
        metric: {
            "calibrated_minus_raw_mean": float(np.mean(samples)),
            "ci95_lower": float(np.percentile(samples, 2.5)),
            "ci95_upper": float(np.percentile(samples, 97.5)),
        }
        for metric, samples in values.items()
    }


def render_reliability_diagram(path, raw_summary, calibrated_summary):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, title, summary, color in (
        (axes[0], "Raw classifier confidence", raw_summary, "#c44e52"),
        (axes[1], "Cross-fitted temperature scaling", calibrated_summary, "#4c72b0"),
    ):
        bins = summary["top_label_bins"]
        x = [item["mean_confidence"] for item in bins]
        y = [item["observed_accuracy"] for item in bins]
        sizes = [max(28, item["count"] * 0.8) for item in bins]
        axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="ideal")
        axis.plot(x, y, color=color, linewidth=1.5)
        axis.scatter(x, y, s=sizes, color=color, edgecolor="white", linewidth=0.7)
        axis.set_xlim(0.45, 1.005)
        axis.set_ylim(0.45, 1.005)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Mean predicted confidence")
        axis.set_ylabel("Observed accuracy")
        axis.set_title(
            f"{title}\nECE={summary['top_label_ece']:.4f}, NLL={summary['multiclass_nll']:.4f}"
        )
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Classifier reliability on 893 internal-validation images\n"
        "Equal-frequency bins; calibrated panel uses out-of-fold probabilities",
        fontsize=13,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_report(result):
    raw = result["metrics"]["raw"]
    calibrated = result["metrics"]["cross_fitted_temperature_scaled"]
    lines = [
        "# Classifier confidence calibration",
        "",
        "Scalar temperature scaling was evaluated with five-fold, similarity-aware cross-fitting on 893 BRISC internal-validation images. No model weights were trained, no test image or test prediction was accessed, and the operational classifier decisions were not changed.",
        "",
        "## Fitted temperatures",
        "",
        f"- Final temperature fitted on all 893 validation cases: `{result['final_temperature_all_validation']:.8f}`",
        "- Out-of-fold temperatures: "
        + ", ".join(f"`{item['temperature']:.8f}`" for item in result["folds"]),
        (
            f"- Similarity groups: `{result['similarity_groups']}` total, "
            f"`{result['multi_member_similarity_groups']}` with multiple validation images, "
            f"largest group `{result['largest_similarity_group']}` images"
        ),
        "",
        "## Calibration metrics",
        "",
        "| Metric | Raw | Cross-fitted calibrated | Change |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("multiclass_nll", "Multiclass NLL"),
        ("multiclass_brier", "Multiclass Brier"),
        ("top_label_ece", "Top-label ECE"),
        ("top_label_mce", "Top-label MCE"),
        ("accuracy", "Accuracy"),
    ):
        lines.append(
            f"| {label} | {raw[key]:.6f} | {calibrated[key]:.6f} | "
            f"{calibrated[key] - raw[key]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "### Classwise ECE",
            "",
            "| Class | Raw | Cross-fitted calibrated |",
            "|---|---:|---:|",
        ]
    )
    for name in CLASSES:
        lines.append(
            f"| {name} | {raw['classwise'][name]['ece']:.6f} | "
            f"{calibrated['classwise'][name]['ece']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Tumor versus no-tumor probability calibration",
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
        raw_value = raw["tumor_binary"][key]
        calibrated_value = calibrated["tumor_binary"][key]
        lines.append(
            f"| {label} | {raw_value:.6f} | {calibrated_value:.6f} | "
            f"{calibrated_value - raw_value:+.6f} |"
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
    lines.extend(
        [
            "",
            "The primary fitting objective is multiclass NLL, but acceptance is conjunctive: an ECE above 0.05 is a failure even when NLL improves.",
            "",
            "The aggregate point estimates improved, but the paired group-bootstrap 95% interval for calibrated-minus-raw Brier score crosses zero. The conservative gate requires the upper bound to be below zero, so this fitted temperature is not accepted for application use.",
            "",
            f"- Brier change 95% interval: `[{result['metrics']['bootstrap_paired_deltas']['multiclass_brier']['ci95_lower']:.6f}, {result['metrics']['bootstrap_paired_deltas']['multiclass_brier']['ci95_upper']:.6f}]`",
            f"- Reliability diagram: `{result['artifacts']['reliability_diagram']}`",
            "",
            "## Limits",
            "",
            "These are post-hoc internal-validation results. The checkpoint was previously selected with this validation partition, known groups are similarity groups rather than patient identifiers, and patient independence is unverified. This does not establish clinical or cross-dataset calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config_path = RUN / "config.json"
    split_path = RUN / "split_snapshot.csv"
    predictions_path = RUN / "validation_predictions.csv"
    checkpoint_path = RUN / "best_model.pt"
    config = load_json(config_path)
    findings = load_json(FINDINGS_PATH)

    if tuple(config["classes"]) != CLASSES:
        raise ValueError("Classifier class order changed")
    if config["test_evaluated"] or findings["test_evaluated"]:
        raise ValueError("A source artifact reports test evaluation")
    if sha256(split_path) != findings["split_sha256"]:
        raise ValueError("Classifier split SHA256 mismatch")
    if sha256(AUDIT_INVENTORY_PATH) != sha256(split_path):
        raise ValueError("Frozen split is not byte-identical to the accepted audit inventory")
    if sha256(checkpoint_path) != findings["checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint SHA256 mismatch")

    split_rows = load_csv(split_path)
    validation_metadata = {
        row["path"]: row for row in split_rows if row["draft_split"] == "validation"
    }
    prediction_rows = load_csv(predictions_path)
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
            [float(prediction[f"score_{name}"]) for name in CLASSES],
            dtype=np.float64,
        )
        predicted = CLASSES[int(probability.argmax())]
        if prediction["predicted"] != predicted:
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
        if sha256(image) != metadata["sha256"]:
            raise ValueError(f"Classifier validation image SHA256 mismatch: {metadata['path']}")

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
    raw_probabilities = temperature_probabilities(logits, 1.0)
    stored_probability_array = np.asarray(stored_probabilities, dtype=np.float64)
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_probability_difference = float(
        np.max(np.abs(raw_probabilities - stored_probability_array))
    )
    if maximum_probability_difference > tolerance:
        raise ValueError(
            "Recomputed/stored classifier probability mismatch: "
            f"{maximum_probability_difference}"
        )
    recomputed_metrics, _ = classifier.metrics_for(label_array, raw_probabilities)
    expected_metrics = load_json(RUN / "validation_metrics.json")["validation"]
    maximum_metric_difference = max(
        abs(recomputed_metrics[name] - expected_metrics[name])
        for name in expected_metrics
    )
    if maximum_metric_difference > tolerance:
        raise ValueError(
            f"Recomputed classifier metric mismatch: {maximum_metric_difference}"
        )
    fold_assignment, calibrated_probabilities, fold_results = cross_fit(
        logits, label_array, records
    )
    final_temperature = fit_temperature(logits, label_array)
    raw_summary = summarize(raw_probabilities, label_array)
    calibrated_summary = summarize(calibrated_probabilities, label_array)
    groups = [row["similarity_group"] for row in records]
    group_counts = Counter(groups)
    bootstrap = bootstrap_metric_deltas(
        raw_probabilities,
        calibrated_probabilities,
        label_array,
        groups,
    )

    raw_predicted = raw_probabilities.argmax(axis=1)
    calibrated_predicted = calibrated_probabilities.argmax(axis=1)
    predictions_identical = bool(np.array_equal(raw_predicted, calibrated_predicted))
    checks = {
        "multiclass_nll_improved": calibrated_summary["multiclass_nll"]
        < raw_summary["multiclass_nll"],
        "multiclass_brier_improved": calibrated_summary["multiclass_brier"]
        < raw_summary["multiclass_brier"],
        "top_label_ece_at_most_0_05": calibrated_summary["top_label_ece"] <= 0.05,
        "every_classwise_ece_at_most_0_08": max(
            item["ece"] for item in calibrated_summary["classwise"].values()
        )
        <= 0.08,
        "brier_delta_bootstrap_ci95_upper_below_zero": bootstrap[
            "multiclass_brier"
        ]["ci95_upper"]
        < 0.0,
        "four_class_predictions_identical": predictions_identical,
    }

    diagram_path = RUN / "classifier_reliability_diagram.png"
    render_reliability_diagram(diagram_path, raw_summary, calibrated_summary)
    fold_path = RUN / "classifier_calibration_folds.csv"
    oof_path = RUN / "classifier_calibration_oof_predictions.csv"
    write_csv(
        fold_path,
        [
            {
                **row,
                "fold": int(fold_assignment[index]) + 1,
            }
            for index, row in enumerate(records)
        ],
        ["path", "label", "similarity_group", "fold"],
    )
    oof_rows = []
    for index, row in enumerate(records):
        raw = raw_probabilities[index]
        calibrated = calibrated_probabilities[index]
        oof_rows.append(
            {
                **row,
                "fold": int(fold_assignment[index]) + 1,
                "raw_predicted": CLASSES[int(raw.argmax())],
                "calibrated_predicted": CLASSES[int(calibrated.argmax())],
                "raw_confidence": float(raw.max()),
                "calibrated_confidence": float(calibrated.max()),
                **{f"raw_probability_{name}": float(raw[i]) for i, name in enumerate(CLASSES)},
                **{
                    f"calibrated_probability_{name}": float(calibrated[i])
                    for i, name in enumerate(CLASSES)
                },
            }
        )
    write_csv(oof_path, oof_rows, list(oof_rows[0]))

    result = {
        "schema_version": 1,
        "status": "CLASSIFIER_TEMPERATURE_SCALING_COMPLETE",
        "scope": "Post-hoc BRISC internal-validation calibration; not clinical validation",
        "method": {
            "name": "scalar temperature scaling",
            "fit_objective": "multiclass negative log-likelihood",
            "temperature_bounds": [0.05, 20.0],
            "logits_source": (
                "recomputed from the frozen accepted checkpoint on the 893 validation "
                "images; probabilities were verified against the frozen prediction CSV"
            ),
            "cross_fitting": "five-fold StratifiedGroupKFold, shuffle=True, seed=42",
            "similarity_groups": {
                "source": "reused unchanged from the frozen split and accepted audit inventory",
                "construction": (
                    "transitive components of exact decoded-pixel matches or perceptual-hash "
                    "Hamming distance <=4 across the full BRISC inventory"
                ),
                "candidate_pairs_include_cross_original_split_links": True,
                "new_groups_computed_for_calibration": False,
                "patient_groups": False,
            },
        },
        "decision_rules": {
            "nll_is_fit_objective": True,
            "ece_is_required_acceptance_gate": True,
            "nll_improves_but_ece_above_0_05": "FAIL",
            "operational_predictions_changed": False,
            "application_integration_performed": False,
            "temperature_accepted_for_application": all(checks.values()),
        },
        "validation_cases": len(records),
        "similarity_groups": len(set(groups)),
        "multi_member_similarity_groups": sum(
            count > 1 for count in group_counts.values()
        ),
        "largest_similarity_group": max(group_counts.values()),
        "folds": fold_results,
        "final_temperature_all_validation": final_temperature,
        "metrics": {
            "raw": raw_summary,
            "cross_fitted_temperature_scaled": calibrated_summary,
            "bootstrap_paired_deltas": bootstrap,
        },
        "gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "rule": "All checks must pass; there is no metric tie-break.",
        },
        "verification": {
            "classifier_run": RUN_NAME,
            "checkpoint_sha256": sha256(checkpoint_path),
            "split_sha256": sha256(split_path),
            "accepted_audit_inventory_sha256": sha256(AUDIT_INVENTORY_PATH),
            "similarity_candidates_sha256": sha256(SIMILARITY_CANDIDATES_PATH),
            "validation_predictions_sha256": sha256(predictions_path),
            "script_sha256": sha256(Path(__file__)),
            "inference_device": str(device),
            "maximum_stored_probability_difference": maximum_probability_difference,
            "maximum_stored_metric_difference": maximum_metric_difference,
            "fold_groups_disjoint": True,
            "test_images_accessed": 0,
            "test_predictions_accessed": 0,
            "model_retrained": False,
            "threshold_changed": False,
        },
        "artifacts": {
            "reliability_diagram": str(diagram_path),
            "fold_assignments": str(fold_path),
            "out_of_fold_predictions": str(oof_path),
        },
        "limitations": [
            "The classifier checkpoint was selected using this validation partition before calibration.",
            "Similarity groups are not verified patient identifiers.",
            "Patient independence and cross-dataset generalization remain unverified.",
            "The fitted all-validation temperature is an operational parameter; reported performance uses only out-of-fold calibrated probabilities.",
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
                "calibration_method": "scalar_temperature_scaling",
                "calibration_folds": FOLDS,
                "calibration_ece_bins": ECE_BINS,
            }
        )
        mlflow.log_metrics(
            {
                "calibration_final_temperature": final_temperature,
                "calibration_raw_nll": raw_summary["multiclass_nll"],
                "calibration_oof_nll": calibrated_summary["multiclass_nll"],
                "calibration_raw_brier": raw_summary["multiclass_brier"],
                "calibration_oof_brier": calibrated_summary["multiclass_brier"],
                "calibration_raw_ece": raw_summary["top_label_ece"],
                "calibration_oof_ece": calibrated_summary["top_label_ece"],
                "calibration_gate_pass": float(all(checks.values())),
            }
        )
        for artifact in (OUTPUT_JSON, OUTPUT_MARKDOWN, diagram_path, fold_path, oof_path):
            mlflow.log_artifact(str(artifact))

    print(f"Final temperature: {final_temperature:.8f}")
    print(
        "Raw -> calibrated: "
        f"NLL {raw_summary['multiclass_nll']:.6f} -> {calibrated_summary['multiclass_nll']:.6f}; "
        f"Brier {raw_summary['multiclass_brier']:.6f} -> {calibrated_summary['multiclass_brier']:.6f}; "
        f"ECE {raw_summary['top_label_ece']:.6f} -> {calibrated_summary['top_label_ece']:.6f}"
    )
    print(f"Gate: {result['gate']['status']}")


if __name__ == "__main__":
    main()
