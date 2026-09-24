"""Run and close C01 vector scaling on the frozen E08 ensemble."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c01_vector_scaling"
ARTIFACTS = EXPERIMENT / "artifacts"
CLASSIFIER_RUN = ROOT / "runs/classifier/20260921-170607-243133-baseline"
CLASSIFIER_CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")


def fit_binary_affine(logits, targets):
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)

    def objective(parameters):
        scale = float(np.exp(parameters[0]))
        transformed = scale * x + parameters[1]
        probability = expit(transformed)
        error = probability - y
        loss = float(np.mean(np.logaddexp(0.0, transformed) - y * transformed))
        gradient = np.array(
            [np.mean(error * scale * x), np.mean(error)], dtype=np.float64
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(np.log(0.05), np.log(20.0)), (-20.0, 20.0)],
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Binary vector scaling failed: {result.message}")
    return {"scale": float(np.exp(result.x[0])), "bias": float(result.x[1])}


def apply_binary_affine(logits, parameters):
    transformed = parameters["scale"] * logits + parameters["bias"]
    return transformed, expit(transformed)


def fit_multiclass_vector(log_probabilities, labels):
    x = np.asarray(log_probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    def objective(parameters):
        scale = np.exp(parameters[:4])
        bias = parameters[4:] - np.mean(parameters[4:])
        transformed = x * scale + bias
        probability = softmax(transformed, axis=1)
        loss = float(-np.mean(np.log(np.clip(probability[np.arange(len(y)), y], 1e-15, 1))))
        error = probability
        error[np.arange(len(y)), y] -= 1.0
        error /= len(y)
        gradient_scale = np.sum(error * x, axis=0) * scale
        gradient_bias = np.sum(error, axis=0)
        gradient_bias -= gradient_bias.mean()
        return loss, np.concatenate([gradient_scale, gradient_bias])

    result = minimize(
        objective,
        np.zeros(8, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(np.log(0.05), np.log(20.0))] * 4 + [(-20.0, 20.0)] * 4,
        options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Multiclass vector scaling failed: {result.message}")
    bias = result.x[4:] - np.mean(result.x[4:])
    return {"scale": np.exp(result.x[:4]), "bias": bias}


def apply_multiclass_vector(log_probabilities, parameters):
    return softmax(
        log_probabilities * parameters["scale"] + parameters["bias"], axis=1
    )


def classifier_metrics(probabilities, labels):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1)
    one_hot = np.eye(4)[labels]
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    ece, mce, bins = common.equal_frequency_binary(
        confidence, predicted == labels, bins=15
    )
    return {
        "nll": float(-np.mean(np.log(chosen))),
        "brier": float(np.mean(np.square(probabilities - one_hot).sum(axis=1))),
        "ece": ece,
        "mce": mce,
        "accuracy": float(np.mean(predicted == labels)),
        "bins": bins,
    }


def binary_metrics(probabilities, targets):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1 - 1e-15)
    targets = np.asarray(targets, dtype=np.float64)
    logits = np.log(probabilities / (1 - probabilities))
    ece, mce, bins = common.equal_frequency_binary(probabilities, targets, bins=15)
    return {
        "nll": float(np.mean(np.logaddexp(0, logits) - targets * logits)),
        "brier": float(np.mean(np.square(probabilities - targets))),
        "ece": ece,
        "mce": mce,
        "accuracy": float(np.mean((probabilities >= 0.5) == targets)),
        "bins": bins,
    }


def render_classifier(path, title, raw, calibrated):
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="black", label="Ideal")
    for item, name, marker in (
        (raw, "Raw", "o"),
        (calibrated, "OOF calibrated", "s"),
    ):
        axis.plot(
            [entry["mean_probability"] for entry in item["bins"]],
            [entry["observed_frequency"] for entry in item["bins"]],
            marker=marker,
            label=f"{name} (ECE={item['ece']:.4f})",
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Observed correctness", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def serialize_parameters(parameters):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in parameters.items()
    }


def classifier_experiment(assignment):
    predictions = sorted(
        common.read_csv(CLASSIFIER_RUN / "validation_predictions.csv"),
        key=lambda row: common.normalized_path(row["path"]),
    )
    fold_rows = {row["path"]: row for row in common.read_csv(common.FOLDS_PATH)}
    probabilities = np.asarray(
        [
            [float(row[f"score_{name}"]) for name in CLASSIFIER_CLASSES]
            for row in predictions
        ],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.asarray(
        [CLASSIFIER_CLASSES.index(row["label"]) for row in predictions], dtype=np.int64
    )
    folds = np.asarray([int(fold_rows[row["path"]]["fold"]) - 1 for row in predictions])
    if set(folds) != set(range(5)) or len(predictions) != 893:
        raise ValueError("Classifier vector-scaling fold population mismatch")
    log_probability = np.log(np.clip(probabilities, 1e-12, 1.0))
    tumor_probability = probabilities[:, :3].sum(axis=1)
    tumor_logit = np.log(np.clip(tumor_probability, 1e-12, 1 - 1e-12) / np.clip(1 - tumor_probability, 1e-12, 1))
    tumor_target = labels != 3
    calibrated = np.empty_like(probabilities)
    calibrated_binary = np.empty(len(labels), dtype=np.float64)
    fold_output = []
    for fold in range(5):
        fit = folds != fold
        evaluate = folds == fold
        multi = fit_multiclass_vector(log_probability[fit], labels[fit])
        binary = fit_binary_affine(tumor_logit[fit], tumor_target[fit])
        calibrated[evaluate] = apply_multiclass_vector(log_probability[evaluate], multi)
        _, calibrated_binary[evaluate] = apply_binary_affine(tumor_logit[evaluate], binary)
        fold_output.append(
            {
                "fold": fold + 1,
                "four_class": serialize_parameters(multi),
                "binary": binary,
                "fit_cases": int(fit.sum()),
                "evaluation_cases": int(evaluate.sum()),
            }
        )
    final_multi = fit_multiclass_vector(log_probability, labels)
    final_binary = fit_binary_affine(tumor_logit, tumor_target)
    raw_multi = classifier_metrics(probabilities, labels)
    oof_multi = classifier_metrics(calibrated, labels)
    raw_binary = binary_metrics(tumor_probability, tumor_target)
    oof_binary = binary_metrics(calibrated_binary, tumor_target)
    return {
        "folds": fold_output,
        "raw_four_class": raw_multi,
        "oof_four_class": oof_multi,
        "raw_binary": raw_binary,
        "oof_binary": oof_binary,
        "final_all_validation": {
            "status": "proposed operational only; not used for reported metrics",
            "four_class": serialize_parameters(final_multi),
            "binary": final_binary,
        },
        "four_class_decisions_changed": int(
            np.count_nonzero(probabilities.argmax(axis=1) != calibrated.argmax(axis=1))
        ),
        "binary_decisions_changed": int(
            np.count_nonzero((tumor_probability >= 0.5) != (calibrated_binary >= 0.5))
        ),
        "assignment_case_count": len(assignment),
    }


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C01 is not in its registered pre-run state")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    oof_logits = np.empty_like(raw_logits)
    oof_probability = np.empty_like(raw_probability)
    fold_parameters = []
    for fold in range(5):
        fit = assignment != fold
        evaluate = assignment == fold
        parameters = fit_binary_affine(raw_logits[fit], targets[fit])
        transformed, probability = apply_binary_affine(raw_logits[evaluate], parameters)
        oof_logits[evaluate] = transformed
        oof_probability[evaluate] = probability
        fold_parameters.append(
            {
                "fold": fold + 1,
                "scale": parameters["scale"],
                "bias": parameters["bias"],
                "fit_images": int(fit.sum()),
                "evaluation_images": int(evaluate.sum()),
            }
        )
    final_parameters = fit_binary_affine(raw_logits, targets)
    regions = common.lesion_regions(targets)
    all_indices = np.arange(680)
    aggregate = {}
    bootstrap = {}
    for region in common.REGIONS:
        raw, raw_nll, raw_brier = common.region_metrics(
            raw_probability, raw_logits, targets, regions, all_indices, region
        )
        calibrated, calibrated_nll, calibrated_brier = common.region_metrics(
            oof_probability, oof_logits, targets, regions, all_indices, region
        )
        aggregate[region] = {"raw": raw, "oof_calibrated": calibrated}
        groups = [row["similarity_group"] for row in rows]
        bootstrap[region] = {
            "nll_change": common.paired_group_bootstrap(
                raw_nll, calibrated_nll, groups
            ),
            "brier_change": common.paired_group_bootstrap(
                raw_brier, calibrated_brier, groups
            ),
        }
    per_category = {}
    label_names = np.asarray([row["label"] for row in rows])
    for label in common.LABELS:
        indices = np.flatnonzero(label_names == label)
        per_category[label] = {}
        for region in common.REGIONS:
            raw, _, _ = common.region_metrics(
                raw_probability, raw_logits, targets, regions, indices, region
            )
            calibrated, _, _ = common.region_metrics(
                oof_probability, oof_logits, targets, regions, indices, region
            )
            per_category[label][region] = {
                "raw": {key: raw[key] for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")},
                "oof_calibrated": {
                    key: calibrated[key]
                    for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")
                },
            }
    checks = {}
    for region in common.REGIONS:
        raw = aggregate[region]["raw"]
        calibrated = aggregate[region]["oof_calibrated"]
        checks[f"{region}_nll_improved"] = calibrated["pixel_nll"] < raw["pixel_nll"]
        checks[f"{region}_ece_at_most_0_05"] = calibrated["pixel_ece"] <= 0.05
    gate_passed = all(checks.values())
    hard_changes = int(
        np.count_nonzero((raw_probability >= 0.5) != (oof_probability >= 0.5))
    )
    classifier_result = classifier_experiment(assignment)
    common.render_binary_diagram(
        ARTIFACTS / "segmenter_lesion_centered_reliability.png",
        "C01 E08 lesion-centered pixel reliability",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    common.render_binary_diagram(
        ARTIFACTS / "segmenter_whole_image_reliability.png",
        "C01 E08 whole-image pixel reliability",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    render_classifier(
        ARTIFACTS / "classifier_reliability.png",
        "C01 four-class classifier confidence reliability",
        classifier_result["raw_four_class"],
        classifier_result["oof_four_class"],
    )
    common.write_csv(ARTIFACTS / "segmenter_fold_parameters.csv", fold_parameters)
    output = {
        "schema_version": 1,
        "experiment_id": "c01_vector_scaling",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "Internal BRISC calibration comparison; not clinical validation",
        "source_model": "e08_ensemble",
        "validation_cases": 680,
        "segmenter": {
            "fold_parameters": fold_parameters,
            "final_all_validation": {
                "status": "proposed operational only; not used for reported metrics",
                **final_parameters,
            },
            "metrics": {
                "aggregate": aggregate,
                "per_category": per_category,
                "paired_group_bootstrap": bootstrap,
            },
            "hard_mask_changed_pixels": hard_changes,
            "hard_mask_changed_images": int(
                np.any((raw_probability >= 0.5) != (oof_probability >= 0.5), axis=(1, 2)).sum()
            ),
        },
        "classifier_descriptive": classifier_result,
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "Every regional NLL must improve and every regional ECE must be at most 0.05.",
        },
        "verification": {
            "cache_run": str(cache_run),
            "cache_manifest_sha256": common.sha256(cache_run / "manifest.json"),
            "fold_assignment_sha256": common.sha256(common.FOLDS_PATH),
            "calibration_outputs_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "All metrics use the internal validation partition used during model development.",
            "Lesion-centered and background regions use reference masks for evaluation and are unavailable operationally.",
            "Patient independence and external generalization are unverified.",
            "Classifier results are descriptive and do not reverse the accepted raw-classifier decision.",
        ],
    }
    common.write_json(EXPERIMENT / "result.json", output)
    config["status"] = "COMPLETE"
    common.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = common.read_json(registry_path)
    next(item for item in registry["experiments"] if item["id"] == "c01_vector_scaling")["status"] = "COMPLETE"
    common.write_json(registry_path, registry)
    decision = "ACCEPTED" if gate_passed else "REJECTED"
    (EXPERIMENT / "decision.md").write_text(
        f"""# Vector scaling calibration decision

**Decision:** {decision}

The predeclared segmenter gate was {'satisfied' if gate_passed else 'not satisfied'} and was not relaxed. Reported values use out-of-fold transformations only; the all-validation parameters are proposals and were not used for evaluation.

Hard-mask changes are reported rather than hidden. Classifier vector-scaling results are descriptive and do not reverse the previously accepted raw-classifier policy. No model was retrained, no threshold was changed, and no sealed-test image was accessed.
""",
        encoding="utf-8",
    )
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    manifest = {
        "status": "PASS",
        "experiment_id": "c01_vector_scaling",
        "registered_decision": output["status"],
        "test_images_accessed": 0,
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": common.sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in included
        ],
    }
    common.write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
