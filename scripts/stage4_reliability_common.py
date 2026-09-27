"""Shared fail-closed utilities for calibrated-E08 reliability studies."""

import inspect
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np
from fit_segmenter_case_reliability import (
    FEATURES,
    make_model,
    prediction_features,
)
from run_c03_isotonic import apply_isotonic
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def verify_feature_contract(expected_features):
    signature = inspect.signature(prediction_features)
    if tuple(signature.parameters) != ("probability", "threshold"):
        raise ValueError("Prediction feature API accepts unexpected inputs")
    if tuple(expected_features) != FEATURES:
        raise ValueError("Registered feature list differs from implementation")
    return {
        "feature_count": len(FEATURES),
        "function_parameters": list(signature.parameters),
        "inference_time_only": True,
        "reference_mask_feature_input": False,
        "classifier_probability_feature_input": False,
        "raw_image_feature_input": False,
    }


def load_calibrated_e08(config):
    mappings_path = ROOT / config["c03_mappings_path"]
    if calibration.sha256(mappings_path) != config["c03_mappings_sha256"]:
        raise ValueError("C03 mapping hash mismatch")
    if (
        calibration.sha256(ROOT / "experiments/c03_isotonic/result.json")
        != config["c03_result_sha256"]
    ):
        raise ValueError("C03 result hash mismatch")
    cache_run, rows, raw_probability, targets, _labels = calibration.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = calibration.fold_assignment(rows)
    if calibration.sha256(calibration.FOLDS_PATH) != config["fold_assignment_sha256"]:
        raise ValueError("Fold assignment hash mismatch")
    mappings = calibration.read_json(mappings_path)["fold_mappings"]
    if [int(item["fold"]) for item in mappings] != [1, 2, 3, 4, 5]:
        raise ValueError("C03 fold mappings are incomplete")
    calibrated = np.empty_like(raw_probability)
    for fold, mapping in enumerate(mappings):
        indices = np.flatnonzero(assignment == fold)
        _logits, probability = apply_isotonic(raw_probability[indices], mapping)
        calibrated[indices] = probability
    raw_mask = raw_probability >= 0.5
    calibrated_mask = calibrated >= 0.5
    changed = raw_mask != calibrated_mask
    changed_pixels = int(np.count_nonzero(changed))
    changed_images = int(np.any(changed, axis=(1, 2)).sum())
    expected = config["expected_c03_hard_mask_changes"]
    if changed_pixels != int(expected["pixels"]) or changed_images != int(
        expected["images"]
    ):
        raise ValueError(
            f"C03 hard-mask reproduction mismatch: {changed_pixels}/{changed_images}"
        )
    return {
        "cache_run": cache_run,
        "rows": rows,
        "raw_probability": raw_probability,
        "probability": calibrated,
        "targets": targets,
        "assignment": assignment,
        "changed_pixels": changed_pixels,
        "changed_images": changed_images,
    }


def dice_scores(probability, targets, threshold=0.5):
    prediction = np.asarray(probability) >= threshold
    target = np.asarray(targets, dtype=bool)
    intersection = np.logical_and(prediction, target).sum(axis=(1, 2))
    denominator = prediction.sum(axis=(1, 2)) + target.sum(axis=(1, 2))
    if np.any(denominator == 0):
        raise ValueError("Dice target contains a double-empty case")
    return 2.0 * intersection / denominator


def feature_matrix(probability, expected_features):
    contract = verify_feature_contract(expected_features)
    rows = [prediction_features(item, threshold=0.5) for item in probability]
    matrix = np.asarray(
        [[row[name] for name in expected_features] for row in rows], dtype=np.float64
    )
    if matrix.shape != (len(probability), len(expected_features)):
        raise ValueError("Feature matrix shape mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    return matrix, contract


def cross_fit(features, success, assignment, feature_indices=None):
    if feature_indices is None:
        feature_indices = np.arange(features.shape[1])
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    scores = np.empty(len(success), dtype=np.float64)
    baseline_scores = np.empty(len(success), dtype=np.float64)
    models = []
    for fold in range(5):
        fitting = assignment != fold
        evaluation = assignment == fold
        if len(np.unique(success[fitting])) != 2 or len(np.unique(success[evaluation])) != 2:
            raise ValueError(f"Reliability target lacks diversity in fold {fold + 1}")
        model = make_model()
        model.fit(features[fitting][:, feature_indices], success[fitting])
        scores[evaluation] = model.predict_proba(
            features[evaluation][:, feature_indices]
        )[:, 1]
        baseline_scores[evaluation] = float(success[fitting].mean())
        models.append(model)
    if not np.isfinite(scores).all() or not np.isfinite(baseline_scores).all():
        raise ValueError("Incomplete OOF reliability scores")
    return scores, baseline_scores, models


def failure_auroc(reliability, success):
    return float(roc_auc_score(~np.asarray(success, dtype=bool), 1.0 - reliability))


def bootstrap_indices(groups, repetitions=2000, seed=42):
    groups = np.asarray(groups)
    unique = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    output = []
    for _ in range(repetitions):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        output.append(indices)
    return output


def bootstrap_auroc(reliability, success, samples):
    values = np.empty(len(samples), dtype=np.float64)
    for index, members in enumerate(samples):
        values[index] = failure_auroc(reliability[members], success[members])
    return values
