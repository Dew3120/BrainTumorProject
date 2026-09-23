"""Tests for prediction-only case-level segmenter reliability."""

import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_segmenter_case_reliability import (
    FEATURES,
    prediction_features,
    reliability_metrics,
    top_risk_indices,
)

ROOT = Path(__file__).resolve().parents[1]


def test_feature_extractor_accepts_prediction_only_and_returns_finite_features():
    assert tuple(inspect.signature(prediction_features).parameters) == (
        "probability",
        "threshold",
    )
    probability = np.zeros((128, 128), dtype=np.float64)
    probability[50:70, 55:75] = 0.95
    features = prediction_features(probability)
    assert tuple(features) == FEATURES
    assert all(np.isfinite(value) for value in features.values())
    assert features["empty_prediction"] == 0.0
    assert features["predicted_area_fraction"] > 0.0


def test_empty_prediction_has_explicit_indicator_and_finite_sentinels():
    features = prediction_features(np.zeros((128, 128), dtype=np.float64))
    assert features["empty_prediction"] == 1.0
    assert all(np.isfinite(value) for value in features.values())


def test_positive_brier_skill_uses_lower_model_brier_than_baseline():
    success = np.array([True, True, True, True, True, False, False, False, False, False])
    reliability = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.25, 0.2, 0.15, 0.1, 0.05])
    baseline = np.full(10, 0.5)
    result = reliability_metrics(reliability, baseline, success)
    assert result["brier_skill"] > 0.0
    assert result["brier"] < result["prevalence_baseline_brier"]


def test_top_risk_ties_use_normalized_path():
    rows = [{"path": "B.png"}, {"path": "a.png"}, {"path": "c.png"}]
    risk = np.array([0.8, 0.8, 0.2])
    selected = top_risk_indices(risk, rows, count=2)
    assert selected.tolist() == [1, 0]


def test_persisted_case_reliability_is_oof_and_fail_closed():
    result = json.loads(
        (ROOT / "audit/segmenter_case_reliability.json").read_text(encoding="utf-8")
    )
    assert result["validation_cases"] == 680
    assert result["target"]["failure_cases"] == 83
    assert result["target"]["zero_dice_cases"] == 17
    assert result["target"]["reference_mask_use"].startswith("target creation")
    assert result["cross_fitting"]["reported_scores"] == "out-of-fold only"
    assert result["final_all_validation_model"]["used_for_reported_metrics"] is False
    assert result["gate"]["status"] == (
        "PASS" if all(result["gate"]["checks"].values()) else "FAIL"
    )
    verification = result["verification"]
    assert verification["base_models_retrained"] is False
    assert verification["test_images_accessed"] == 0
    assert verification["test_predictions_accessed"] == 0
    assert verification["threshold_changed"] is False
    assert verification["application_integration_performed"] is False
