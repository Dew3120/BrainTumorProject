"""Tests for binary tumor/no-tumor temperature scaling."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_classifier_binary_confidence import (
    binary_temperature_probabilities,
    binary_tumor_margin,
    fit_binary_temperature,
)

ROOT = Path(__file__).resolve().parents[1]


def test_collapsed_margin_matches_summed_tumor_softmax_probability():
    logits = np.array(
        [
            [2.0, 1.0, -1.0, 0.5],
            [-3.0, -2.0, -1.0, 4.0],
            [8.0, 7.0, 6.0, -8.0],
        ],
        dtype=np.float64,
    )
    expected = softmax(logits, axis=1)[:, :3].sum(axis=1)
    actual = binary_temperature_probabilities(binary_tumor_margin(logits), 1.0)
    assert np.allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_positive_binary_temperature_preserves_half_threshold_decisions():
    margins = np.array([-10.0, -0.01, 0.01, 10.0], dtype=np.float64)
    raw = binary_temperature_probabilities(margins, 1.0)
    calibrated = binary_temperature_probabilities(margins, 3.7)
    assert np.array_equal(raw >= 0.5, calibrated >= 0.5)


def test_binary_temperature_uses_binary_nll_to_reduce_overconfidence():
    margins = np.array([8.0, 7.0, 6.0, -8.0, -7.0, -6.0], dtype=np.float64)
    targets = np.array([True, False, True, False, True, False])
    temperature = fit_binary_temperature(margins, targets)
    assert temperature > 1.0


def test_persisted_binary_calibration_is_oof_and_gate_consistent():
    result = json.loads(
        (ROOT / "audit/classifier_binary_calibration.json").read_text(encoding="utf-8")
    )
    assert result["method"]["fit_objective"] == (
        "binary tumor/no-tumor negative log-likelihood"
    )
    assert result["method"]["four_class_nll_used_for_fitting"] is False
    assert result["final_temperature_role"]["used_for_reported_metrics"] is False
    assert len(result["folds"]) == 5
    assert result["validation_cases"] == 893
    assert result["similarity_groups"] == 784
    assert result["gate"]["status"] == (
        "PASS" if all(result["gate"]["checks"].values()) else "FAIL"
    )
    assert result["decision_rules"]["temperature_accepted_for_application"] == (
        result["gate"]["status"] == "PASS"
    )
    assert result["verification"]["binary_decisions_changed_at_0_50"] == 0
    assert result["verification"]["test_images_accessed"] == 0
    assert result["verification"]["test_predictions_accessed"] == 0
    assert result["verification"]["model_retrained"] is False
    assert result["verification"]["threshold_changed"] is False
