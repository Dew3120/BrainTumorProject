"""Tests for frozen segmenter pixel temperature scaling."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_segmenter_pixels import (
    lesion_region,
    pixel_nll,
    pixel_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]


def test_lesion_region_uses_fixed_eight_pixel_euclidean_dilation():
    reference = np.zeros((32, 32), dtype=bool)
    reference[16, 16] = True
    region = lesion_region(reference, radius=8.0)
    assert region[16, 16]
    assert region[16, 24]
    assert not region[16, 25]
    assert not region[0, 0]


def test_positive_temperature_preserves_pixel_threshold():
    logits = np.array([[-8.0, -0.01, 0.01, 8.0]], dtype=np.float64)
    raw = pixel_probabilities(logits, 1.0)
    calibrated = pixel_probabilities(logits, 3.5)
    assert np.array_equal(raw >= 0.5, calibrated >= 0.5)


def test_binary_pixel_nll_is_finite():
    logits = np.array([[-100.0, 100.0]], dtype=np.float64)
    targets = np.array([[False, True]])
    assert np.isfinite(pixel_nll(logits, targets, 1.0))


def test_persisted_segmenter_calibration_is_fail_closed_and_oof():
    result = json.loads(
        (ROOT / "audit/segmenter_pixel_calibration.json").read_text(encoding="utf-8")
    )
    assert result["validation_images"] == 680
    assert result["method"]["bootstrap"]["fold_stratified"] is False
    assert result["method"]["bootstrap"]["same_structure_as_classifier"] is True
    assert result["method"]["binning"]["count"] == 15
    assert result["final_temperature_role"]["used_for_reported_metrics"] is False
    assert len(result["folds"]) == 5
    assert result["gate"]["status"] == (
        "PASS" if all(result["gate"]["checks"].values()) else "FAIL"
    )
    verification = result["verification"]
    assert verification["oof_changed_mask_pixels"] == 0
    assert verification["final_temperature_changed_mask_pixels"] == 0
    assert verification["test_images_accessed"] == 0
    assert verification["test_predictions_accessed"] == 0
    assert verification["model_retrained"] is False
    assert verification["threshold_changed"] is False
    assert verification["application_integration_performed"] is False
