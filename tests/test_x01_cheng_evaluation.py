from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_x01_cheng import (
    boundary_metrics,
    equal_frequency_calibration,
    mask_scores,
)


def test_mask_scores_exact_and_disjoint():
    reference = np.zeros((2, 8, 8), dtype=bool)
    prediction = np.zeros_like(reference)
    reference[0, 2:4, 2:4] = True
    prediction[0] = reference[0]
    reference[1, 2:4, 2:4] = True
    prediction[1, 5:7, 5:7] = True
    dice, iou = mask_scores(prediction, reference)
    assert np.allclose(dice, [1, 0])
    assert np.allclose(iou, [1, 0])


def test_boundary_metrics_identical_masks():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 5:11] = True
    boundary_f1, hausdorff = boundary_metrics(mask, mask, 0.49)
    assert boundary_f1 == 1.0
    assert hausdorff == 0.0


def test_equal_frequency_calibration_perfect_predictions():
    target = np.asarray([0, 0, 1, 1], dtype=bool)
    probability = target.astype(float)
    result = equal_frequency_calibration(probability, target, bins=2)
    assert result["ece"] == 0.0
    assert result["mce"] == 0.0
