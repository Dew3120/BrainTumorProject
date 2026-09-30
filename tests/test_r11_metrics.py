import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_r11_size_robust_metrics import lesion_counts, surface_metrics


def test_identical_mask_surface_and_lesion_metrics():
    mask = np.zeros((128, 128), dtype=bool)
    mask[20:30, 40:50] = True
    surface_dice, hd95 = surface_metrics(mask, mask, 2.0, np.sqrt(2 * 127**2))
    references, detected, predictions = lesion_counts(mask, mask, 0.5)
    assert surface_dice == 1.0
    assert hd95 == 0.0
    assert (references, detected, predictions) == (1, 1, 1)


def test_empty_prediction_rules():
    reference = np.zeros((128, 128), dtype=bool)
    reference[20:30, 40:50] = True
    prediction = np.zeros_like(reference)
    diagonal = np.sqrt(2 * 127**2)
    assert surface_metrics(prediction, reference, 2.0, diagonal) == (0.0, diagonal)
    assert lesion_counts(prediction, reference, 0.5) == (1, 0, 0)
