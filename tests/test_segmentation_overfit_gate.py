"""Decision rules must not silently accept one easy mask or use validation scores."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segmentation_overfit_gate import (
    decision,
    foreground_statistics,
    make_diagnostic_model,
    per_image_balanced_bce,
    per_image_positive_weights,
)


def row(mean=0.91, minimum=0.81, validation=0.0):
    return {"train_dice": mean, "train_min_dice": minimum, "val_dice": validation}


def test_requires_five_consecutive_epochs_and_all_six_images():
    assert decision([row()] * 4) == "CONTINUE"
    assert decision([row()] * 5) == "MEMORIZATION_PASS"
    assert decision([row(minimum=0.79)] * 5) == "CONTINUE"
    assert decision([row()] * 4 + [row(mean=0.89)]) == "CONTINUE"


def test_validation_does_not_select_or_stop_run():
    assert decision([row(validation=0.0)] * 5) == decision([row(validation=1.0)] * 5)


def test_low_plateau_and_budget_limits():
    assert decision([row(mean=0.02, minimum=0.0)] * 49) == "CONTINUE"
    assert (
        decision([row(mean=0.02, minimum=0.0)] * 50) == "STOP_AND_DIAGNOSE_LOW_PLATEAU"
    )
    assert (
        decision([row(mean=0.7, minimum=0.3)] * 100)
        == "BUDGET_EXHAUSTED_DIAGNOSE_BEFORE_FULL_TRAINING"
    )


def test_foreground_statistics_use_all_training_masks_once():
    rows = [{"filename": "a.png"}, {"filename": "b.png"}]
    targets = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
        ]
    )
    per_image, aggregate = foreground_statistics(rows, targets)
    assert [item["foreground_fraction"] for item in per_image] == [0.25, 0.5]
    assert aggregate == {
        "foreground_pixels": 3,
        "background_pixels": 5,
        "total_pixels": 8,
        "foreground_fraction": 3 / 8,
        "pos_weight": 5 / 3,
    }


def test_per_image_weighting_balances_each_mask_independently():
    targets = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
        ]
    )
    logits = torch.zeros_like(targets)
    torch.testing.assert_close(
        per_image_positive_weights(targets), torch.tensor([3.0, 1.0])
    )
    expected = torch.log(torch.tensor(2.0)) * 1.25
    torch.testing.assert_close(per_image_balanced_bce(logits, targets), expected)


def test_compact_linear_model_has_more_capacity_and_preserves_shape():
    tiny = make_diagnostic_model("linear", "tiny")
    compact = make_diagnostic_model("linear", "compact")
    assert sum(p.numel() for p in tiny.parameters()) == 11878
    assert sum(p.numel() for p in compact.parameters()) > 30 * 11878
    assert compact(torch.zeros(1, 1, 128, 128)).shape == (1, 1, 128, 128)
