"""Full segmentation training must preserve split and checkpoint invariants."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from monai.losses import DiceLoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase4_train import (
    qualifies_for_top3,
    retain_top3,
    segmentation_loss,
    summarize_scores,
)


def test_validation_metrics_are_reported_per_tumor_type():
    dice = np.asarray([0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    iou = dice / 2
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    result = summarize_scores(dice, iou, labels)
    assert result["val_dice_glioma"] == pytest.approx(0.3)
    assert result["val_dice_meningioma"] == pytest.approx(0.7)
    assert result["val_dice_pituitary"] == pytest.approx(0.95)
    assert result["val_dice_macro_type_mean"] == pytest.approx(0.65)
    assert result["val_count_glioma"] == 2


def test_top3_retention_keeps_highest_scores_and_deletes_fourth(tmp_path):
    entries = []
    for epoch, score in enumerate((0.5, 0.7, 0.6, 0.65), start=1):
        checkpoint = tmp_path / f"epoch-{epoch}.pt"
        checkpoint.write_text("checkpoint", encoding="utf-8")
        if qualifies_for_top3(entries, score):
            entries = retain_top3(
                entries,
                {"epoch": epoch, "score": score, "path": str(checkpoint)},
            )
    assert [(item["epoch"], item["score"]) for item in entries] == [
        (2, 0.7),
        (4, 0.65),
        (3, 0.6),
    ]
    assert not (tmp_path / "epoch-1.pt").exists()


def test_weighted_segmentation_loss_is_finite_in_float32():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0, 0] = 1
    targets[1, 0, :2, :2] = 1
    loss = segmentation_loss(logits, targets, DiceLoss(sigmoid=True))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
