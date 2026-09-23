"""Verify regional BCE contributions use all-pixel normalization."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segmentation_single_pair_diagnostic import breakdown, make_diagnostic_model


def test_region_means_and_weighted_contributions_are_distinct():
    logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    loss, metrics, components = breakdown(logits, target)
    assert metrics["bce_foreground_mean"] == pytest.approx(math.log(2))
    assert metrics["bce_background_mean"] == pytest.approx(math.log(2))
    assert metrics["bce_foreground_contribution"] == pytest.approx(math.log(2) / 4)
    assert metrics["bce_background_contribution"] == pytest.approx(math.log(2) * 3 / 4)
    assert metrics["bce_total"] == pytest.approx(math.log(2))
    assert metrics["dice"] == pytest.approx(0.4)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert set(components) == {
        "dice",
        "bce_foreground_contribution",
        "bce_background_contribution",
    }


def test_pos_weight_changes_only_positive_bce_and_not_dice():
    logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    _, unweighted, _ = breakdown(logits, target)
    objective, weighted, _ = breakdown(logits, target, torch.tensor(3.0))
    assert weighted["dice_loss"] == unweighted["dice_loss"]
    assert weighted["bce_foreground_contribution"] == pytest.approx(
        3 * unweighted["bce_foreground_contribution"]
    )
    assert (
        weighted["bce_background_contribution"]
        == unweighted["bce_background_contribution"]
    )
    assert (
        weighted["unweighted_bce_foreground_mean"] == unweighted["bce_foreground_mean"]
    )
    objective.backward()
    assert torch.isfinite(logits.grad).all()


def test_linear_head_produces_unrestricted_logits_at_target_shape():
    model = make_diagnostic_model("linear")
    output = model(torch.zeros(1, 1, 128, 128))
    assert output.shape == (1, 1, 128, 128)
    assert isinstance(model.model[2], torch.nn.ConvTranspose2d)
