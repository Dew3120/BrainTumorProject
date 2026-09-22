"""Verify regional BCE contributions use all-pixel normalization."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase4_single_pair import breakdown


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
