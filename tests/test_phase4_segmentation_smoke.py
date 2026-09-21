"""Checks for paired geometry, binary masks and segmentation metric correctness."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/phase4_segmentation_smoke.py"
SPEC = importlib.util.spec_from_file_location("phase4", SCRIPT)
phase4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase4)


def test_asymmetric_rectangle_stays_aligned_and_binary():
    pixels = np.zeros((40, 80), dtype=np.uint8)
    pixels[4:15, 50:72] = 255
    image, target = phase4.transform_pair(
        Image.fromarray(pixels), Image.fromarray(pixels)
    )
    assert image.shape == target.shape == (1, 128, 128)
    assert set(torch.unique(target).tolist()) == {0.0, 1.0}
    image_points = torch.nonzero(image[0] > 0.5).float().mean(dim=0)
    target_points = torch.nonzero(target[0] > 0.5).float().mean(dim=0)
    assert torch.max(torch.abs(image_points - target_points)) < 1
    assert target_points[1] > 80  # No accidental horizontal flip.
    assert not target[0, :32].any()  # Aspect-preserving padding, not stretched.


def test_fixed_threshold_and_zero_one_masks():
    grayscale = Image.fromarray(np.array([[0, 10, 127, 128, 246, 255]], dtype=np.uint8))
    assert np.asarray(phase4.binarize_mask(grayscale)).tolist() == [
        [0, 0, 0, 255, 255, 255]
    ]
    binary = Image.fromarray(np.array([[0, 1]], dtype=np.uint8))
    assert np.asarray(phase4.binarize_mask(binary)).tolist() == [[0, 255]]


def test_empty_positive_mask_and_mismatched_geometry_fail():
    with pytest.raises(ValueError, match="empty"):
        phase4.transform_pair(Image.new("L", (16, 16)), Image.new("L", (16, 16)))
    with pytest.raises(ValueError, match="dimensions"):
        phase4.transform_pair(Image.new("L", (16, 16)), Image.new("L", (12, 16), 255))


def test_dice_and_iou_known_overlap():
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    dice, iou = phase4.mask_scores(logits, target)
    assert dice.item() == pytest.approx(0.5)
    assert iou.item() == pytest.approx(1 / 3)
