import numpy as np
import torch
import train_segmenter_boundary_loss as boundary
from monai.losses import DiceLoss


def test_signed_distance_has_registered_sign_and_pixel_units():
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 3:6] = True
    distance = boundary.signed_distance_map(mask)
    assert distance.dtype == np.float32
    assert distance[4, 4] < 0
    assert distance[0, 0] > 0
    assert distance[3, 3] == -1.0
    assert distance[2, 3] == 1.0


def test_boundary_term_pushes_inside_up_and_outside_down():
    target = torch.zeros((1, 1, 9, 9))
    target[:, :, 3:6, 3:6] = 1
    distance = torch.from_numpy(
        boundary.signed_distance_map(target[0, 0].numpy() >= 0.5)
    ).reshape_as(target)
    logits = torch.zeros_like(target, requires_grad=True)
    term = boundary.loss_components(
        logits, target, distance, DiceLoss(sigmoid=True)
    )["boundary"]
    term.backward()
    assert logits.grad[0, 0, 4, 4] < 0
    assert logits.grad[0, 0, 0, 0] > 0


def test_total_loss_uses_equal_unscaled_component_sum():
    target = torch.zeros((1, 1, 9, 9))
    target[:, :, 3:6, 3:6] = 1
    distance = torch.from_numpy(
        boundary.signed_distance_map(target[0, 0].numpy() >= 0.5)
    ).reshape_as(target)
    logits = torch.zeros_like(target)
    dice = DiceLoss(sigmoid=True)
    components = boundary.loss_components(logits, target, distance, dice)
    total = boundary.segmentation_loss(logits, target, distance, dice)
    assert tuple(components) == ("balanced_bce", "foreground_dice", "boundary")
    torch.testing.assert_close(total, sum(components.values()))
