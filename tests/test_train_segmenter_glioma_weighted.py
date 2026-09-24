import torch
import train_segmenter as baseline
import train_segmenter_glioma_weighted as weighted
from monai.losses import DiceLoss


def example_batch():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((3, 1, 16, 16), generator=generator)
    targets = torch.zeros_like(logits)
    targets[0, 0, 2:6, 3:8] = 1
    targets[1, 0, 7:12, 1:5] = 1
    targets[2, 0, 4:13, 8:14] = 1
    return logits, targets


def test_unweighted_per_case_mean_matches_v1_objective():
    logits, targets = example_batch()
    expected = baseline.segmentation_loss(
        logits, targets, DiceLoss(sigmoid=True)
    )
    actual = weighted.segmentation_loss(
        logits, targets, DiceLoss(sigmoid=True, reduction="none")
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_glioma_weight_is_normalized_three_to_one():
    logits, targets = example_batch()
    loss = DiceLoss(sigmoid=True, reduction="none")
    per_case = weighted.per_case_segmentation_loss(logits, targets, loss)
    labels = torch.tensor(
        [
            weighted.LABEL_INDEX["glioma"],
            weighted.LABEL_INDEX["meningioma"],
            weighted.LABEL_INDEX["pituitary"],
        ]
    )
    actual = weighted.weighted_training_loss(logits, targets, labels, loss)
    expected = (3 * per_case[0] + per_case[1] + per_case[2]) / 5
    torch.testing.assert_close(actual, expected)
