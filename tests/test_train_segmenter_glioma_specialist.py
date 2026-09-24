import numpy as np
import pytest
import train_segmenter_glioma_specialist as specialist


def test_specialist_summary_accepts_only_complete_glioma_validation_population():
    dice = np.linspace(0.1, 0.9, 204)
    iou = np.linspace(0.05, 0.8, 204)
    labels = np.full(204, specialist.LABEL_INDEX["glioma"])

    result = specialist.summarize_glioma_scores(dice, iou, labels)

    assert result["val_count_glioma"] == 204
    assert result["val_dice_glioma"] == pytest.approx(dice.mean())
    assert result["val_iou_glioma"] == pytest.approx(iou.mean())


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros(203, dtype=np.int64),
        np.concatenate([np.zeros(203, dtype=np.int64), np.ones(1, dtype=np.int64)]),
    ],
)
def test_specialist_summary_rejects_changed_population(labels):
    scores = np.ones(len(labels), dtype=np.float64)
    with pytest.raises(ValueError, match="exactly 204 glioma"):
        specialist.summarize_glioma_scores(scores, scores, labels)
