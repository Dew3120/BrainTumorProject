"""Section C agreement logic must preserve all four cells and special flags."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase4_classifier_segmenter_agreement import (
    build_agreement,
    summarize_agreement,
)


def classifier_row(path, category, predicted, binary="tumor"):
    return {
        "path": path,
        "category": category,
        "original_split": "train",
        "draft_split": "validation",
        "predicted_class": predicted,
        "binary_prediction": binary,
        "tumor_score": 0.9 if binary == "tumor" else 0.1,
        "score_glioma": 0.25,
        "score_meningioma": 0.25,
        "score_pituitary": 0.25,
        "score_no_tumor": 0.25,
    }


def segmenter_row(path, category, dice, predicted_pixels):
    return {
        "path": path,
        "label": category,
        "dice": dice,
        "iou": dice / (2 - dice) if dice else 0.0,
        "reference_pixels": 10,
        "predicted_pixels": predicted_pixels,
    }


def test_agreement_covers_all_four_cells():
    classifiers = [
        classifier_row("a", "glioma", "glioma"),
        classifier_row("b", "glioma", "glioma"),
        classifier_row("c", "glioma", "meningioma"),
        classifier_row("d", "glioma", "meningioma"),
    ]
    segmenters = [
        segmenter_row("a", "glioma", 0.8, 10),
        segmenter_row("b", "glioma", 0.2, 10),
        segmenter_row("c", "glioma", 0.8, 10),
        segmenter_row("d", "glioma", 0.2, 10),
    ]
    rows = build_agreement(classifiers, segmenters)
    table, _ = summarize_agreement(rows)
    assert table["classifier_correct"] == {
        "segmenter_dice_at_least_0_50": 1,
        "segmenter_dice_below_0_50": 1,
        "total": 2,
    }
    assert table["classifier_incorrect"] == {
        "segmenter_dice_at_least_0_50": 1,
        "segmenter_dice_below_0_50": 1,
        "total": 2,
    }


def test_empty_and_nonempty_special_cases_remain_separate():
    classifiers = [
        classifier_row("a", "glioma", "no_tumor", "no_tumor"),
        classifier_row("b", "glioma", "glioma", "tumor"),
        classifier_row("c", "glioma", "glioma", "tumor"),
    ]
    segmenters = [
        segmenter_row("a", "glioma", 0.0, 5),
        segmenter_row("b", "glioma", 0.0, 0),
        segmenter_row("c", "glioma", 0.0, 5),
    ]
    rows = build_agreement(classifiers, segmenters)
    _, special = summarize_agreement(rows)
    assert special["flag_classifier_no_tumor_segmenter_nonempty"]["count"] == 1
    assert special["flag_classifier_tumor_segmenter_empty"]["count"] == 1
    assert special["flag_classifier_correct_type_segmenter_zero_dice"]["count"] == 2
