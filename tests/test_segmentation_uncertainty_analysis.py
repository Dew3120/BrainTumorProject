"""segmentation uncertainty extraction and deterministic selection tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segmentation_uncertainty_analysis import (
    probability_statistics,
    select_groups,
    summarize_group,
)


def test_probability_statistics_separate_inside_and_outside():
    probability = np.array([[0.1, 0.2], [0.8, 0.4]])
    reference = np.array([[True, True], [False, False]])
    result = probability_statistics(probability, reference)
    assert result["segmenter_mean_probability_inside_reference"] == pytest.approx(0.15)
    assert result["segmenter_max_probability_inside_reference"] == 0.2
    assert result["segmenter_max_probability_outside_reference"] == 0.8
    assert result["segmenter_mean_probability_whole_image"] == pytest.approx(0.375)


def make_record(index, dice, category="glioma"):
    return {
        "path": f"case_{index:03d}",
        "category": category,
        "dice": dice,
    }


def test_group_selection_is_complete_and_deterministic():
    zeros = [make_record(index, 0.0) for index in range(17)]
    others = [make_record(index + 17, 0.5 + index / 1000) for index in range(30)]
    selected_zero, selected_best = select_groups(zeros + others)
    assert len(selected_zero) == 17
    assert [row["path"] for row in selected_zero[:2]] == ["case_000", "case_001"]
    assert len(selected_best) == 20
    assert selected_best[0]["path"] == "case_046"
    assert selected_best[-1]["path"] == "case_027"


def test_group_summary_reports_classifier_and_spatial_signals():
    cases = []
    for index, category in enumerate(("glioma", "meningioma")):
        cases.append(
            {
                "category": category,
                "classifier_correct": index == 0,
                "classifier_predicted_class_confidence": 0.8 + index / 10,
                "classifier_true_class_probability": 0.7 + index / 10,
                "segmenter_mean_probability_inside_reference": 0.2 + index / 10,
                "segmenter_max_probability_outside_reference": 0.9,
                "segmenter_mean_probability_whole_image": 0.1,
                "segmenter_max_probability_inside_reference": 0.4,
            }
        )
    summary = summarize_group(cases)
    assert summary["count"] == 2
    assert summary["category_counts"] == {"glioma": 1, "meningioma": 1}
    assert summary["classifier_correct_count"] == 1
    assert summary["distributions"]["classifier_predicted_class_confidence"][
        "median"
    ] == pytest.approx(0.85)
