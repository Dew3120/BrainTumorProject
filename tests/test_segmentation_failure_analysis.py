"""Failure modes must distinguish empty, misplaced, and overlapping predictions."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segmentation_failure_analysis import (
    component_statistics,
    primary_mode_table,
    summarize,
)


def test_empty_and_nonempty_zero_overlap_remain_separate():
    reference = np.zeros((8, 8), dtype=bool)
    reference[2:4, 2:4] = True
    empty = component_statistics(np.zeros_like(reference), reference)
    assert empty["empty_prediction"] is True
    assert empty["nonempty_zero_overlap"] is False
    assert empty["primary_failure_mode"] == "empty_prediction"
    misplaced_mask = np.zeros_like(reference)
    misplaced_mask[5:7, 5:7] = True
    misplaced = component_statistics(misplaced_mask, reference)
    assert misplaced["empty_prediction"] is False
    assert misplaced["nonempty_zero_overlap"] is True
    assert misplaced["primary_failure_mode"] == "nonempty_zero_overlap"


def test_component_and_ratio_failure_flags():
    reference = np.zeros((10, 10), dtype=bool)
    reference[2:6, 2:6] = True
    prediction = np.zeros_like(reference)
    prediction[2:4, 2:4] = True
    prediction[8, 8] = True
    result = component_statistics(prediction, reference)
    assert result["overlap_exists"] is True
    assert result["severe_under_prediction"] is True
    assert result["fragmented_prediction"] is True
    assert result["prediction_connected_components"] == 2
    assert result["nonoverlap_component_count"] == 1
    assert result["disconnected_false_positive"] is True
    assert result["largest_component_fraction"] == 0.8


def test_summary_reports_all_and_each_tumor_type():
    rows = []
    for category in ("glioma", "meningioma", "pituitary"):
        rows.append(
            {
                "category": category,
                "empty_prediction": category == "glioma",
                "nonempty_zero_overlap": category == "meningioma",
                "severe_under_prediction": category == "pituitary",
                "severe_over_prediction": False,
                "fragmented_prediction": False,
                "disconnected_false_positive": False,
                "reference_touches_model_edge": False,
                "prediction_touches_model_edge": False,
                "primary_failure_mode": {
                    "glioma": "empty_prediction",
                    "meningioma": "nonempty_zero_overlap",
                    "pituitary": "severe_under_prediction",
                }[category],
            }
        )
    result = summarize(rows)
    assert result["all"]["count"] == 3
    assert result["all"]["complete_miss"]["count"] == 2
    assert result["glioma"]["empty_prediction"]["fraction"] == 1
    assert result["pituitary"]["severe_under_prediction"]["fraction"] == 1
    table = primary_mode_table(result)
    assert "| all | 1 (33.3%) | 1 (33.3%) | 1 (33.3%) |" in table
