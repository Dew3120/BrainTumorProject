"""Tests for the fail-closed Phase 4 findings consolidation."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase4_consolidate_findings import summarize_failures


def failure_row(category, fragmented=False, disconnected=False, dice="0.25"):
    return {
        "draft_split": "validation",
        "category": category,
        "dice": dice,
        "fragmented_prediction": str(fragmented),
        "disconnected_false_positive": str(disconnected),
    }


def valid_failure_rows():
    categories = ["glioma"] * 60 + ["meningioma"] * 8 + ["pituitary"] * 15
    rows = []
    for index, category in enumerate(categories):
        rows.append(
            failure_row(
                category,
                fragmented=index < 70,
                disconnected=index < 67,
            )
        )
    return rows


def test_failure_summary_preserves_verified_section_b_counts():
    result = summarize_failures(valid_failure_rows())
    assert result["case_count"] == 83
    assert result["fragmented_prediction"]["count"] == 70
    assert result["fragmented_prediction"]["reported_percent"] == 84.3
    assert result["disconnected_false_positive"]["count"] == 67
    assert result["disconnected_false_positive"]["reported_percent"] == 80.7


def test_failure_summary_rejects_changed_membership():
    with pytest.raises(ValueError, match="Expected 83"):
        summarize_failures(valid_failure_rows()[:-1])


def test_generated_findings_contains_required_section_f_evidence():
    findings = json.loads(
        (ROOT / "audit/phase4_full_training_findings.json").read_text(encoding="utf-8")
    )
    section = findings["phase4_consolidation"]
    assert section["overall_metrics"]["reported_4dp"] == {
        "image_mean_dice": 0.7694,
        "image_mean_iou": 0.6724,
        "macro_type_mean_dice": 0.7627,
    }
    assert section["per_category_mean_dice"]["glioma"]["reported_4dp"] == 0.609
    assert (
        section["classifier_segmenter_agreement"][
            "failures_with_correct_classifier"
        ]["count"]
        == 78
    )
    assert (
        section["uncertainty_signals"]["spearman_correlations_all_680"][
            "reported_4dp"
        ]["classifier_confidence_vs_segmenter_dice"]
        == 0.3252
    )
    hypothesis = section["falsified_hypotheses"][0]
    assert hypothesis["result"] == "NOT_CONFIRMED_BY_PREDECLARED_SUPPORT_RULE"
    assert hypothesis["supporting_numbers"]["centroids_below_image_midline"] == 11
    assert hypothesis["supporting_numbers"]["centroids_below_normalized_y_0_60"] == 1
    git_limit = section["reproducibility_posture"]["training_time_git_head"]
    assert git_limit["recorded"] is False
    assert "not a proven launch-time commit" in git_limit["explicit_limitation"]
    assert section["constraints_verified"] == {
        "retraining_performed": False,
        "threshold_changed": False,
        "test_images_accessed": 0,
        "phase5_started": False,
    }
