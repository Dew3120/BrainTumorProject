"""Tests for fail-closed confidence-calibration consolidation."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import consolidate_confidence_calibration as consolidation


def test_source_artifacts_reproduce_all_final_decisions():
    sources = consolidation.load_sources()
    consolidation.verify_inputs(sources)
    findings = consolidation.build_findings(sources)

    decisions = findings["component_decisions"]
    assert decisions["classifier"]["status"] == "CLOSED_RAW_PROBABILITIES_ACCEPTED"
    assert decisions["segmenter_pixel"]["status"] == "CLOSED_RAW_PROBABILITIES_ACCEPTED"
    assert (
        decisions["segmenter_case_reliability"]["status"]
        == "ACCEPTED_FOR_APPLICATION_DESIGN_HANDOFF"
    )
    assert (
        decisions["segmenter_case_reliability"]["application_integration_performed"]
        is False
    )
    assert findings["constraints_verified"]["test_images_accessed"] == 0
    assert findings["constraints_verified"]["threshold_changed"] is False


def test_consolidation_fails_closed_if_case_gate_changes():
    sources = consolidation.load_sources()
    changed = copy.deepcopy(sources)
    changed["segmenter_case"]["gate"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="case reliability decision changed"):
        consolidation.verify_inputs(changed)


def test_glioma_qualification_and_coefficient_limits_are_explicit():
    findings = json.loads(
        (ROOT / "audit" / "confidence_calibration_findings.json").read_text(
            encoding="utf-8"
        )
    )
    subgroup = findings["cross_cutting_finding"]
    assert subgroup["title"] == "Glioma remains the weakest subgroup"
    assert subgroup["glioma"]["failure_detection_auroc"] == pytest.approx(
        0.8474537037037038
    )
    assert subgroup["glioma"]["failures"] == 60
    assert subgroup["resolved"] is False

    signals = findings["component_decisions"]["segmenter_case_reliability"][
        "interpretable_signals"
    ]
    assert signals["strongest_mean_standardized_coefficient"]["feature"] == (
        "boundary_entropy_mean"
    )
    assert signals["confident_wrong_signal"]["negative_folds"] == 5
    assert signals["causal_interpretation_allowed"] is False


def test_completion_report_matches_decisions_and_scope():
    report = (ROOT / "audit" / "confidence_calibration_completion.md").read_text(
        encoding="utf-8"
    )
    assert "Glioma remains the weakest subgroup" in report
    assert "descriptive associations, not causal effects" in report
    assert "60/83" in report
    assert "no test image or prediction was accessed" in report
    assert "no application integration was performed" in report
