"""Integrity and boundary tests for the registered OOD policy."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_o06_ood_policy as policy


def result():
    return json.loads((ROOT / "experiments/o06_ood_policy/result.json").read_text())


def test_registered_thresholds_and_prior_thresholds_unchanged():
    value = result()
    assert value["thresholds"]["review"] == 0.36854275476100085
    assert value["thresholds"]["refuse_enabled"] is False
    assert value["verification"]["segmentation_threshold"] == 0.5
    assert value["verification"]["reliability_policy_threshold"] == "0.9490250173235835"


def test_policy_boundaries_and_fail_closed_values():
    review = result()["thresholds"]["review"]
    refuse = result()["thresholds"]["refuse_candidate"]
    assert policy.state(np.nextafter(review, 0), review, refuse, False) == "ACCEPT"
    assert policy.state(review, review, refuse, False) == "REVIEW"
    assert policy.state(float("nan"), review, refuse, False) == "REVIEW"
    assert policy.state(float("inf"), review, refuse, False) == "REVIEW"
    assert policy.state(-0.1, review, refuse, False) == "REVIEW"
    assert policy.state(1.1, review, refuse, False) == "REVIEW"


def test_catalogue_has_required_warning_and_disclaimer():
    catalogue = json.loads((ROOT / "experiments/o06_ood_policy/artifacts/message_catalogue.json").read_text())
    for message in catalogue["messages"]:
        assert message["disclaimer"] == policy.DISCLAIMER
        if message["status"] in {"REVIEW", "REFUSE"}:
            assert policy.REQUIRED_WARNING in message["body"]


def test_ood_experiments_are_complete_and_constraints_hold():
    for experiment_id in ("o01_ood_benchmark", "o02_feature_ood", "o03_encoder_mahalanobis", "o04_model_confidence_ood", "o05_ensemble_ood", "o06_ood_policy"):
        value = json.loads((ROOT / "experiments" / experiment_id / "result.json").read_text())
        assert value["status"] == "COMPLETE"
        verification = value.get("verification", {})
        if verification:
            assert verification["brisc_test_images_accessed"] == 0
            assert verification["neural_retraining_or_refitting"] == 0
            assert verification["segmentation_threshold"] == 0.5
            assert verification["reliability_policy_threshold"] == "0.9490250173235835"


def test_policy_assignment_population_and_coverage():
    path = ROOT / "experiments/o06_ood_policy/artifacts/policy_assignments.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2680
    evaluation = [row for row in rows if row["detector_split"] == "evaluation"]
    assert len(evaluation) == 1380
    assert sum(row["status"] != "ACCEPT" for row in evaluation if row["domain"] == "ID") == 60
    assert sum(row["status"] != "ACCEPT" for row in evaluation if row["domain"] != "ID") == 699
