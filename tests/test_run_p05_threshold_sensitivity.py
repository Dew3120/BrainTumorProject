import csv
import json
from pathlib import Path

import numpy as np
from run_p05_threshold_sensitivity import SELECTED_THRESHOLD, evaluate

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv"
CURVE = ROOT / "experiments/p05_threshold_sensitivity/artifacts/sensitivity_curve.csv"
RESULT = ROOT / "experiments/p05_threshold_sensitivity/result.json"


def load_source():
    with OOF.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    reliability = np.asarray([float(row["reduced_oof_reliability"]) for row in rows])
    acceptable = np.asarray(
        [bool(int(row["success_dice_at_least_0_50"])) for row in rows]
    )
    return reliability, acceptable


def independent_counts(reliability, acceptable, threshold):
    flagged = [score < threshold for score in reliability]
    caught = sum(is_flagged and not good for is_flagged, good in zip(flagged, acceptable, strict=True))
    missed = sum(not is_flagged and not good for is_flagged, good in zip(flagged, acceptable, strict=True))
    acceptable_flagged = sum(is_flagged and good for is_flagged, good in zip(flagged, acceptable, strict=True))
    acceptable_accepted = sum(not is_flagged and good for is_flagged, good in zip(flagged, acceptable, strict=True))
    return {
        "cases_flagged": caught + acceptable_flagged,
        "cases_accepted": missed + acceptable_accepted,
        "failures_caught": caught,
        "failures_missed": missed,
        "acceptable_cases_flagged": acceptable_flagged,
        "acceptable_cases_accepted": acceptable_accepted,
        "total_cost": caught + acceptable_flagged + 10 * missed,
    }


def test_selected_threshold_reproduces_p02():
    reliability, acceptable = load_source()
    row = evaluate(reliability, acceptable, SELECTED_THRESHOLD)
    assert row == {
        "threshold": SELECTED_THRESHOLD,
        "cases_flagged": 195,
        "cases_accepted": 485,
        "failures_caught": 81,
        "failures_missed": 1,
        "acceptable_cases_flagged": 114,
        "acceptable_cases_accepted": 484,
        "total_cost": 205,
        "mean_cost": 205 / 680,
    }


def test_every_curve_row_is_independently_recomputed():
    reliability, acceptable = load_source()
    with CURVE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        expected = independent_counts(reliability, acceptable, float(row["threshold"]))
        for key, value in expected.items():
            assert int(row[key]) == value
        assert np.isclose(float(row["mean_cost"]), expected["total_cost"] / 680)


def test_registered_robustness_rule_and_constraints_are_preserved():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    requested = {row["label"]: row for row in result["requested_perturbations"]}
    selected_cost = requested["selected"]["total_cost"]
    local_worst = max(requested["-0.02"]["total_cost"], requested["+0.02"]["total_cost"])
    worst_005 = max(requested["-0.05"]["total_cost"], requested["+0.05"]["total_cost"])
    expected_flat = local_worst <= 1.10 * selected_cost
    expected_robust = expected_flat and worst_005 <= 1.25 * selected_cost
    assert result["sensitivity"]["locally_flat"] is expected_flat
    assert (result["sensitivity"]["classification"] == "ROBUST") is expected_robust
    assert result["verification"]["test_images_accessed"] == 0
    assert result["verification"]["segmentation_threshold"] == 0.5
    assert result["verification"]["segmentation_threshold_changed"] is False
    assert result["verification"]["model_retrained_or_refit"] is False


def test_positive_perturbations_are_clipped_to_probability_domain():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    requested = {row["label"]: row for row in result["requested_perturbations"]}
    assert requested["+0.10"]["threshold"] == 1.0
    assert requested["+0.10"]["was_clipped"] is True
