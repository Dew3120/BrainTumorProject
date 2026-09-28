import numpy as np
from run_p02_threshold_selection import (
    binary_curve,
    candidate_thresholds,
    select_binary,
    three_state_grid,
)


def test_binary_selection_uses_registered_cost_and_strict_threshold():
    reliability = np.asarray([0.1, 0.2, 0.8, 0.9])
    success = np.asarray([False, True, False, True])
    rows = binary_curve(reliability, success, candidate_thresholds(reliability))
    selected = select_binary(rows)

    assert selected["threshold"] == 0.9
    assert selected["tp_bad_reviewed"] == 2
    assert selected["fp_acceptable_reviewed"] == 1
    assert selected["fn_bad_accepted"] == 0
    assert selected["total_cost"] == 3


def test_binary_tie_break_prefers_fewer_failures_then_less_review():
    rows = [
        {"total_cost": 5, "fn_bad_accepted": 1, "flagged_cases": 1, "threshold": 0.8},
        {"total_cost": 5, "fn_bad_accepted": 0, "flagged_cases": 3, "threshold": 0.6},
        {"total_cost": 5, "fn_bad_accepted": 0, "flagged_cases": 2, "threshold": 0.5},
    ]
    assert select_binary(rows)["threshold"] == 0.5


def test_three_state_grid_uses_exact_registered_quarter_costs():
    reliability = np.asarray([0.1, 0.4, 0.7, 0.9])
    success = np.asarray([False, True, False, True])
    thresholds = candidate_thresholds(reliability)
    rows, selected = three_state_grid(reliability, success, thresholds)

    assert len(rows) == len(thresholds) * (len(thresholds) + 1) // 2
    recomputed = (
        selected["review_acceptable"]
        + selected["review_bad"]
        + 0.25 * selected["warning_acceptable"]
        + 5 * selected["warning_bad"]
        + 10 * selected["standard_bad"]
    )
    assert selected["total_cost"] == recomputed
