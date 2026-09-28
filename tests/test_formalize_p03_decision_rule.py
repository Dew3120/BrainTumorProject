import math

import numpy as np
from formalize_p03_decision_rule import THRESHOLD, apply_policy, policy_schema


def test_exact_boundary_is_standard_and_adjacent_values_split_correctly():
    below = float(np.nextafter(THRESHOLD, -np.inf))
    above = float(np.nextafter(THRESHOLD, np.inf))

    assert apply_policy(below)["reliability_status"] == "REVIEW_REQUIRED"
    assert apply_policy(below)["status_reason"] == "BELOW_THRESHOLD"
    assert apply_policy(THRESHOLD)["reliability_status"] == "STANDARD"
    assert apply_policy(THRESHOLD)["requires_manual_review"] is False
    assert apply_policy(above)["reliability_status"] == "STANDARD"


def test_valid_endpoints_follow_registered_rule():
    assert apply_policy(0.0)["reliability_status"] == "REVIEW_REQUIRED"
    assert apply_policy(1.0)["reliability_status"] == "STANDARD"


def test_unavailable_score_fails_closed_with_distinct_reason():
    output = apply_policy(None)
    assert output["reliability_score"] is None
    assert output["reliability_status"] == "REVIEW_REQUIRED"
    assert output["requires_manual_review"] is True
    assert output["status_reason"] == "RELIABILITY_UNAVAILABLE"


def test_invalid_scores_fail_closed_and_normalize_to_null():
    invalid = [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        "0.99",
        True,
        False,
    ]
    for value in invalid:
        output = apply_policy(value)
        assert output["reliability_score"] is None
        assert output["reliability_status"] == "REVIEW_REQUIRED"
        assert output["requires_manual_review"] is True
        assert output["status_reason"] == "RELIABILITY_INVALID"


def test_outputs_never_emit_warning_state():
    values = [None, -1, 0, THRESHOLD, 1, math.nan, "bad"]
    assert all(apply_policy(value)["reliability_status"] != "WARNING" for value in values)
    assert policy_schema()["properties"]["reliability_status"]["enum"] == [
        "STANDARD",
        "REVIEW_REQUIRED",
    ]
