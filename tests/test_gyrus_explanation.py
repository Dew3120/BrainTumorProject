"""Human-facing explanations must preserve the registered decision semantics."""

from __future__ import annotations

import pytest

from gyrus.domain.explanation import (
    ood_explanation,
    policy_explanation,
    reliability_explanation,
)


def test_review_reasons_are_readable_and_complete() -> None:
    result = {
        "final_policy": {
            "status": "REVIEW_REQUIRED",
            "reasons": [
                "AT_OR_ABOVE_REVIEW_THRESHOLD",
                "SEGMENTATION_RELIABILITY_NOT_APPLICABLE",
            ],
        },
        "ood": {"status": "REVIEW_REQUIRED"},
        "reliability": {"score": None},
    }
    explanation = policy_explanation(result)
    assert "manual review" in explanation
    assert "no tumor" in explanation
    assert "_" not in explanation
    assert "mask" in reliability_explanation(result)
    assert "Review" in ood_explanation(result)


def test_unknown_reason_never_gets_an_invented_explanation() -> None:
    result = {
        "final_policy": {
            "status": "REVIEW_REQUIRED",
            "reasons": ["UNREGISTERED_REASON"],
        }
    }
    with pytest.raises(ValueError, match="Unrecognized final policy reason"):
        policy_explanation(result)
