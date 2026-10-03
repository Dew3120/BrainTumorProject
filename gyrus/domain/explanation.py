"""Plain-language descriptions of the frozen display policy.

The registered decisions and exact thresholds live in ``policy``.  This module
only translates already-verified reason codes for people reading the UI or PDF.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_REASONS = {
    "AT_OR_ABOVE_REVIEW_THRESHOLD": (
        "This image differs from the development images. Its model output needs manual review."
    ),
    "OOD_SCORE_UNAVAILABLE": "Image similarity could not be assessed. Manual review is required.",
    "OOD_SCORE_INVALID": "Image similarity returned an invalid value. Manual review is required.",
    "BELOW_THRESHOLD": (
        "The proposed mask fell below the internal reliability threshold. "
        "Check the highlighted region against the image."
    ),
    "RELIABILITY_UNAVAILABLE": "Mask reliability could not be assessed. Manual review is required.",
    "RELIABILITY_INVALID": "Mask reliability returned an invalid value. Manual review is required.",
    "SEGMENTATION_RELIABILITY_NOT_APPLICABLE": (
        "The model predicted no tumor, so it produced no mask to assess. "
        "This is a classification-only result that requires review."
    ),
}


def policy_explanation(result: Mapping[str, Any]) -> str:
    """Explain a verified final result without exposing policy identifiers."""

    policy = result["final_policy"]
    if policy["status"] == "STANDARD":
        return (
            "Both internal review checks passed. Inspect the image and proposed mask; "
            "this status does not establish medical accuracy."
        )
    if policy["status"] != "REVIEW_REQUIRED":
        raise ValueError("Unrecognized final policy status")
    reasons = policy["reasons"]
    if not reasons:
        raise ValueError("Review status has no reason")
    try:
        return " ".join(_REASONS[reason] for reason in reasons)
    except KeyError as exc:
        raise ValueError("Unrecognized final policy reason") from exc


def ood_explanation(result: Mapping[str, Any]) -> str:
    """Describe the input-distribution gate without suggesting mask correctness."""

    status = result["ood"]["status"]
    if status == "ACCEPT":
        return "Input similarity check passed. This does not verify the proposed mask."
    if status == "REVIEW_REQUIRED":
        return "Input differs from development images. Review the model output carefully."
    raise ValueError("Unrecognized input-similarity status")


def reliability_explanation(result: Mapping[str, Any]) -> str:
    """Keep the score's meaning tied to the *model* mask, never a human edit."""

    score = result["reliability"]["score"]
    if score is None:
        return "Not applicable: no model mask was produced, so mask reliability cannot be assessed."
    return (
        "Internal estimate for the original model mask only. "
        "It is not a clinical probability or a guarantee that the mask is correct."
    )
