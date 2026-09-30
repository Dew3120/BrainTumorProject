"""Pure OOD, reliability, and final-display policy functions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from numbers import Real
from typing import Any

from .constants import (
    CLASS_NAMES,
    OOD_REVIEW_THRESHOLD,
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD,
    RELIABILITY_THRESHOLD_TEXT,
    DisplayStatus,
    OodReason,
    OodStatus,
    ReliabilityReason,
)
from .errors import PolicyError


@dataclass(frozen=True, slots=True)
class OodDecision:
    score: float | None
    threshold: str
    status: OodStatus
    reason: OodReason
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReliabilityDecision:
    score: float | None
    threshold: str
    status: DisplayStatus
    reason: ReliabilityReason
    requires_manual_review: bool
    policy_version: str = "p03_decision_rule+amendment22"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FinalPolicyDecision:
    status: DisplayStatus
    primary_reason: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    score = float(value)
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _decimal_score(score: float) -> Decimal:
    return Decimal(str(score))


def decide_ood(score_value: object) -> OodDecision:
    """Apply O06 with REFUSE disabled and invalid values failing closed."""

    if score_value is None:
        return OodDecision(
            None,
            OOD_REVIEW_THRESHOLD_TEXT,
            OodStatus.REVIEW_REQUIRED,
            OodReason.OOD_SCORE_UNAVAILABLE,
            False,
        )
    score = _finite_score(score_value)
    if score is None:
        return OodDecision(
            None,
            OOD_REVIEW_THRESHOLD_TEXT,
            OodStatus.REVIEW_REQUIRED,
            OodReason.OOD_SCORE_INVALID,
            False,
        )
    if _decimal_score(score) >= OOD_REVIEW_THRESHOLD:
        return OodDecision(
            score,
            OOD_REVIEW_THRESHOLD_TEXT,
            OodStatus.REVIEW_REQUIRED,
            OodReason.AT_OR_ABOVE_REVIEW_THRESHOLD,
            True,
        )
    return OodDecision(
        score,
        OOD_REVIEW_THRESHOLD_TEXT,
        OodStatus.ACCEPT,
        OodReason.BELOW_REVIEW_THRESHOLD,
        True,
    )


def decide_reliability(
    predicted_category: str,
    score_value: object,
) -> ReliabilityDecision:
    """Apply P03 plus the Amendment 22 no-tumor path."""

    if predicted_category not in CLASS_NAMES:
        raise PolicyError("PREDICTED_CATEGORY_INVALID", repr(predicted_category))
    if predicted_category == "no_tumor":
        return ReliabilityDecision(
            None,
            RELIABILITY_THRESHOLD_TEXT,
            DisplayStatus.REVIEW_REQUIRED,
            ReliabilityReason.SEGMENTATION_RELIABILITY_NOT_APPLICABLE,
            True,
        )
    if score_value is None:
        return ReliabilityDecision(
            None,
            RELIABILITY_THRESHOLD_TEXT,
            DisplayStatus.REVIEW_REQUIRED,
            ReliabilityReason.RELIABILITY_UNAVAILABLE,
            True,
        )
    score = _finite_score(score_value)
    if score is None:
        return ReliabilityDecision(
            None,
            RELIABILITY_THRESHOLD_TEXT,
            DisplayStatus.REVIEW_REQUIRED,
            ReliabilityReason.RELIABILITY_INVALID,
            True,
        )
    if _decimal_score(score) < RELIABILITY_THRESHOLD:
        return ReliabilityDecision(
            score,
            RELIABILITY_THRESHOLD_TEXT,
            DisplayStatus.REVIEW_REQUIRED,
            ReliabilityReason.BELOW_THRESHOLD,
            True,
        )
    return ReliabilityDecision(
        score,
        RELIABILITY_THRESHOLD_TEXT,
        DisplayStatus.STANDARD,
        ReliabilityReason.AT_OR_ABOVE_THRESHOLD,
        False,
    )


def combine_policy(
    ood: OodDecision,
    reliability: ReliabilityDecision,
) -> FinalPolicyDecision:
    """Require both registered gates to accept before returning STANDARD."""

    reasons: list[str] = []
    if ood.status != OodStatus.ACCEPT:
        reasons.append(ood.reason.value)
    if reliability.status != DisplayStatus.STANDARD:
        reasons.append(reliability.reason.value)
    if reasons:
        return FinalPolicyDecision(
            DisplayStatus.REVIEW_REQUIRED,
            reasons[0],
            tuple(reasons),
        )
    return FinalPolicyDecision(
        DisplayStatus.STANDARD,
        ReliabilityReason.AT_OR_ABOVE_THRESHOLD.value,
        (OodReason.BELOW_REVIEW_THRESHOLD.value,),
    )
