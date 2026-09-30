import math

from gyrus.domain.constants import DisplayStatus, OodStatus, ReliabilityReason
from gyrus.domain.policy import combine_policy, decide_ood, decide_reliability


def test_reliability_exact_boundary_is_standard():
    threshold = float("0.9490250173235835")
    below = math.nextafter(threshold, -math.inf)
    assert decide_reliability("glioma", threshold).status == DisplayStatus.STANDARD
    assert decide_reliability("glioma", below).status == DisplayStatus.REVIEW_REQUIRED


def test_ood_exact_boundary_requires_review():
    threshold = float("0.36854275476100085")
    below = math.nextafter(threshold, -math.inf)
    assert decide_ood(threshold).status == OodStatus.REVIEW_REQUIRED
    assert decide_ood(below).status == OodStatus.ACCEPT


def test_no_tumor_ignores_any_supplied_segmentation_reliability():
    decision = decide_reliability("no_tumor", 1.0)
    assert decision.score is None
    assert decision.status == DisplayStatus.REVIEW_REQUIRED
    assert (
        decision.reason
        == ReliabilityReason.SEGMENTATION_RELIABILITY_NOT_APPLICABLE
    )


def test_invalid_scores_fail_closed():
    assert decide_ood(float("nan")).status == OodStatus.REVIEW_REQUIRED
    assert not decide_ood(float("nan")).valid
    assert decide_reliability("pituitary", True).status == DisplayStatus.REVIEW_REQUIRED


def test_joint_policy_requires_both_gates():
    standard = combine_policy(decide_ood(0.1), decide_reliability("pituitary", 1.0))
    reviewed = combine_policy(decide_ood(0.9), decide_reliability("pituitary", 1.0))
    assert standard.status == DisplayStatus.STANDARD
    assert reviewed.status == DisplayStatus.REVIEW_REQUIRED
