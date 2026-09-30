import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_p06_joint_policy import joint_status


def test_joint_policy_boundary_and_fail_closed():
    ood = 0.36854275476100085
    reliability = 0.9490250173235835
    assert joint_status(ood - 1e-12, reliability, ood, reliability)[0] == "STANDARD"
    assert joint_status(ood, reliability, ood, reliability)[1] == "OOD_REVIEW"
    assert joint_status(0.0, reliability - 1e-12, ood, reliability)[1] == "RELIABILITY_REVIEW"
    assert joint_status(math.nan, reliability, ood, reliability)[1] == "INVALID_SCORE"
