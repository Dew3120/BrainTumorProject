import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_o07_ood_failure_correlation import cohort_metrics


def test_high_ood_quintile_enriches_constructed_failures():
    score = np.arange(10, dtype=float)
    dice = np.array([0.9] * 8 + [0.0, 0.0])
    result = cohort_metrics(score, dice, [f"case-{i}" for i in range(10)])
    assert result["highest_ood_quintile"]["cases"] == 2
    assert result["highest_ood_quintile"]["failures_captured"] == 2
    assert result["failure_detection_auroc"] == 1.0
