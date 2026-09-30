import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_r12_external_baselines import coverage_rows, make_comparator


def test_comparator_contract_and_low_risk_coverage():
    model = make_comparator()
    assert model.named_steps["logisticregression"].C == 1.0
    reliability = np.array([0.9, 0.1, 0.8, 0.2])
    dice = np.array([0.8, 0.0, 0.7, 0.1])
    rows = coverage_rows("x", "m", reliability, dice, ["b", "a", "d", "c"], 2)
    half = next(row for row in rows if row["selection"] == "coverage_0.5")
    assert half["accepted_count"] == 2
    assert half["accepted_mean_dice"] == 0.75
