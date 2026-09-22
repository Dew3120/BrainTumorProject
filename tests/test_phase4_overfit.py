"""Decision rules must not silently accept one easy mask or use validation scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase4_overfit import decision


def row(mean=0.91, minimum=0.81, validation=0.0):
    return {"train_dice": mean, "train_min_dice": minimum, "val_dice": validation}


def test_requires_five_consecutive_epochs_and_all_six_images():
    assert decision([row()] * 4) == "CONTINUE"
    assert decision([row()] * 5) == "MEMORIZATION_PASS"
    assert decision([row(minimum=0.79)] * 5) == "CONTINUE"
    assert decision([row()] * 4 + [row(mean=0.89)]) == "CONTINUE"


def test_validation_does_not_select_or_stop_run():
    assert decision([row(validation=0.0)] * 5) == decision([row(validation=1.0)] * 5)


def test_low_plateau_and_budget_limits():
    assert decision([row(mean=0.02, minimum=0.0)] * 49) == "CONTINUE"
    assert (
        decision([row(mean=0.02, minimum=0.0)] * 50) == "STOP_AND_DIAGNOSE_LOW_PLATEAU"
    )
    assert (
        decision([row(mean=0.7, minimum=0.3)] * 100)
        == "BUDGET_EXHAUSTED_DIAGNOSE_BEFORE_FULL_TRAINING"
    )
