import pytest
from run_r08_prospective_fold_evaluation import gate_passes, summarize_fold_metrics


def test_r08_gate_uses_inclusive_registered_boundary():
    assert gate_passes(0.941, 0.941)
    assert not gate_passes(0.940999, 0.941)


def test_fold_summary_is_unweighted_with_sample_sd():
    rows = [
        {
            "failure_detection_auroc": value,
            "ece": value,
            "mce": value,
            "brier": value,
            "prevalence_baseline_brier": value,
            "brier_skill": value,
        }
        for value in (0.8, 0.9, 1.0, 0.9, 0.9)
    ]

    summary = summarize_fold_metrics(rows)

    assert summary["failure_detection_auroc"]["mean"] == pytest.approx(0.9)
    assert summary["failure_detection_auroc"]["sample_sd"] == pytest.approx(
        0.0707106781
    )
