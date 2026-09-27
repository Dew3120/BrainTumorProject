import numpy as np
from run_r07_coefficient_bootstrap import coefficient_summary, compare_r01


def test_coefficient_summary_ranks_by_absolute_bootstrap_mean_and_counts_signs():
    folds = np.asarray(
        [
            [0.3, -1.0],
            [0.4, -0.8],
            [0.5, -0.9],
            [0.6, -1.1],
            [0.7, -1.2],
        ]
    )
    bootstraps = np.asarray(
        [
            [0.2, -0.8],
            [0.4, -1.0],
            [-0.1, -1.2],
            [0.6, -1.4],
        ]
    )

    rows = coefficient_summary(
        ("smaller", "larger"), folds, bootstraps, np.asarray([0.5, -1.0])
    )

    assert [row["feature"] for row in rows] == ["larger", "smaller"]
    smaller = next(row for row in rows if row["feature"] == "smaller")
    assert smaller["five_fold_sign_agreement_count"] == 5
    assert smaller["bootstrap_sign_agreement_count"] == 3
    assert smaller["bootstrap_sign_agreement_fraction"] == 0.75


def test_r01_comparison_reports_mean_sign_change_without_causal_claim():
    r01 = np.asarray([[-1.0], [-0.8], [-0.6], [-0.4], [-0.2]])
    r02 = np.asarray([[0.2], [0.4], [0.6], [0.8], [1.0]])

    row = compare_r01(("feature",), r01, r02, np.asarray([0.5]))[0]

    assert row["mean_sign_changed"] is True
    assert row["r01_negative_folds"] == 5
    assert row["r02_positive_folds"] == 5
