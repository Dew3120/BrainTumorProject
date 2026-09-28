from finalize_stage4_reliability import select_candidate


def test_selection_uses_registered_auroc_then_calibration_order():
    candidates = [
        {
            "experiment": "R02",
            "selection_eligible": True,
            "aggregate_oof_auroc": 0.95,
            "ece": 0.02,
            "brier": 0.05,
        },
        {
            "experiment": "R04",
            "selection_eligible": True,
            "aggregate_oof_auroc": 0.94,
            "ece": 0.01,
            "brier": 0.04,
        },
        {
            "experiment": "R03",
            "selection_eligible": False,
            "aggregate_oof_auroc": 0.99,
            "ece": 0.01,
            "brier": 0.01,
        },
    ]

    assert select_candidate(candidates)["experiment"] == "R02"


def test_selection_uses_experiment_id_as_final_tie_breaker():
    candidates = [
        {
            "experiment": experiment,
            "selection_eligible": True,
            "aggregate_oof_auroc": 0.95,
            "ece": 0.02,
            "brier": 0.05,
        }
        for experiment in ("R04", "R02")
    ]

    assert select_candidate(candidates)["experiment"] == "R02"
