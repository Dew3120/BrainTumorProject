import numpy as np
from run_r10_seed_stability import make_seeded_model, stability_label


def test_seeded_model_changes_only_registered_random_state():
    model = make_seeded_model(168)

    logistic = model.named_steps["logistic"]
    assert logistic.random_state == 168
    assert logistic.C == 1.0
    assert logistic.solver == "lbfgs"
    assert logistic.max_iter == 5000


def test_stability_label_prefers_exact_then_numeric_then_sensitive():
    rules = {
        "exact_probability_tolerance": 1e-12,
        "numerically_stable_maximum_auroc_sample_sd": 0.001,
        "numerically_stable_maximum_absolute_r02_auroc_difference": 0.001,
    }
    assert stability_label(0.0, [0.95] * 5, 0.95, rules)["label"] == "EXACT"
    assert (
        stability_label(1e-6, [0.9500, 0.9501, 0.9499, 0.95, 0.95], 0.95, rules)[
            "label"
        ]
        == "NUMERICALLY_STABLE"
    )
    assert (
        stability_label(0.01, np.array([0.94, 0.95, 0.96, 0.95, 0.95]), 0.95, rules)[
            "label"
        ]
        == "SEED_SENSITIVE"
    )
