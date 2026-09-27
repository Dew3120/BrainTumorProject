import numpy as np
from run_r09_fold_sensitivity import dispersion_label, generate_assignment


def test_dispersion_label_uses_registered_narrow_and_moderate_limits():
    rules = {
        "narrow": {
            "maximum_auroc_sample_sd": 0.01,
            "maximum_absolute_original_difference": 0.02,
        },
        "moderate": {
            "maximum_auroc_sample_sd": 0.02,
            "maximum_absolute_original_difference": 0.04,
        },
    }
    assert dispersion_label([0.95, 0.96, 0.955], 0.955, rules)["label"] == "NARROW"
    assert dispersion_label([0.93, 0.96, 0.95], 0.955, rules)["label"] == "MODERATE"
    assert dispersion_label([0.90, 0.96, 0.95], 0.955, rules)["label"] == "MATERIAL"


def test_generated_assignment_keeps_groups_in_one_fold():
    groups = np.repeat(np.arange(50), 2)
    success = np.tile([False, True], 50)

    assignment, support = generate_assignment(success, groups, seed=17)

    assert len(support) == 5
    assert set(assignment) == set(range(5))
    assert all(len(set(assignment[groups == group])) == 1 for group in set(groups))
