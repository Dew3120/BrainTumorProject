import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c07_hierarchical import apply_candidate, fit_scalar


def test_convex_scalar_fit_is_positive_and_bounded():
    logits = np.array([[[2.0, 1.0], [-1.0, -2.0]]])
    targets = np.array([[[True, True], [False, False]]])
    regions = np.ones_like(targets, dtype=bool)
    fitted = fit_scalar(
        logits, targets, regions, np.array([0]), "whole_image", np.ones(1)
    )
    assert 0.05 <= fitted["temperature"] <= 20.0


def test_hierarchical_application_uses_route_and_region_cells():
    logits = np.array([[[2.0, 2.0], [-2.0, -2.0]]])
    regions = np.array([[[True, False], [True, False]]])
    routes = np.array(["glioma"])
    model = {
        "specification": {
            "region_split": True,
            "reliability_weighted": False,
            "category_routed": True,
        },
        "cells": {
            "all::lesion_centered": {"temperature": 1.0},
            "all::background": {"temperature": 1.0},
            "glioma::lesion_centered": {"temperature": 2.0},
            "glioma::background": {"temperature": 0.5},
        },
    }
    transformed, _probability = apply_candidate(
        model, logits, regions, routes, np.array([0])
    )
    np.testing.assert_allclose(transformed, [[[1.0, 4.0], [-1.0, -4.0]]])
