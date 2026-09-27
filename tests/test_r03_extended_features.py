import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_r03_extended_features import auc_many, gate_passes


def test_r03_gate_includes_exact_boundary():
    assert gate_passes(0.9595028142589118, 0.9595028142589118)
    assert not gate_passes(0.9595028142589117, 0.9595028142589118)


def test_r03_vectorized_auc_matches_sklearn_with_ties():
    failure = np.array([False, True, False, True, True, False])
    scores = np.array(
        [
            [0.1, 0.8, 0.2, 0.8, 0.9, 0.1],
            [0.2, 0.7, 0.2, 0.6, 0.6, 0.1],
        ]
    )
    expected = np.array([roc_auc_score(failure, row) for row in scores])
    np.testing.assert_allclose(auc_many(scores, failure), expected)
