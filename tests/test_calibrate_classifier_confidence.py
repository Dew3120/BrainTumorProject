"""Tests for classifier temperature scaling and similarity-aware cross-fitting."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_classifier_confidence import (
    assign_similarity_aware_folds,
    calibration_error,
    equal_frequency_bins,
    fit_temperature,
    probabilities_to_logits,
    temperature_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]


def test_temperature_scaling_preserves_argmax_and_reduces_overconfidence():
    probabilities = np.array(
        [
            [0.99, 0.005, 0.003, 0.002],
            [0.98, 0.01, 0.005, 0.005],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 1, 2, 3])
    logits = probabilities_to_logits(probabilities)
    temperature = fit_temperature(logits, labels)
    calibrated = temperature_probabilities(logits, temperature)
    assert temperature > 1.0
    assert np.array_equal(probabilities.argmax(axis=1), calibrated.argmax(axis=1))
    assert calibrated.max(axis=1).mean() < probabilities.max(axis=1).mean()


def test_similarity_groups_never_cross_folds():
    rows = []
    labels = []
    for label, name in enumerate(("glioma", "meningioma", "pituitary", "no_tumor")):
        for index in range(10):
            rows.append(
                {
                    "path": f"{name}/{index}.jpg",
                    "label": name,
                    "similarity_group": f"{name}-{index // 2}",
                }
            )
            labels.append(label)
    assignment = assign_similarity_aware_folds(rows, np.asarray(labels))
    for group in {row["similarity_group"] for row in rows}:
        folds = {
            int(assignment[index])
            for index, row in enumerate(rows)
            if row["similarity_group"] == group
        }
        assert len(folds) == 1


def test_equal_frequency_ece_matches_weighted_bin_gaps():
    confidence = np.array([0.1, 0.2, 0.8, 0.9])
    correctness = np.array([0.0, 0.0, 1.0, 1.0])
    bins = equal_frequency_bins(confidence, correctness, bins=2)
    ece, mce = calibration_error(bins)
    assert np.isclose(ece, 0.15)
    assert np.isclose(mce, 0.15)


def test_persisted_calibration_is_similarity_aware_and_fail_closed():
    result = json.loads(
        (ROOT / "audit/classifier_calibration.json").read_text(encoding="utf-8")
    )
    assert result["validation_cases"] == 893
    assert result["similarity_groups"] == 784
    assert result["gate"]["status"] == "FAIL"
    assert not result["gate"]["checks"][
        "brier_delta_bootstrap_ci95_upper_below_zero"
    ]
    assert not result["decision_rules"]["temperature_accepted_for_application"]
    assert result["verification"]["test_images_accessed"] == 0
    assert result["verification"]["test_predictions_accessed"] == 0
    assert result["verification"]["threshold_changed"] is False

    fold_path = Path(result["artifacts"]["fold_assignments"])
    with fold_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    group_folds = {}
    for row in rows:
        group_folds.setdefault(row["similarity_group"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in group_folds.values())
