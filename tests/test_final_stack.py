from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_source_runtime():
    path = ROOT / "scripts/final_stack_runtime.py"
    spec = importlib.util.spec_from_file_location("source_final_stack_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prediction_features_empty_contract():
    runtime = load_source_runtime()
    features = runtime.prediction_features(np.zeros((128, 128)), 0.5)
    assert list(features) == list(runtime.R02_FEATURES)
    assert features["empty_prediction"] == 1.0
    assert features["predicted_area_fraction"] == 0.0


def test_prediction_features_nonempty_are_finite():
    runtime = load_source_runtime()
    probability = np.zeros((128, 128), dtype=np.float64)
    probability[40:80, 50:90] = 0.9
    features = runtime.prediction_features(probability, 0.5)
    assert list(features) == list(runtime.R02_FEATURES)
    assert features["empty_prediction"] == 0.0
    assert all(np.isfinite(value) for value in features.values())


def test_threshold_strings_are_exact():
    verifier_path = ROOT / "scripts/verify_final_stack.py"
    spec = importlib.util.spec_from_file_location("source_final_stack_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    assert verifier.EXPECTED_THRESHOLDS == {
        "segmentation": "0.50",
        "reliability": "0.9490250173235835",
        "ood_review": "0.36854275476100085",
        "ood_refuse": None,
    }
