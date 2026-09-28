import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r02_operational_artifacts_are_self_consistent():
    manifest_path = ROOT / "models/r02_reliability_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "OPERATIONAL_ARTIFACT_FROZEN"
    assert len(manifest["feature_order"]) == 10
    assert manifest["population"]["cases"] == 680
    assert manifest["verification"]["external_images_accessed"] == 0
    assert manifest["verification"]["brisc_test_images_accessed"] == 0
    assert not manifest["verification"]["neural_model_retrained"]
    assert not manifest["verification"]["segmentation_threshold_changed"]
    assert manifest["verification"]["r07_coefficient_max_abs_error"] <= 1e-12
    assert max(manifest["verification"]["independent_fit_max_abs_errors"].values()) <= 1e-12
    assert manifest["verification"]["serialization_reload_probability_max_abs_error"] <= 1e-12
    for artifact in manifest["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.exists()
        assert sha256(path) == artifact["sha256"]


def test_r02_pickles_reproduce_reference_probabilities():
    manifest_path = ROOT / "models/r02_reliability_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with (ROOT / manifest["artifacts"]["scaler"]["path"]).open("rb") as handle:
        scaler = pickle.load(handle)
    with (ROOT / manifest["artifacts"]["logistic"]["path"]).open("rb") as handle:
        logistic = pickle.load(handle)
    assert np.array_equal(scaler.mean_, np.asarray(manifest["scaler_mean"]))
    assert np.array_equal(scaler.scale_, np.asarray(manifest["scaler_scale"]))
    assert np.array_equal(logistic.coef_[0], np.asarray(manifest["standardized_coefficients"]))
    assert float(logistic.intercept_[0]) == manifest["intercept"]
    reference = ROOT / manifest["artifacts"]["reference_predictions"]["path"]
    with reference.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 680
    assert all(0.0 <= float(row["operational_probability"]) <= 1.0 for row in rows)
