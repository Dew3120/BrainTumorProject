import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import r03_extended_feature_contract as extended
import segmentation_smoke_test as smoke


def test_image_transform_matches_segmenter_pixel_for_pixel():
    source = np.arange(73 * 41 * 3, dtype=np.uint32).reshape(41, 73, 3) % 256
    image = Image.fromarray(source.astype(np.uint8), mode="RGB")
    mask = Image.fromarray(np.pad(np.ones((9, 11), dtype=np.uint8) * 255, 15))
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    segmenter_image, _target = smoke.transform_pair(image, mask, size=128)
    expected = segmenter_image.squeeze(0).numpy().astype(np.float64)
    observed = extended.model_space_image(image)
    np.testing.assert_array_equal(observed, expected)


def test_extended_feature_signature_excludes_reference_inputs():
    assert tuple(inspect.signature(extended.extended_prediction_features).parameters) == (
        "probability",
        "model_image",
        "threshold",
    )


def test_extended_features_are_finite_and_symmetric_mask_scores_one():
    probability = np.zeros((128, 128), dtype=np.float64)
    probability[48:80, 48:80] = 0.9
    image = np.tile(np.linspace(0, 1, 128), (128, 1))
    features = extended.extended_prediction_features(probability, image)
    assert tuple(features) == extended.NEW_FEATURES
    assert np.isfinite(list(features.values())).all()
    assert features["predicted_vertical_symmetry"] == 1.0
    assert 0 < features["mask_solidity"] <= 1
    assert features["boundary_fractal_dimension"] > 0


def test_empty_prediction_has_registered_zero_values():
    features = extended.extended_prediction_features(
        np.zeros((128, 128)), np.zeros((128, 128))
    )
    assert features == {name: 0.0 for name in extended.NEW_FEATURES}
