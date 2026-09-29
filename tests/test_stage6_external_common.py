from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import stage6_external_common as external


def test_segmenter_preprocessing_matches_registered_letterbox_geometry():
    source = np.arange(40 * 80, dtype=np.uint16).reshape(40, 80)
    tensor, geometry, window = external.preprocess_segmenter(source)
    assert tensor.shape == (1, 128, 128)
    assert tensor.dtype == np.float32
    assert 0 <= tensor.min() <= tensor.max() <= 1
    assert geometry.resized_width == 128
    assert geometry.resized_height == 64
    assert geometry.offset_x == 0
    assert geometry.offset_y == 32
    assert window["method"] == "nonzero_percentile_0.5_99.5"
    assert np.all(tensor[:, :32] == 0)
    assert np.all(tensor[:, 96:] == 0)


def test_classifier_preprocessing_uses_same_display_bridge_and_imagenet_shape():
    source = np.arange(30 * 60, dtype=np.uint16).reshape(30, 60)
    tensor, geometry, window = external.preprocess_classifier(source)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert geometry.resized_height == 112
    assert window["method"] == "nonzero_percentile_0.5_99.5"


def test_uint8_bridge_is_identity():
    source = np.arange(100, dtype=np.uint8).reshape(10, 10)
    observed, record = external.display_uint8(source)
    assert np.array_equal(observed, source)
    assert record["method"] == "native_uint8"


def test_mask_uses_nearest_neighbor_and_round_trips_geometry():
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[10:30, 20:60] = 1
    transformed, geometry = external.preprocess_reference_mask(mask, allow_empty=False)
    restored = external.restore_to_source(transformed, geometry, mask=True)
    assert transformed.dtype == bool
    assert restored.shape == mask.shape
    assert restored.dtype == bool
    assert restored.any()


def test_empty_mask_policy_is_explicit():
    mask = np.zeros((16, 16), dtype=np.uint8)
    transformed, _geometry = external.preprocess_reference_mask(mask, allow_empty=True)
    assert not transformed.any()
    with pytest.raises(ValueError, match="cannot be empty"):
        external.preprocess_reference_mask(mask, allow_empty=False)


def test_nonfinite_and_constant_inputs_fail_closed():
    with pytest.raises(ValueError, match="non-finite"):
        external.display_uint8(np.asarray([[np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="usable intensity range"):
        external.display_uint8(np.ones((8, 8), dtype=np.float32))


def test_canonical_nifti_contract(tmp_path: Path):
    path = tmp_path / "synthetic.nii.gz"
    data = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), path)
    canonical, canonical_affine, geometry = external.load_canonical_nifti(path)
    assert canonical.shape == (5, 6, 7)
    assert np.isfinite(canonical_affine).all()
    assert geometry.canonical_axcodes == ("R", "A", "S")
    assert geometry.voxel_sizes_mm == (2.0, 3.0, 4.0)


def test_constant_uint8_nifti_preserves_dtype_and_uses_native_adapter(tmp_path: Path):
    path = tmp_path / "constant_uint8.nii.gz"
    data = np.zeros((8, 9, 3), dtype=np.uint8)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    canonical, _affine, _geometry = external.load_canonical_nifti(path)
    assert canonical.dtype == np.uint8
    segmenter, _segmenter_geometry, segmenter_window = (
        external.preprocess_segmenter(canonical[:, :, 0])
    )
    classifier, _classifier_geometry, classifier_window = (
        external.preprocess_classifier(canonical[:, :, 0])
    )
    assert segmenter_window["method"] == "native_uint8"
    assert classifier_window["method"] == "native_uint8"
    assert segmenter.shape == (1, 128, 128)
    assert classifier.shape == (3, 224, 224)
    assert np.isfinite(segmenter).all()
    assert np.isfinite(classifier).all()


class FakeDicom:
    def __init__(self, z):
        self.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        self.ImagePositionPatient = [0, 0, z]
        self.FrameOfReferenceUID = "1.2.3"


def test_dicom_sort_is_geometry_based():
    datasets = [FakeDicom(4), FakeDicom(0), FakeDicom(2)]
    ordered, projection, normal = external.sort_dicom_slices(datasets)
    assert [item.ImagePositionPatient[2] for item in ordered] == [0, 2, 4]
    assert np.array_equal(projection, [0, 2, 4])
    assert np.allclose(normal, [0, 0, 1])


def test_operational_c03_and_r02_contract():
    raw = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float64).reshape(128, 128)
    calibrated = external.apply_operational_c03(raw)
    assert calibrated.shape == raw.shape
    assert np.isfinite(calibrated).all()
    assert 0 <= calibrated.min() <= calibrated.max() <= 1
    output = external.operational_reliability(calibrated)
    assert set(output) == {"score", "threshold", "status", "features"}
    assert output["threshold"] == "0.9490250173235835"
    assert output["status"] in {"STANDARD", "REVIEW_REQUIRED"}
    assert tuple(output["features"]) == external.R02_FEATURES


def test_frozen_hashes_and_thresholds_are_locked():
    observed = external.verify_frozen_artifacts()
    assert observed == external.FROZEN_HASHES
    assert external.SEGMENTATION_THRESHOLD == 0.5
    assert external.RELIABILITY_POLICY_THRESHOLD == 0.9490250173235835
