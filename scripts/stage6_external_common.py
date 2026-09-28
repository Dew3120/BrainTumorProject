"""Fail-closed, outcome-independent adapters for Stage 6 external MRI cohorts."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from run_c03_isotonic import apply_isotonic

ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_SIZE = 128
CLASSIFICATION_SIZE = 224
SEGMENTATION_THRESHOLD = 0.5
RELIABILITY_POLICY_THRESHOLD = 0.9490250173235835
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
R02_FEATURES = (
    "empty_prediction",
    "predicted_area_fraction",
    "whole_probability_mean",
    "whole_probability_sd",
    "predicted_region_entropy_mean",
    "boundary_entropy_mean",
    "high_confidence_foreground_fraction",
    "uncertain_pixel_fraction",
    "bounding_box_fill_ratio",
    "normalized_perimeter",
)
FROZEN_HASHES = {
    "models/r02_reliability_scaler.pkl": "c99a67f8157515d815a9b7a05555b45e26b5a2da0e458b359cc83d63fe30dce8",
    "models/r02_reliability_logistic.pkl": "9f8ea8530c1c2ee6a0773c6ec7c0b92e6da7923ddf3c50d3d32176379339f044",
    "models/r02_reliability_manifest.json": "0d56aab5d562a34325003cee116c4f9d5f8fbe3fcc5278717338706d58e5303f",
    "experiments/c03_isotonic/artifacts/isotonic_mappings.json": "fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa",
}


@dataclass(frozen=True)
class LetterboxGeometry:
    source_width: int
    source_height: int
    canvas_size: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int


@dataclass(frozen=True)
class NiftiGeometry:
    source_shape: tuple[int, int, int]
    canonical_shape: tuple[int, int, int]
    canonical_axcodes: tuple[str, str, str]
    voxel_sizes_mm: tuple[float, float, float]
    affine_determinant: float


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_frozen_artifacts() -> dict[str, str]:
    observed = {}
    for relative, expected in FROZEN_HASHES.items():
        path = ROOT / relative
        value = sha256(path)
        if value != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {relative}")
        observed[relative] = value
    return observed


def letterbox_geometry(width: int, height: int, size: int) -> LetterboxGeometry:
    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError("Image and canvas dimensions must be positive")
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return LetterboxGeometry(
        source_width=width,
        source_height=height,
        canvas_size=size,
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=(size - resized_width) // 2,
        offset_y=(size - resized_height) // 2,
    )


def display_uint8(array: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map a source slice to uint8 before the frozen training transform.

    Native uint8 inputs are preserved. Other MRI arrays use a fixed, label-free
    nonzero 0.5/99.5 percentile window registered before external data access.
    """
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError("Expected one two-dimensional MRI slice")
    if value.dtype == np.uint8:
        return value.copy(), {"method": "native_uint8", "lower": 0.0, "upper": 255.0}
    numeric = np.asarray(value, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Source slice contains non-finite intensities")
    foreground = numeric[numeric != 0]
    sample = foreground if foreground.size else numeric.reshape(-1)
    lower, upper = np.percentile(sample, [0.5, 99.5])
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("Source slice has no usable intensity range")
    scaled = np.clip((numeric - lower) / (upper - lower), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8), {
        "method": "nonzero_percentile_0.5_99.5",
        "lower": float(lower),
        "upper": float(upper),
    }


def _letterbox_pil(image: Image.Image, size: int, resample: int):
    geometry = letterbox_geometry(*image.size, size)
    canvas = Image.new("L", (size, size), 0)
    resized = image.convert("L").resize(
        (geometry.resized_width, geometry.resized_height), resample
    )
    canvas.paste(resized, (geometry.offset_x, geometry.offset_y))
    return canvas, geometry


def preprocess_segmenter(array: np.ndarray):
    display, window = display_uint8(array)
    canvas, geometry = _letterbox_pil(
        Image.fromarray(display, mode="L"),
        SEGMENTATION_SIZE,
        Image.Resampling.BILINEAR,
    )
    tensor = np.asarray(canvas, dtype=np.float32)[None, :, :] / 255.0
    if tensor.shape != (1, 128, 128) or not np.isfinite(tensor).all():
        raise ValueError("Segmenter preprocessing produced an invalid tensor")
    return tensor, geometry, window


def preprocess_classifier(array: np.ndarray):
    display, window = display_uint8(array)
    canvas, geometry = _letterbox_pil(
        Image.fromarray(display, mode="L"),
        CLASSIFICATION_SIZE,
        Image.Resampling.BILINEAR,
    )
    gray = np.asarray(canvas, dtype=np.float32) / 255.0
    rgb = np.repeat(gray[None, :, :], 3, axis=0)
    tensor = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    if tensor.shape != (3, 224, 224) or not np.isfinite(tensor).all():
        raise ValueError("Classifier preprocessing produced an invalid tensor")
    return tensor.astype(np.float32), geometry, window


def preprocess_reference_mask(mask: np.ndarray, *, allow_empty: bool = True):
    value = np.asarray(mask)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Reference mask must be a finite two-dimensional array")
    unique = np.unique(value)
    if not np.all(np.isin(unique, [0, 1, 255])):
        raise ValueError("Reference mask must be binary")
    binary = value > 0
    if not allow_empty and not binary.any():
        raise ValueError("A tumor-bearing reference slice cannot be empty")
    image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    canvas, geometry = _letterbox_pil(
        image, SEGMENTATION_SIZE, Image.Resampling.NEAREST
    )
    output = np.asarray(canvas, dtype=np.uint8) > 0
    if not allow_empty and not output.any():
        raise ValueError("Reference mask disappeared during resize")
    return output[None, :, :], geometry


def restore_to_source(model_array: np.ndarray, geometry: LetterboxGeometry, *, mask: bool):
    value = np.asarray(model_array)
    if value.shape == (1, geometry.canvas_size, geometry.canvas_size):
        value = value[0]
    if value.shape != (geometry.canvas_size, geometry.canvas_size):
        raise ValueError("Model-space array does not match recorded canvas")
    y0, x0 = geometry.offset_y, geometry.offset_x
    crop = value[y0 : y0 + geometry.resized_height, x0 : x0 + geometry.resized_width]
    if crop.shape != (geometry.resized_height, geometry.resized_width):
        raise ValueError("Recorded letterbox crop is invalid")
    if mask:
        image = Image.fromarray(np.asarray(crop, dtype=np.uint8) * 255, mode="L")
        restored = image.resize(
            (geometry.source_width, geometry.source_height), Image.Resampling.NEAREST
        )
        return np.asarray(restored, dtype=np.uint8) > 0
    image = Image.fromarray(np.asarray(crop, dtype=np.float32), mode="F")
    restored = image.resize(
        (geometry.source_width, geometry.source_height), Image.Resampling.BILINEAR
    )
    output = np.asarray(restored, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Restored probability map is non-finite")
    return output


def load_canonical_nifti(path: Path):
    image = nib.load(str(path))
    if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
        raise ValueError("NIfTI input must be a non-empty 3D volume")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("NIfTI affine is invalid or singular")
    canonical = nib.as_closest_canonical(image)
    data = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("NIfTI voxel array contains non-finite values")
    axcodes = tuple(str(code) for code in nib.aff2axcodes(canonical.affine))
    if axcodes != ("R", "A", "S"):
        raise ValueError(f"Canonical NIfTI orientation is not RAS: {axcodes}")
    voxel_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(canonical.affine))
    geometry = NiftiGeometry(
        source_shape=tuple(int(value) for value in image.shape),
        canonical_shape=tuple(int(value) for value in canonical.shape),
        canonical_axcodes=axcodes,
        voxel_sizes_mm=voxel_sizes,
        affine_determinant=float(np.linalg.det(canonical.affine[:3, :3])),
    )
    return data, canonical.affine.copy(), geometry


def sort_dicom_slices(datasets):
    if not datasets:
        raise ValueError("DICOM series is empty")
    orientations = np.asarray(
        [[float(value) for value in item.ImageOrientationPatient] for item in datasets],
        dtype=np.float64,
    )
    if orientations.shape[1] != 6 or not np.allclose(orientations, orientations[0], atol=1e-6):
        raise ValueError("DICOM slice orientations differ")
    row, column = orientations[0, :3], orientations[0, 3:]
    normal = np.cross(row, column)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("DICOM orientation vectors are degenerate")
    normal /= norm
    positions = np.asarray(
        [[float(value) for value in item.ImagePositionPatient] for item in datasets],
        dtype=np.float64,
    )
    projections = positions @ normal
    order = np.argsort(projections, kind="stable")
    sorted_projection = projections[order]
    if len(sorted_projection) > 1 and np.any(np.diff(sorted_projection) <= 1e-6):
        raise ValueError("DICOM slice positions are duplicate or non-monotonic")
    frames = {str(item.FrameOfReferenceUID) for item in datasets}
    if len(frames) != 1:
        raise ValueError("DICOM slices use multiple frames of reference")
    return [datasets[int(index)] for index in order], sorted_projection, normal


def apply_operational_c03(raw_probability: np.ndarray) -> np.ndarray:
    verify_frozen_artifacts()
    mappings = json.loads(
        (ROOT / "experiments/c03_isotonic/artifacts/isotonic_mappings.json").read_text(
            encoding="utf-8"
        )
    )
    mapping = mappings["final_all_validation_mapping"]
    _logits, probability = apply_isotonic(
        np.asarray(raw_probability, dtype=np.float64), mapping
    )
    if probability.shape != np.asarray(raw_probability).shape:
        raise ValueError("C03 calibration changed probability shape")
    return probability


def operational_reliability(calibrated_probability: np.ndarray):
    verify_frozen_artifacts()
    from fit_segmenter_case_reliability import prediction_features

    values = prediction_features(calibrated_probability, SEGMENTATION_THRESHOLD)
    vector = np.asarray([[values[name] for name in R02_FEATURES]], dtype=np.float64)
    with (ROOT / "models/r02_reliability_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (ROOT / "models/r02_reliability_logistic.pkl").open("rb") as handle:
        logistic = pickle.load(handle)
    score = float(logistic.predict_proba(scaler.transform(vector))[0, 1])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Operational reliability score is invalid")
    status = "STANDARD" if score >= RELIABILITY_POLICY_THRESHOLD else "REVIEW_REQUIRED"
    return {
        "score": score,
        "threshold": "0.9490250173235835",
        "status": status,
        "features": {name: float(values[name]) for name in R02_FEATURES},
    }


def geometry_dict(geometry):
    return asdict(geometry)
