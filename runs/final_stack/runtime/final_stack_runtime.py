"""Self-contained runtime for the immutable Stage 8 inference package."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import UNet
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.special import expit
from scipy.stats import kurtosis, skew
from skimage.feature import canny, corner_harris, corner_peaks
from torchvision.models import resnet18

SEGMENTATION_SIZE = 128
CLASSIFICATION_SIZE = 224
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
CLASS_NAMES = ("glioma", "meningioma", "pituitary", "no_tumor")
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
O02_FEATURES = (
    "histogram_entropy",
    "intensity_mean",
    "intensity_sd",
    "intensity_skew",
    "intensity_kurtosis",
    "fraction_q1",
    "fraction_q2",
    "fraction_q3",
    "fraction_q4",
    "aspect_ratio",
    "edge_density",
    "harris_corner_count",
    "file_bytes",
)
O05_SCORE_ORDER = (
    "o02_probability",
    "o03_mahalanobis",
    "o04_classifier",
    "o04_segmenter_entropy",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _segmenter() -> torch.nn.Module:
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=0.0,
    )
    model.model[2] = torch.nn.ConvTranspose2d(
        32, 1, 3, stride=2, padding=1, output_padding=1
    )
    return model


def _load_state(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _display_uint8(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2:
        raise ValueError("Expected one two-dimensional image")
    if value.dtype == np.uint8:
        return value.copy()
    numeric = np.asarray(value, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Input has non-finite intensities")
    nonzero = numeric[numeric != 0]
    sample = nonzero if nonzero.size else numeric.reshape(-1)
    lower, upper = np.percentile(sample, [0.5, 99.5])
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("Input has no usable intensity range")
    scaled = np.clip((numeric - lower) / (upper - lower), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _letterbox(value: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(value, mode="L")
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def preprocess_segmenter(value: np.ndarray) -> np.ndarray:
    return _letterbox(_display_uint8(value), SEGMENTATION_SIZE)[None, :, :]


def preprocess_classifier(value: np.ndarray) -> np.ndarray:
    gray = _letterbox(_display_uint8(value), CLASSIFICATION_SIZE)
    rgb = np.repeat(gray[None, :, :], 3, axis=0)
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -clipped * np.log(clipped) - (1 - clipped) * np.log(1 - clipped)


def prediction_features(probability: np.ndarray, threshold: float) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (128, 128) or not np.isfinite(probability).all():
        raise ValueError("Reliability features require one finite 128x128 map")
    hard = probability >= threshold
    area = int(hard.sum())
    entropy = binary_entropy(probability)
    uncertain = (probability >= 0.40) & (probability <= 0.60)
    if area == 0:
        return {
            "empty_prediction": 1.0,
            "predicted_area_fraction": 0.0,
            "whole_probability_mean": float(probability.mean()),
            "whole_probability_sd": float(probability.std()),
            "predicted_region_entropy_mean": 0.0,
            "boundary_entropy_mean": 0.0,
            "high_confidence_foreground_fraction": 0.0,
            "uncertain_pixel_fraction": float(uncertain.mean()),
            "bounding_box_fill_ratio": 0.0,
            "normalized_perimeter": 0.0,
        }
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(hard, structure=structure, iterations=2, border_value=0)
    dilated = binary_dilation(hard, structure=structure, iterations=2, border_value=0)
    boundary = dilated ^ eroded
    one_eroded = binary_erosion(hard, structure=structure, iterations=1, border_value=0)
    perimeter = int((hard & ~one_eroded).sum())
    y, x = np.where(hard)
    box_area = (int(y.max()) - int(y.min()) + 1) * (int(x.max()) - int(x.min()) + 1)
    region = probability[hard]
    return {
        "empty_prediction": 0.0,
        "predicted_area_fraction": float(area / hard.size),
        "whole_probability_mean": float(probability.mean()),
        "whole_probability_sd": float(probability.std()),
        "predicted_region_entropy_mean": float(entropy[hard].mean()),
        "boundary_entropy_mean": float(entropy[boundary].mean()),
        "high_confidence_foreground_fraction": float((region >= 0.90).mean()),
        "uncertain_pixel_fraction": float(uncertain.mean()),
        "bounding_box_fill_ratio": float(area / box_area),
        "normalized_perimeter": float(perimeter / np.sqrt(area)),
    }


def o02_features(gray_uint8: np.ndarray, file_bytes: int) -> dict[str, float]:
    gray = np.asarray(gray_uint8, dtype=np.float64) / 255.0
    hist = np.histogram(gray, bins=256, range=(0, 1))[0].astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    flat = gray.ravel()
    corners = corner_peaks(corner_harris(gray), min_distance=2, threshold_rel=0.01)
    return {
        "histogram_entropy": float(-(p * np.log2(p)).sum()),
        "intensity_mean": float(flat.mean()),
        "intensity_sd": float(flat.std()),
        "intensity_skew": float(np.nan_to_num(skew(flat), nan=0.0, posinf=0.0, neginf=0.0)),
        "intensity_kurtosis": float(np.nan_to_num(kurtosis(flat), nan=0.0, posinf=0.0, neginf=0.0)),
        "fraction_q1": float((flat < 0.25).mean()),
        "fraction_q2": float(((flat >= 0.25) & (flat < 0.5)).mean()),
        "fraction_q3": float(((flat >= 0.5) & (flat < 0.75)).mean()),
        "fraction_q4": float((flat >= 0.75).mean()),
        "aspect_ratio": float(gray.shape[1] / gray.shape[0]),
        "edge_density": float(canny(gray, sigma=1.0).mean()),
        "harris_corner_count": float(len(corners)),
        "file_bytes": float(file_bytes),
    }


def _manual_logistic(vector, mean, scale, coefficient, intercept) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    z = (vector - np.asarray(mean)) / np.asarray(scale)
    return float(expit(float(z @ np.asarray(coefficient) + intercept)))


class FinalStackRuntime:
    """Load and run inference using files inside one frozen package only."""

    def __init__(self, root: str | Path, *, verify_hashes: bool = True):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if verify_hashes:
            self.verify_hashes()
        thresholds = self.manifest["thresholds"]
        self.segmentation_threshold = float(thresholds["segmentation"])
        self.reliability_threshold = float(thresholds["reliability"])
        self.ood_threshold = float(thresholds["ood_review"])
        self.classifier = _load_state(
            resnet18(weights=None, num_classes=4), self.root / "models/classifier_v1.pt"
        )
        self.segmenters = {
            "e05": _load_state(_segmenter(), self.root / "models/segmenter_e05_weighted.pt"),
            "e06": _load_state(_segmenter(), self.root / "models/segmenter_e06_boundary.pt"),
            "e07": _load_state(_segmenter(), self.root / "models/segmenter_e07_glioma_specialist.pt"),
            "v1": _load_state(_segmenter(), self.root / "models/segmenter_v1_fallback.pt"),
        }
        module_name = self.manifest["ood"]["o03_bottleneck_module"]
        modules = dict(self.segmenters["v1"].named_modules())
        if module_name not in modules:
            raise ValueError("Frozen O03 bottleneck module is missing")
        self._bottleneck = None
        self._hook = modules[module_name].register_forward_hook(self._capture_bottleneck)
        self.c03 = json.loads(
            (self.root / "calibration/c03_isotonic_mappings.json").read_text(encoding="utf-8")
        )["final_all_validation_mapping"]
        with (self.root / "reliability/r02_reliability_scaler.pkl").open("rb") as handle:
            self.r02_scaler = pickle.load(handle)
        with (self.root / "reliability/r02_reliability_logistic.pkl").open("rb") as handle:
            self.r02_logistic = pickle.load(handle)
        self.o02 = json.loads((self.root / "ood/o02_state.json").read_text(encoding="utf-8"))
        self.o05 = json.loads((self.root / "ood/o05_state.json").read_text(encoding="utf-8"))
        o03 = np.load(self.root / "ood/o03_mahalanobis_state.npz")
        self.o03_location = o03["location"].astype(np.float64)
        self.o03_precision = o03["precision"].astype(np.float64)
        refs = np.load(self.root / "ood/o05_normalization_reference.npz")
        self.o05_references = {name: refs[name].astype(np.float64) for name in O05_SCORE_ORDER}

    def _capture_bottleneck(self, _module, _inputs, output):
        self._bottleneck = output.detach()

    def close(self):
        self._hook.remove()

    def verify_hashes(self) -> None:
        expected = {item["path"]: item for item in self.manifest["artifacts"]}
        observed = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if observed != set(expected):
            raise ValueError(
                f"Frozen allowlist mismatch: missing={sorted(set(expected)-observed)}, "
                f"unexpected={sorted(observed-set(expected))}"
            )
        for relative, item in expected.items():
            path = self.root / relative
            if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"Frozen artifact mismatch: {relative}")

    def _cdf(self, name: str, value: float) -> float:
        reference = self.o05_references[name]
        return float(np.searchsorted(reference, value, side="right") / len(reference))

    def infer_path(self, path: str | Path) -> dict:
        path = Path(path)
        with Image.open(path) as image:
            native = np.asarray(image.convert("L"), dtype=np.uint8)
        return self.infer_array(native, file_bytes=path.stat().st_size, input_sha256=sha256(path))

    def infer_array(self, value: np.ndarray, *, file_bytes: int, input_sha256: str) -> dict:
        display = _display_uint8(value)
        seg = torch.from_numpy(preprocess_segmenter(display)[None]).float()
        cls = torch.from_numpy(preprocess_classifier(display)[None]).float()
        self._bottleneck = None
        with torch.inference_mode():
            logits = self.classifier(cls)
            class_probability = logits.softmax(dim=1)[0].cpu().numpy().astype(np.float64)
            predicted_index = int(np.argmax(class_probability))
            e05 = self.segmenters["e05"](seg).sigmoid()[0, 0]
            e06 = self.segmenters["e06"](seg).sigmoid()[0, 0]
            specialist = self.segmenters["e07"](seg).sigmoid()[0, 0]
            self._bottleneck = None
            fallback = self.segmenters["v1"](seg).sigmoid()[0, 0]
            if self._bottleneck is None or self._bottleneck.shape[1] != 128:
                raise ValueError("O03 bottleneck capture failed")
            embedding = self._bottleneck.mean(dim=(2, 3))[0].cpu().numpy().astype(np.float64)
        routed = specialist if predicted_index == 0 else fallback
        raw = ((e05 + e06 + routed) / 3.0).cpu().numpy().astype(np.float64)
        x = np.asarray(self.c03["x_thresholds"], dtype=np.float64)
        y = np.asarray(self.c03["y_thresholds"], dtype=np.float64)
        calibrated = np.interp(raw.reshape(-1), x, y).reshape(raw.shape)
        hard = calibrated >= self.segmentation_threshold

        feature_values = o02_features(display, file_bytes)
        o02_vector = [feature_values[name] for name in O02_FEATURES]
        o02_score = _manual_logistic(
            o02_vector,
            self.o02["scaler_mean"],
            self.o02["scaler_scale"],
            self.o02["coefficients"],
            self.o02["intercept"],
        )
        centered = embedding - self.o03_location
        o03_score = float(centered @ self.o03_precision @ centered)
        o04_classifier = float(1.0 - class_probability.max())
        o04_segmenter = float(binary_entropy(raw).mean())
        component = {
            "o02_probability": o02_score,
            "o03_mahalanobis": o03_score,
            "o04_classifier": o04_classifier,
            "o04_segmenter_entropy": o04_segmenter,
        }
        normalized = [self._cdf(name, component[name]) for name in O05_SCORE_ORDER]
        ood_score = _manual_logistic(
            normalized,
            self.o05["scaler_mean"],
            self.o05["scaler_scale"],
            self.o05["coefficients"],
            self.o05["intercept"],
        )
        ood_status = "ACCEPT" if ood_score < self.ood_threshold else "REVIEW"

        r02_values = prediction_features(calibrated, self.segmentation_threshold)
        r02_vector = np.asarray([[r02_values[name] for name in R02_FEATURES]], dtype=np.float64)
        reliability = float(
            self.r02_logistic.predict_proba(self.r02_scaler.transform(r02_vector))[0, 1]
        )
        reliability_status = (
            "STANDARD" if reliability >= self.reliability_threshold else "REVIEW_REQUIRED"
        )
        return {
            "input_sha256": input_sha256,
            "classifier": {
                "class_order": list(CLASS_NAMES),
                "probabilities": class_probability.tolist(),
                "predicted_class": CLASS_NAMES[predicted_index],
                "probability_source": "raw",
            },
            "segmenter": {
                "route": "glioma_specialist" if predicted_index == 0 else "v1_fallback",
                "raw_probability_sha256": array_sha256(raw),
                "calibrated_probability_sha256": array_sha256(calibrated),
                "hard_mask_sha256": array_sha256(hard.astype(np.uint8)),
                "hard_foreground_pixels": int(hard.sum()),
            },
            "reliability": {
                "score": reliability,
                "threshold": self.manifest["thresholds"]["reliability"],
                "status": reliability_status,
            },
            "ood": {
                "components": component,
                "normalized_components": dict(zip(O05_SCORE_ORDER, normalized, strict=True)),
                "score": ood_score,
                "threshold": self.manifest["thresholds"]["ood_review"],
                "status": ood_status,
                "refuse_enabled": False,
            },
            "application": {
                "continue_to_prediction_display": ood_status == "ACCEPT",
                "reliability_status_if_accepted": reliability_status,
            },
        }


def compare_outputs(actual: dict, expected: dict, *, tolerance: float = 1e-12) -> None:
    """Recursively compare reconstruction output, with exact strings and hashes."""
    if actual.keys() != expected.keys():
        raise ValueError("Reconstruction output keys differ")
    for key, left in actual.items():
        right = expected[key]
        if isinstance(left, dict):
            compare_outputs(left, right, tolerance=tolerance)
        elif isinstance(left, list):
            if len(left) != len(right):
                raise ValueError(f"Reconstruction list length differs at {key}")
            for a, b in zip(left, right, strict=True):
                if isinstance(a, float):
                    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
                        raise ValueError(f"Reconstruction float differs at {key}")
                elif a != b:
                    raise ValueError(f"Reconstruction value differs at {key}")
        elif isinstance(left, float):
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"Reconstruction float differs at {key}: {left} vs {right}")
        elif left != right:
            raise ValueError(f"Reconstruction value differs at {key}: {left} vs {right}")
