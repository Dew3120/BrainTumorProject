"""Validated local-image decoding and frozen preprocessing contract."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .constants import CLASSIFICATION_SIZE, SEGMENTATION_SIZE
from .errors import InputValidationError

_FORMAT_BY_SUFFIX = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True, slots=True)
class InputLimits:
    """Parser limits applied before a worker can receive image pixels."""

    max_file_bytes: int = 64 * 1024 * 1024
    min_width: int = 16
    min_height: int = 16
    max_width: int = 16_384
    max_height: int = 16_384
    max_pixels: int = 100_000_000


DEFAULT_INPUT_LIMITS = InputLimits()


@dataclass(frozen=True, slots=True)
class InputDescriptor:
    """Serializable metadata for one validated source image."""

    source_path: str
    filename: str
    image_format: str
    source_mode: str
    width: int
    height: int
    file_bytes: int
    source_sha256: str
    grayscale_sha256: str


@dataclass(frozen=True, slots=True)
class DecodedInput:
    """Validated source descriptor and an immutable uint8 grayscale image."""

    descriptor: InputDescriptor
    gray_uint8: np.ndarray


@dataclass(frozen=True, slots=True)
class PreprocessedInput:
    """Exact arrays consumed by the frozen classifier and segmenters."""

    descriptor: InputDescriptor
    gray_uint8: np.ndarray
    segmenter_chw: np.ndarray
    classifier_chw: np.ndarray


def _fail(code: str, message: str) -> InputValidationError:
    return InputValidationError(code, message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    contiguous.setflags(write=False)
    return contiguous


def validate_and_decode_image(
    path: str | Path,
    *,
    limits: InputLimits = DEFAULT_INPUT_LIMITS,
) -> DecodedInput:
    """Decode one nonanimated PNG/JPEG without applying EXIF geometry changes."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise _fail("INPUT_SYMLINK_REJECTED", "Select the image file directly.")
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise _fail("INPUT_NOT_FOUND", f"The selected image cannot be opened: {exc}") from exc
    if not source.is_file():
        raise _fail("INPUT_NOT_FILE", "The selected path is not a regular file.")
    suffix = source.suffix.lower()
    expected_format = _FORMAT_BY_SUFFIX.get(suffix)
    if expected_format is None:
        raise _fail("INPUT_EXTENSION_UNSUPPORTED", "Only PNG, JPG, and JPEG are supported.")
    file_bytes = source.stat().st_size
    if file_bytes <= 0 or file_bytes > limits.max_file_bytes:
        raise _fail(
            "INPUT_FILE_SIZE_INVALID",
            f"File size must be between 1 and {limits.max_file_bytes} bytes.",
        )
    payload = source.read_bytes()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as probe:
                probe.verify()
            with Image.open(source) as image:
                image_format = image.format
                if image_format != expected_format:
                    raise _fail(
                        "INPUT_FORMAT_MISMATCH",
                        f"Extension {suffix} does not match decoded format {image_format!r}.",
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise _fail("INPUT_ANIMATION_UNSUPPORTED", "Animated images are unsupported.")
                width, height = image.size
                if not (
                    limits.min_width <= width <= limits.max_width
                    and limits.min_height <= height <= limits.max_height
                    and width * height <= limits.max_pixels
                ):
                    raise _fail(
                        "INPUT_DIMENSIONS_INVALID",
                        f"Decoded dimensions {width}x{height} violate the input limits.",
                    )
                orientation = image.getexif().get(274, 1)
                if orientation not in (None, 1):
                    raise _fail(
                        "INPUT_EXIF_ORIENTATION_UNSUPPORTED",
                        "Rotate the image pixels explicitly before analysis; EXIF orientation is not applied.",
                    )
                if image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info:
                    raise _fail(
                        "INPUT_ALPHA_UNSUPPORTED",
                        "Images with transparency are unsupported; export an opaque PNG or JPEG.",
                    )
                source_mode = image.mode
                gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except InputValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _fail("INPUT_DECOMPRESSION_BOMB", "The decoded image is too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise _fail("INPUT_DECODE_FAILED", f"Image decoding failed: {exc}") from exc

    if gray.shape != (height, width):
        raise _fail("INPUT_DECODE_SHAPE_INVALID", repr(gray.shape))
    descriptor = InputDescriptor(
        source_path=str(source),
        filename=source.name,
        image_format=image_format,
        source_mode=source_mode,
        width=width,
        height=height,
        file_bytes=file_bytes,
        source_sha256=_sha256_bytes(payload),
        grayscale_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
    )
    return DecodedInput(descriptor=descriptor, gray_uint8=_readonly(gray))


def _letterbox(gray_uint8: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(gray_uint8)
    if value.ndim != 2 or value.dtype != np.uint8:
        raise _fail("INPUT_PIXELS_INVALID", "Expected one two-dimensional uint8 image.")
    image = Image.fromarray(value, mode="L")
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def preprocess_for_frozen_stack(decoded: DecodedInput) -> PreprocessedInput:
    """Reproduce the Stage 8 segmenter and classifier preprocessing exactly."""

    segmenter = _letterbox(decoded.gray_uint8, SEGMENTATION_SIZE)[None, :, :]
    classifier_gray = _letterbox(decoded.gray_uint8, CLASSIFICATION_SIZE)
    classifier_rgb = np.repeat(classifier_gray[None, :, :], 3, axis=0)
    classifier = ((classifier_rgb - _IMAGENET_MEAN) / _IMAGENET_STD).astype(
        np.float32
    )
    return PreprocessedInput(
        descriptor=decoded.descriptor,
        gray_uint8=decoded.gray_uint8,
        segmenter_chw=_readonly(segmenter.astype(np.float32)),
        classifier_chw=_readonly(classifier),
    )
