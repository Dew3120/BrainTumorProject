"""Inference-only extended reliability features registered for R03."""

import math

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.measure import perimeter
from skimage.morphology import convex_hull_image

SIZE = 128
NEW_FEATURES = (
    "mask_circularity",
    "mask_solidity",
    "boundary_fractal_dimension",
    "predicted_boundary_image_gradient_mean",
    "predicted_centroid_to_image_center_distance",
    "predicted_vertical_symmetry",
)
BOX_WIDTHS = (2, 4, 8, 16, 32)


def model_space_image(image, size=SIZE):
    """Apply the segmenter's image-only grayscale letterbox transform."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Source image has invalid dimensions")
    scale = min(size / width, size / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(
        image.convert("L").resize(resized, Image.Resampling.BILINEAR), offset
    )
    # Match segmentation_smoke_test.transform_pair exactly: conversion to
    # float32 occurs before division. Feature arithmetic may promote afterward.
    output = np.asarray(canvas, dtype=np.float32) / 255.0
    if output.shape != (size, size) or not np.isfinite(output).all():
        raise ValueError("Model-space image transform failed")
    if np.any((output < 0) | (output > 1)):
        raise ValueError("Model-space image lies outside [0, 1]")
    return output


def predicted_boundary(hard_mask):
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(
        hard_mask, structure=structure, iterations=1, border_value=0
    )
    return hard_mask & ~eroded


def boundary_fractal_dimension(boundary):
    counts = []
    inverse_scales = []
    for width in BOX_WIDTHS:
        occupied = boundary.reshape(
            SIZE // width, width, SIZE // width, width
        ).any(axis=(1, 3))
        count = int(occupied.sum())
        if count > 0:
            counts.append(count)
            inverse_scales.append(SIZE / width)
    if len(counts) < 2:
        return 0.0
    slope = np.polyfit(
        np.log(np.asarray(inverse_scales, dtype=np.float64)),
        np.log(np.asarray(counts, dtype=np.float64)),
        deg=1,
    )[0]
    return float(slope)


def extended_prediction_features(probability, model_image, threshold=0.5):
    """Extract R03 features without accepting a reference mask or metadata."""
    probability = np.asarray(probability, dtype=np.float64)
    model_image = np.asarray(model_image, dtype=np.float64)
    if probability.shape != (SIZE, SIZE) or model_image.shape != (SIZE, SIZE):
        raise ValueError("Expected 128x128 probability and model-space image")
    if not np.isfinite(probability).all() or not np.isfinite(model_image).all():
        raise ValueError("R03 feature input contains non-finite values")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probability input lies outside [0, 1]")
    if np.any((model_image < 0) | (model_image > 1)):
        raise ValueError("Model-space image lies outside [0, 1]")

    hard = probability >= threshold
    area = int(hard.sum())
    if area == 0:
        return {name: 0.0 for name in NEW_FEATURES}

    boundary = predicted_boundary(hard)
    measured_perimeter = float(perimeter(hard, neighborhood=8))
    circularity = (
        4.0 * math.pi * area / measured_perimeter**2
        if measured_perimeter > 0
        else 0.0
    )
    hull = convex_hull_image(
        hard,
        offset_coordinates=True,
        tolerance=1e-10,
        include_borders=True,
    )
    hull_area = int(hull.sum())
    solidity = area / hull_area if hull_area else 0.0

    gradient_y, gradient_x = np.gradient(model_image)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    boundary_gradient = (
        float(gradient_magnitude[boundary].mean()) if boundary.any() else 0.0
    )

    y, x = np.where(hard)
    centre = (SIZE - 1) / 2.0
    centroid_distance = math.hypot(float(y.mean()) - centre, float(x.mean()) - centre)
    normalized_centroid_distance = centroid_distance / (math.sqrt(2.0) * centre)
    symmetry = float(np.logical_and(hard, np.fliplr(hard)).sum() / area)

    result = {
        "mask_circularity": float(circularity),
        "mask_solidity": float(solidity),
        "boundary_fractal_dimension": boundary_fractal_dimension(boundary),
        "predicted_boundary_image_gradient_mean": boundary_gradient,
        "predicted_centroid_to_image_center_distance": float(
            normalized_centroid_distance
        ),
        "predicted_vertical_symmetry": symmetry,
    }
    if tuple(result) != NEW_FEATURES or not np.isfinite(list(result.values())).all():
        raise ValueError("R03 feature extraction produced an invalid result")
    return result
