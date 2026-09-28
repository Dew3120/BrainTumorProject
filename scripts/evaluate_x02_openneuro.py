"""Evaluate the frozen E08+C03+R02 stack on audited OpenNeuro pituitary MRI."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import advanced_calibration_common as calibration_common
import evaluate_x01_cheng as x01
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_failure_analysis as failure_analysis
import stage6_external_common as external
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x02_openneuro_pituitary"
ARTIFACTS = EXPERIMENT / "artifacts"
PAYLOAD = ROOT / "data/external/openneuro_ds006248_v1.0.0_payload"
AUDIT_RESULT = ARTIFACTS / "dataset_audit.json"
SUBJECT_MANIFEST = ARTIFACTS / "subject_manifest.csv"
TUMOR_SLICE_MANIFEST = ARTIFACTS / "tumor_slice_manifest.csv"
CLASS_NAMES = x01.CLASS_NAMES
LABEL_INDEX = x01.LABEL_INDEX
PITUITARY_INDEX = LABEL_INDEX["pituitary"]
BOOTSTRAPS = 2000
SEED = 42
EPSILON = 1e-12
EXPECTED_AUDIT_SHA256 = "999ff3a09f2918024924c9a16151b9ef5b60a6527752bdec36a0fc0b3b3737d4"
EXPECTED_SUBJECT_MANIFEST_SHA256 = (
    "b60c64803b6e4171b63f0b1a9014ad0eab11713a4c2ea53c8862d8ebefeb6642"
)
EXPECTED_TUMOR_SLICE_MANIFEST_SHA256 = (
    "bd85dfce6a62884c081ed73af8f6c4b5f7d963a8e273c6ebf4d99fbb068cabd8"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_evaluation_rows(subject_rows: list[dict], tumor_rows: list[dict]) -> list[dict]:
    tumor_membership = {
        (row["subject_id"], int(row["coronal_index_ras"])) for row in tumor_rows
    }
    rows = []
    for subject in sorted(subject_rows, key=lambda row: row["subject_id"]):
        shape = tuple(int(value) for value in subject["canonical_shape"].split("x"))
        if len(shape) != 3 or shape[1] <= 0:
            raise ValueError(f"Invalid canonical shape: {subject['subject_id']}")
        for coronal_index in range(shape[1]):
            rows.append(
                {
                    "case_id": f"{subject['subject_id']}:coronal:{coronal_index:03d}",
                    "patient_id": subject["subject_id"],
                    "subject_id": subject["subject_id"],
                    "label": "pituitary",
                    "coronal_index_ras": coronal_index,
                    "reference_has_tumor": (
                        subject["subject_id"], coronal_index
                    )
                    in tumor_membership,
                    "image_path": subject["image_path"],
                    "mask_path": subject["mask_path"],
                    "width": shape[0],
                    "height": shape[2],
                    "voxel_x_mm": float(subject["voxel_size_x_mm"]),
                    "voxel_y_mm": float(subject["voxel_size_y_mm"]),
                    "voxel_z_mm": float(subject["voxel_size_z_mm"]),
                }
            )
    if len(rows) != sum(int(row["canonical_shape"].split("x")[1]) for row in subject_rows):
        raise ValueError("Evaluation row construction is incomplete")
    if sum(bool(row["reference_has_tumor"]) for row in rows) != len(tumor_rows):
        raise ValueError("Primary tumor-bearing membership changed")
    return rows


class OpenNeuroDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self._cached_subject = None
        self._cached_image = None
        self._cached_mask = None

    def __len__(self):
        return len(self.rows)

    def _volume(self, row):
        if self._cached_subject != row["subject_id"]:
            image, image_affine, _ = external.load_canonical_nifti(
                PAYLOAD / row["image_path"]
            )
            mask, mask_affine, _ = external.load_canonical_nifti(
                PAYLOAD / row["mask_path"]
            )
            if image.shape != mask.shape or np.max(np.abs(image_affine - mask_affine)) > 1e-4:
                raise ValueError(f"Geometry changed for {row['subject_id']}")
            self._cached_subject = row["subject_id"]
            self._cached_image = image
            self._cached_mask = np.rint(mask).astype(np.uint8)
        return self._cached_image, self._cached_mask

    def __getitem__(self, position: int):
        row = self.rows[position]
        image, mask = self._volume(row)
        index = int(row["coronal_index_ras"])
        image_slice = image[:, index, :].T
        target_slice = (mask[:, index, :] == 1).T
        segmenter, geometry, _window = external.preprocess_segmenter(image_slice)
        classifier, _class_geometry, _class_window = external.preprocess_classifier(
            image_slice
        )
        target, mask_geometry = external.preprocess_reference_mask(
            target_slice, allow_empty=True
        )
        if geometry != mask_geometry:
            raise ValueError(f"Image/mask transform mismatch: {row['case_id']}")
        if bool(target.any()) != bool(row["reference_has_tumor"]):
            raise ValueError(f"Reference membership mismatch: {row['case_id']}")
        return (
            torch.from_numpy(segmenter),
            torch.from_numpy(classifier),
            torch.from_numpy(target.astype(np.float32)),
            position,
        )


def run_inference(rows, classifier, models, device):
    count = len(rows)
    raw_probability = np.empty((count, 128, 128), dtype=np.float32)
    calibrated = np.empty_like(raw_probability)
    targets = np.empty((count, 128, 128), dtype=bool)
    class_probability = np.empty((count, 4), dtype=np.float64)
    embeddings = np.empty((count, 512), dtype=np.float32)
    reliability = np.empty(count, dtype=np.float64)
    intensity = np.empty((count, 3), dtype=np.float64)
    route = np.empty(count, dtype=object)
    loader = DataLoader(
        OpenNeuroDataset(rows),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    captured = []

    def hook(_module, _input, output):
        captured.append(output.detach().float().flatten(1).cpu().numpy())

    hook_handle = classifier.avgpool.register_forward_hook(hook)
    with torch.inference_mode():
        for segmenter_images, classifier_images, references, indices in loader:
            segmenter_images = segmenter_images.to(device, non_blocking=True)
            classifier_images = classifier_images.to(device, non_blocking=True)
            references = references.numpy()[:, 0] > 0.5
            index_values = indices.numpy()
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                class_logits = classifier(classifier_images)
                e05 = models["e05"](segmenter_images)
                e06 = models["e06"](segmenter_images)
                specialist = models["e07_specialist"](segmenter_images)
                fallback = models["v1_fallback"](segmenter_images)
            probabilities = class_logits.float().softmax(dim=1).cpu().numpy()
            predicted = probabilities.argmax(axis=1)
            glioma_route = predicted == LABEL_INDEX["glioma"]
            routed = fallback.float().sigmoid()
            if np.any(glioma_route):
                route_index = torch.from_numpy(glioma_route).to(device)
                routed[route_index] = specialist.float().sigmoid()[route_index]
            ensemble = (e05.float().sigmoid() + e06.float().sigmoid() + routed) / 3.0
            batch_raw = ensemble[:, 0].cpu().numpy().astype(np.float32)
            batch_calibrated = external.apply_operational_c03(batch_raw).astype(np.float32)
            raw_probability[index_values] = batch_raw
            calibrated[index_values] = batch_calibrated
            targets[index_values] = references
            class_probability[index_values] = probabilities
            embeddings[index_values] = captured.pop(0)
            for batch_position, output_index in enumerate(index_values):
                route[output_index] = (
                    "glioma_specialist" if glioma_route[batch_position] else "v1_fallback"
                )
                reliability[output_index] = external.operational_reliability(
                    batch_calibrated[batch_position]
                )["score"]
                values = segmenter_images[batch_position, 0].float().cpu().numpy()
                nonzero = values[values > 0]
                intensity[output_index] = [
                    float(values.mean()),
                    float(values.std()),
                    float(nonzero.mean()) if len(nonzero) else 0.0,
                ]
    hook_handle.remove()
    numeric = (raw_probability, calibrated, class_probability, embeddings, reliability, intensity)
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError("Inference produced a non-finite output")
    return raw_probability, calibrated, targets, class_probability, embeddings, reliability, intensity, route


def subject_classifier_evaluation(rows, probabilities):
    keys, members = x01.subject_index(rows)
    subject_probability = np.asarray(
        [probabilities[members[key]].mean(axis=0) for key in keys], dtype=np.float64
    )

    def metrics(values):
        predicted = values.argmax(axis=1)
        correct = predicted == PITUITARY_INDEX
        onehot = np.eye(4, dtype=np.float64)[np.full(len(values), PITUITARY_INDEX)]
        calibration = x01.equal_frequency_calibration(values.max(axis=1), correct)
        return {
            "accuracy": float(correct.mean()),
            "pituitary_recall": float(correct.mean()),
            "nll": float(-np.log(np.clip(values[:, PITUITARY_INDEX], EPSILON, 1)).mean()),
            "brier": float(np.square(values - onehot).sum(axis=1).mean()),
            "top_label_ece": calibration["ece"],
            "top_label_mce": calibration["mce"],
            "reliability_bins": calibration["bins"],
            "confusion": {
                name: int(np.sum(predicted == index))
                for index, name in enumerate(CLASS_NAMES)
            },
            "one_vs_rest_auroc": "NOT_ESTIMABLE: single true category",
            "no_tumor_specificity": "NOT_ESTIMABLE: no no-tumor subjects",
            "binary_tumor_auroc": "NOT_ESTIMABLE: no no-tumor subjects",
        }

    point = metrics(subject_probability)
    generator = np.random.default_rng(SEED)
    bootstrap = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        selected = subject_probability[generator.integers(0, len(keys), size=len(keys))]
        value = metrics(selected)
        for name in (
            "accuracy",
            "pituitary_recall",
            "nll",
            "brier",
            "top_label_ece",
            "top_label_mce",
        ):
            bootstrap[name].append(value[name])
    point["bootstrap_ci95"] = {
        name: x01.percentile_ci(values) for name, values in bootstrap.items()
    }
    point["subjects"] = len(keys)
    point["aggregation"] = "Mean slice probability per subject over tumor-bearing slices"
    return point


def boundary_metrics_physical(prediction, reference, mm_y: float, mm_x: float):
    structure = np.ones((3, 3), dtype=bool)
    predicted_boundary = prediction ^ binary_erosion(prediction, structure=structure)
    reference_boundary = reference ^ binary_erosion(reference, structure=structure)
    if not predicted_boundary.any():
        return 0.0, None
    prediction_close = binary_dilation(predicted_boundary, structure=structure, iterations=2)
    reference_close = binary_dilation(reference_boundary, structure=structure, iterations=2)
    precision = float((predicted_boundary & reference_close).sum() / predicted_boundary.sum())
    recall = float((reference_boundary & prediction_close).sum() / reference_boundary.sum())
    f1 = 2 * precision * recall / max(precision + recall, EPSILON)
    pred_points = np.column_stack(np.where(predicted_boundary)).astype(np.float64)
    ref_points = np.column_stack(np.where(reference_boundary)).astype(np.float64)
    scale = np.asarray([mm_y, mm_x], dtype=np.float64)
    pred_points *= scale
    ref_points *= scale
    pred_tree, ref_tree = cKDTree(pred_points), cKDTree(ref_points)
    hausdorff = max(
        float(ref_tree.query(pred_points, k=1)[0].max()),
        float(pred_tree.query(ref_points, k=1)[0].max()),
    )
    return f1, hausdorff


def whole_volume_scores(rows, hard):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["subject_id"]].append((index, row))
    output = []
    slice_counts = []
    for subject_id in sorted(grouped):
        members = sorted(grouped[subject_id], key=lambda item: int(item[1]["coronal_index_ras"]))
        mask, _, _ = external.load_canonical_nifti(PAYLOAD / members[0][1]["mask_path"])
        mask = np.rint(mask).astype(np.uint8) == 1
        intersection = predicted = reference = 0
        for index, row in members:
            source_reference = mask[:, int(row["coronal_index_ras"]), :].T
            geometry = external.letterbox_geometry(int(row["width"]), int(row["height"]), 128)
            source_prediction = external.restore_to_source(hard[index], geometry, mask=True)
            current_intersection = int(np.count_nonzero(source_prediction & source_reference))
            current_prediction = int(np.count_nonzero(source_prediction))
            current_reference = int(np.count_nonzero(source_reference))
            intersection += current_intersection
            predicted += current_prediction
            reference += current_reference
            slice_counts.append(
                {
                    "case_id": row["case_id"],
                    "subject_id": subject_id,
                    "coronal_index_ras": row["coronal_index_ras"],
                    "reference_has_tumor": bool(current_reference),
                    "intersection_pixels": current_intersection,
                    "prediction_pixels": current_prediction,
                    "reference_pixels": current_reference,
                }
            )
        output.append(
            {
                "subject_id": subject_id,
                "intersection_pixels": intersection,
                "prediction_pixels": predicted,
                "reference_pixels": reference,
                "dice": 2 * intersection / max(predicted + reference, 1),
                "iou": intersection / max(predicted + reference - intersection, 1),
            }
        )
    dice = np.asarray([row["dice"] for row in output], dtype=np.float64)
    iou = np.asarray([row["iou"] for row in output], dtype=np.float64)
    generator = np.random.default_rng(SEED)
    dice_boot, iou_boot = [], []
    for _ in range(BOOTSTRAPS):
        indices = generator.integers(0, len(output), size=len(output))
        dice_boot.append(float(dice[indices].mean()))
        iou_boot.append(float(iou[indices].mean()))
    total_i = sum(row["intersection_pixels"] for row in output)
    total_p = sum(row["prediction_pixels"] for row in output)
    total_r = sum(row["reference_pixels"] for row in output)
    summary = {
        "subject_macro_dice": float(dice.mean()),
        "subject_macro_dice_ci95": x01.percentile_ci(dice_boot),
        "subject_macro_iou": float(iou.mean()),
        "subject_macro_iou_ci95": x01.percentile_ci(iou_boot),
        "cohort_global_dice": 2 * total_i / max(total_p + total_r, 1),
        "cohort_global_iou": total_i / max(total_p + total_r - total_i, 1),
        "subjects": len(output),
        "coronal_slices_including_empty_reference": len(rows),
    }
    return summary, output, slice_counts


def summarize_failure_modes(rows):
    if not rows:
        return {"all": {"count": 0}, "pituitary": {"count": 0}, "glioma": "NOT_ESTIMABLE", "meningioma": "NOT_ESTIMABLE"}
    flags = (
        "empty_prediction",
        "nonempty_zero_overlap",
        "severe_under_prediction",
        "severe_over_prediction",
        "fragmented_prediction",
        "disconnected_false_positive",
        "reference_touches_model_edge",
        "prediction_touches_model_edge",
    )
    summary = {}
    for name, members in (("all", rows), ("pituitary", rows)):
        values = {"count": len(members)}
        for flag in flags:
            count = sum(bool(row[flag]) for row in members)
            values[flag] = {"count": count, "fraction": count / len(members)}
        values["complete_miss"] = {
            "count": values["empty_prediction"]["count"] + values["nonempty_zero_overlap"]["count"],
            "fraction": (
                values["empty_prediction"]["count"] + values["nonempty_zero_overlap"]["count"]
            )
            / len(members),
        }
        primary = Counter(row["primary_failure_mode"] for row in members)
        values["primary_failure_modes"] = {
            key: {"count": count, "fraction": count / len(members)}
            for key, count in sorted(primary.items())
        }
        summary[name] = values
    summary["glioma"] = "NOT_ESTIMABLE"
    summary["meningioma"] = "NOT_ESTIMABLE"
    return summary


def mask_features(targets):
    foreground, centroid_y, centroid_x, components = [], [], [], []
    for target in targets:
        foreground.append(float(target.mean()))
        points = np.column_stack(np.where(target))
        centroid_y.append(float(points[:, 0].mean() / 127))
        centroid_x.append(float(points[:, 1].mean() / 127))
        components.append(float(ndimage.label(target, structure=np.ones((3, 3)))[1]))
    return {
        "foreground_fraction": np.asarray(foreground),
        "centroid_y": np.asarray(centroid_y),
        "centroid_x": np.asarray(centroid_x),
        "component_count": np.asarray(components),
    }


def boundary_gradient(images, targets):
    values = []
    for image, target in zip(images, targets, strict=True):
        gradient_y, gradient_x = np.gradient(image.astype(np.float64))
        magnitude = np.hypot(gradient_y, gradient_x)
        boundary = target ^ binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
        values.append(float(magnitude[boundary].mean()))
    return np.asarray(values)


def js_divergence(left, right, bins):
    left_hist = np.histogram(left, bins=bins)[0].astype(np.float64) + EPSILON
    right_hist = np.histogram(right, bins=bins)[0].astype(np.float64) + EPSILON
    left_hist /= left_hist.sum()
    right_hist /= right_hist.sum()
    return float(jensenshannon(left_hist, right_hist, base=2.0) ** 2)


def distribution_gap(left, right, bins):
    return {
        "wasserstein": float(wasserstein_distance(left, right)),
        "jensen_shannon_divergence": js_divergence(left, right, bins),
        "external_mean": float(np.mean(left)),
        "brisc_validation_mean": float(np.mean(right)),
    }


def brisc_segmenter_images(order):
    images = []
    for row in order:
        with Image.open(ROOT / row["path"]) as source:
            tensor, _geometry, _window = external.preprocess_segmenter(np.asarray(source.convert("L")))
        images.append(tensor[0])
    return np.asarray(images, dtype=np.float32)


def domain_gap(rows, images, intensity, embeddings, targets, reliability, classifier, device, audit):
    brisc = x01.brisc_domain_reference(classifier, device)
    brisc_images = brisc_segmenter_images(brisc["order"])
    external_features = mask_features(targets)
    brisc_features = mask_features(brisc["targets"])
    external_gradient = boundary_gradient(images, targets)
    brisc_gradient = boundary_gradient(brisc_images, brisc["targets"])
    embedding_wasserstein = [
        wasserstein_distance(embeddings[:, index], brisc["embeddings"][:, index])
        for index in range(embeddings.shape[1])
    ]
    return {
        "reference": "BRISC development validation only; sealed test pixels unopened",
        "cohort": {
            "modality": "3T coronal contrast-enhanced T1 MRI",
            "orientation": "coronal; canonical RAS axis 1",
            "subjects": len({row["subject_id"] for row in rows}),
            "tumor_bearing_slices": len(rows),
            "canonical_shape_counts": audit["dataset"]["shape_counts"],
            "voxel_spacing_mm": {
                "x_range": [min(row["voxel_x_mm"] for row in rows), max(row["voxel_x_mm"] for row in rows)],
                "y_range": [min(row["voxel_y_mm"] for row in rows), max(row["voxel_y_mm"] for row in rows)],
                "z_range": [min(row["voxel_z_mm"] for row in rows), max(row["voxel_z_mm"] for row in rows)],
            },
        },
        "continuous_distribution_shift": {
            "model_input_mean": distribution_gap(intensity[:, 0], brisc["intensity"][:, 0], np.linspace(0, 1, 31)),
            "model_input_sd": distribution_gap(intensity[:, 1], brisc["intensity"][:, 1], np.linspace(0, 0.5, 31)),
            "nonzero_mean": distribution_gap(intensity[:, 2], brisc["intensity"][:, 2], np.linspace(0, 1, 31)),
            "foreground_fraction": distribution_gap(external_features["foreground_fraction"], brisc_features["foreground_fraction"], np.linspace(0, 0.5, 31)),
            "centroid_y": distribution_gap(external_features["centroid_y"], brisc_features["centroid_y"], np.linspace(0, 1, 31)),
            "centroid_x": distribution_gap(external_features["centroid_x"], brisc_features["centroid_x"], np.linspace(0, 1, 31)),
            "component_count": distribution_gap(external_features["component_count"], brisc_features["component_count"], np.arange(0.5, 10.6, 1)),
            "reference_boundary_gradient": distribution_gap(external_gradient, brisc_gradient, np.linspace(0, 1.5, 31)),
        },
        "fragmented_reference_fraction": {
            "openneuro": float(np.mean(external_features["component_count"] > 1)),
            "brisc_validation": float(np.mean(brisc_features["component_count"] > 1)),
        },
        "encoder_representation_shift": {
            "centroid_euclidean": float(np.linalg.norm(embeddings.mean(axis=0) - brisc["embeddings"].mean(axis=0))),
            "dimension_mean_wasserstein": float(np.mean(embedding_wasserstein)),
            "dimension_median_wasserstein": float(np.median(embedding_wasserstein)),
            "encoder": "frozen classifier ResNet18 avgpool (512 dimensions)",
        },
        "reliability_score_shift": distribution_gap(reliability, brisc["reliability"], np.linspace(0, 1, 31)),
        "policy_review_rate": {
            "openneuro": float(np.mean(reliability < external.RELIABILITY_POLICY_THRESHOLD)),
            "brisc_operational_reference": float(np.mean(brisc["reliability"] < external.RELIABILITY_POLICY_THRESHOLD)),
        },
    }


def render_visual_panel(path: Path, rows, calibrated, targets, dice):
    ordered = sorted(range(len(rows)), key=lambda index: (dice[index], rows[index]["case_id"]))
    selected = ordered[:6] + ordered[-6:]
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    dataset = OpenNeuroDataset(rows)
    for output_index, index in enumerate(selected):
        model_input = dataset[index][0].numpy()[0]
        reference = targets[index]
        prediction = calibrated[index] >= 0.5
        axes[output_index, 0].imshow(model_input, cmap="gray")
        axes[output_index, 0].set_title(f"{rows[index]['case_id']} model input")
        axes[output_index, 1].imshow(model_input, cmap="gray")
        axes[output_index, 1].imshow(np.ma.masked_where(~reference, reference), cmap="Reds", alpha=0.55)
        axes[output_index, 1].set_title("Reference (red)")
        axes[output_index, 2].imshow(model_input, cmap="gray")
        axes[output_index, 2].imshow(np.ma.masked_where(~prediction, prediction), cmap="Blues", alpha=0.55)
        axes[output_index, 2].set_title(f"Prediction (cyan), Dice {dice[index]:.3f}")
        for axis in axes[output_index]:
            axis.axis("off")
    figure.suptitle("X02 deterministic six worst / six best pituitary slices", y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    config = read_json(EXPERIMENT / "config.json")
    audit = read_json(AUDIT_RESULT)
    if config["status"] != "AUDIT_COMPLETE" or audit["audit_outcome"] != "ELIGIBLE_INDEPENDENT":
        raise ValueError("X02 does not have a passed independent audit-only state")
    if x01.sha256(AUDIT_RESULT) != EXPECTED_AUDIT_SHA256:
        raise ValueError("X02 audit result hash mismatch")
    if x01.sha256(SUBJECT_MANIFEST) != EXPECTED_SUBJECT_MANIFEST_SHA256:
        raise ValueError("X02 subject manifest hash mismatch")
    if x01.sha256(TUMOR_SLICE_MANIFEST) != EXPECTED_TUMOR_SLICE_MANIFEST_SHA256:
        raise ValueError("X02 tumor-slice manifest hash mismatch")
    if config["segmentation_threshold"] != 0.5:
        raise ValueError("Segmentation threshold changed")
    if config["reliability_policy_threshold"] != "0.9490250173235835":
        raise ValueError("Reliability threshold representation changed")

    all_rows = build_evaluation_rows(read_csv(SUBJECT_MANIFEST), read_csv(TUMOR_SLICE_MANIFEST))
    primary_all_indices = np.asarray(
        [index for index, row in enumerate(all_rows) if row["reference_has_tumor"]], dtype=np.int64
    )
    frozen = external.verify_frozen_artifacts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, models = x01.load_models(device)
    (
        raw_all,
        calibrated_all,
        targets_all,
        class_all,
        embeddings_all,
        reliability_all,
        intensity_all,
        route_all,
    ) = run_inference(all_rows, classifier, models, device)

    primary_rows = [all_rows[int(index)] for index in primary_all_indices]
    raw = raw_all[primary_all_indices]
    calibrated = calibrated_all[primary_all_indices]
    targets = targets_all[primary_all_indices]
    class_probability = class_all[primary_all_indices]
    embeddings = embeddings_all[primary_all_indices]
    reliability = reliability_all[primary_all_indices]
    intensity = intensity_all[primary_all_indices]
    route = route_all[primary_all_indices]
    hard = calibrated >= 0.5
    raw_hard = raw >= 0.5
    dice, iou = x01.mask_scores(hard, targets)
    raw_dice, raw_iou = x01.mask_scores(raw_hard, targets)
    predicted_class = class_probability.argmax(axis=1)
    classifier_subject = subject_classifier_evaluation(primary_rows, class_probability)

    boundary_f1 = np.empty(len(primary_rows), dtype=np.float64)
    hausdorff_mm = np.full(len(primary_rows), np.nan, dtype=np.float64)
    per_case, failure_rows = [], []
    primary_images = []
    primary_dataset = OpenNeuroDataset(primary_rows)
    for index, row in enumerate(primary_rows):
        image_tensor = primary_dataset[index][0].numpy()[0]
        primary_images.append(image_tensor)
        geometry = external.letterbox_geometry(int(row["width"]), int(row["height"]), 128)
        mm_x = float(row["voxel_x_mm"]) * int(row["width"]) / geometry.resized_width
        mm_y = float(row["voxel_z_mm"]) * int(row["height"]) / geometry.resized_height
        boundary_f1[index], hausdorff = boundary_metrics_physical(hard[index], targets[index], mm_y, mm_x)
        if hausdorff is not None:
            hausdorff_mm[index] = hausdorff
        record = {
            "case_id": row["case_id"],
            "subject_id": row["subject_id"],
            "category": "pituitary",
            "coronal_index_ras": row["coronal_index_ras"],
            "classifier_predicted": CLASS_NAMES[predicted_class[index]],
            **{f"classifier_probability_{name}": float(class_probability[index, position]) for position, name in enumerate(CLASS_NAMES)},
            "e07_route": route[index],
            "raw_dice": float(raw_dice[index]),
            "raw_iou": float(raw_iou[index]),
            "calibrated_dice": float(dice[index]),
            "calibrated_iou": float(iou[index]),
            "reference_pixels": int(targets[index].sum()),
            "predicted_pixels": int(hard[index].sum()),
            "boundary_f1_tolerance_2": float(boundary_f1[index]),
            "hausdorff_mm": None if np.isnan(hausdorff_mm[index]) else float(hausdorff_mm[index]),
            "r02_reliability": float(reliability[index]),
            "policy_status": "STANDARD" if reliability[index] >= external.RELIABILITY_POLICY_THRESHOLD else "REVIEW_REQUIRED",
            "failure_dice_below_0_50": bool(dice[index] < 0.5),
        }
        per_case.append(record)
        if dice[index] < 0.5:
            failure_rows.append({**record, **failure_analysis.component_statistics(hard[index], targets[index])})
    write_csv(ARTIFACTS / "per_case_predictions.csv", per_case)
    if failure_rows:
        write_csv(ARTIFACTS / "failure_taxonomy.csv", failure_rows)

    whole_summary, whole_subject_rows, whole_slice_rows = whole_volume_scores(all_rows, calibrated_all >= 0.5)
    write_csv(ARTIFACTS / "whole_volume_subject_metrics.csv", whole_subject_rows)
    write_csv(ARTIFACTS / "whole_volume_slice_counts.csv", whole_slice_rows)
    segmentation = {
        "primary_view": "tumor-bearing 2D coronal slices",
        "slice_macro": {
            "dice": float(dice.mean()),
            "iou": float(iou.mean()),
            "raw_dice": float(raw_dice.mean()),
            "raw_iou": float(raw_iou.mean()),
            "boundary_f1_tolerance_2": float(boundary_f1.mean()),
            "hausdorff_mm_nonempty_predictions": float(np.nanmean(hausdorff_mm)),
        },
        "subject_macro": {
            "dice": x01.mean_with_patient_bootstrap(dice, primary_rows),
            "iou": x01.mean_with_patient_bootstrap(iou, primary_rows),
            "boundary_f1_tolerance_2": x01.mean_with_patient_bootstrap(boundary_f1, primary_rows),
        },
        "per_category": {
            "glioma": "NOT_ESTIMABLE",
            "meningioma": "NOT_ESTIMABLE",
            "pituitary": {
                "dice": x01.mean_with_patient_bootstrap(dice, primary_rows),
                "iou": x01.mean_with_patient_bootstrap(iou, primary_rows),
                "slice_mean_dice": float(dice.mean()),
                "slice_mean_iou": float(iou.mean()),
                "boundary_f1_tolerance_2": x01.mean_with_patient_bootstrap(boundary_f1, primary_rows),
                "failures_below_0_50": int(np.sum(dice < 0.5)),
                "zero_dice": int(np.sum(dice == 0)),
            },
        },
        "zero_dice": int(np.sum(dice == 0)),
        "empty_prediction": int(np.sum(hard.sum(axis=(1, 2)) == 0)),
        "dice_below_0_50": int(np.sum(dice < 0.5)),
        "secondary_whole_volume": whole_summary,
    }

    regions = {
        "lesion_centered": np.stack([calibration_common.lesion_regions(targets[index : index + 1])[0] for index in range(len(primary_rows))])
    }
    regions["background"] = ~regions["lesion_centered"]
    regions["whole_image"] = np.ones_like(targets, dtype=bool)
    pixel_calibration = {}
    for region_name, region_masks in regions.items():
        raw_items = [x01.pixel_region_contributions(raw[index], targets[index], region_masks[index]) for index in range(len(primary_rows))]
        calibrated_items = [x01.pixel_region_contributions(calibrated[index], targets[index], region_masks[index]) for index in range(len(primary_rows))]
        pixel_calibration[region_name] = {
            "raw": x01.summarize_pixel_contributions(raw_items, primary_rows),
            "c03_operational": x01.summarize_pixel_contributions(calibrated_items, primary_rows),
        }

    success = dice >= 0.5
    if success.any() and (~success).any():
        reliability_result = x01.reliability_metrics(reliability, success, primary_rows)
        order = np.argsort(reliability)
        top_count = math.ceil(0.20 * len(primary_rows))
        top = order[:top_count]
        zero = dice == 0
        review = reliability < external.RELIABILITY_POLICY_THRESHOLD
        reliability_result["top20_risk"] = {
            "cases": top_count,
            "failures_captured": int((~success[top]).sum()),
            "failures_total": int((~success).sum()),
            "zero_dice_captured": int(zero[top].sum()),
            "zero_dice_total": int(zero.sum()),
        }
        reliability_result["policy"] = {
            "threshold": "0.9490250173235835",
            "review_required": int(review.sum()),
            "standard": int((~review).sum()),
            "review_rate": float(review.mean()),
            "failures_caught": int((review & ~success).sum()),
            "failures_missed": int((~review & ~success).sum()),
            "acceptable_reviewed": int((review & success).sum()),
            "p01_total_cost": int((review & ~success).sum() + (review & success).sum() + 10 * (~review & ~success).sum()),
        }
    else:
        reliability_result = "NOT_ESTIMABLE: outcome has one class"

    taxonomy = summarize_failure_modes(failure_rows)
    gap = domain_gap(
        primary_rows,
        np.asarray(primary_images),
        intensity,
        embeddings,
        targets,
        reliability,
        classifier,
        device,
        audit,
    )
    render_visual_panel(ARTIFACTS / "visual_audit_panel.png", primary_rows, calibrated, targets, dice)
    x01.plot_reliability(
        ARTIFACTS / "classifier_reliability.png",
        classifier_subject["reliability_bins"],
        "X02 patient-level classifier reliability",
    )
    if isinstance(reliability_result, dict):
        x01.plot_reliability(
            ARTIFACTS / "case_reliability.png",
            reliability_result["bins"],
            "X02 case reliability transfer",
        )

    gates = {
        "classifier_pituitary_recall_at_least_0_85": classifier_subject["pituitary_recall"] >= 0.85,
        "classifier_top_label_ece_at_most_0_05": classifier_subject["top_label_ece"] <= 0.05,
        "pituitary_subject_macro_dice_at_least_0_75": segmentation["subject_macro"]["dice"]["subject_macro_mean"] >= 0.75,
    }
    if isinstance(reliability_result, dict):
        gates.update(
            {
                "reliability_auroc_at_least_0_75": reliability_result["failure_detection_auroc"] >= 0.75,
                "reliability_auroc_ci_lower_at_least_0_70": reliability_result["bootstrap_ci95"]["failure_detection_auroc"][0] >= 0.70,
                "reliability_ece_at_most_0_05": reliability_result["ece"] <= 0.05,
                "reliability_brier_skill_positive": reliability_result["brier_skill"] > 0,
                "reliability_brier_skill_ci_lower_positive": reliability_result["bootstrap_ci95"]["brier_skill"][0] > 0,
                "top20_failure_capture_at_least_half": reliability_result["top20_risk"]["failures_captured"] / max(reliability_result["top20_risk"]["failures_total"], 1) >= 0.50,
            }
        )
    else:
        gates["reliability_gate"] = "NOT_ESTIMABLE"
    estimable_gate_values = [value for value in gates.values() if isinstance(value, bool)]

    result = {
        "schema_version": 1,
        "experiment_id": "x02_openneuro_pituitary",
        "status": "EVALUATION_COMPLETE_VISUAL_REVIEW_PENDING",
        "audit_outcome": "ELIGIBLE_INDEPENDENT",
        "independent_external_evidence": True,
        "scope": "Independent OpenNeuro pituitary transport; retrospective research evaluation, not clinical validation",
        "population": {
            "subjects": len({row["subject_id"] for row in primary_rows}),
            "primary_tumor_bearing_slices": len(primary_rows),
            "secondary_all_coronal_slices": len(all_rows),
            "true_category": "pituitary",
        },
        "classifier": classifier_subject,
        "segmentation": segmentation,
        "pixel_calibration": pixel_calibration,
        "reliability_transfer": reliability_result,
        "failure_taxonomy": taxonomy,
        "domain_gap": gap,
        "gates": {
            "status": "PASS" if all(estimable_gate_values) else "FAIL",
            "checks": gates,
            "unidentifiable_metrics": [
                "classifier OVR AUROC",
                "no-tumor specificity",
                "binary tumor/no-tumor AUROC",
                "glioma metrics",
                "meningioma metrics",
            ],
        },
        "verification": {
            "frozen_hashes": {**frozen, **x01.EXPECTED_HASHES},
            "audit_result_sha256": x01.sha256(AUDIT_RESULT),
            "subject_manifest_sha256": x01.sha256(SUBJECT_MANIFEST),
            "tumor_slice_manifest_sha256": x01.sha256(TUMOR_SLICE_MANIFEST),
            "device": str(device),
            "bootstrap_repetitions": BOOTSTRAPS,
            "bootstrap_seed": SEED,
            "bootstrap_unit": "subject_id",
            "segmentation_threshold": 0.5,
            "reliability_threshold": "0.9490250173235835",
            "brisc_files_opened": {"development_validation": 680, "sealed_test": 0},
            "external_subjects_evaluated": 50,
            "external_slices_inferred": len(all_rows),
            "neural_retraining": False,
            "calibrator_refit": False,
            "reliability_refit": False,
            "threshold_changed": False,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "limitations": [
            "The cohort contains only pituitary adenoma subjects, so no-tumor specificity and binary tumor/no-tumor AUROC are not estimable.",
            "The cohort cannot estimate glioma or meningioma performance.",
            "Primary 2D analysis selects tumor-bearing slices using the reference mask; whole-volume analysis including empty-reference slices is reported separately.",
            "This single-site retrospective cohort does not establish prospective performance, clinical validity, diagnostic safety, or suitability for patient care.",
            "The frozen release contains conflicting CC0 and CC BY-NC 4.0 statements; project governance applies the more restrictive statement.",
        ],
    }
    write_json(ARTIFACTS / "evaluation_result.json", result)
    write_json(EXPERIMENT / "result.json", result)
    write_json(ARTIFACTS / "domain_gap.json", gap)
    write_json(ARTIFACTS / "failure_taxonomy_summary.json", taxonomy)
    write_json(ARTIFACTS / "pixel_calibration.json", pixel_calibration)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "classifier": classifier_subject,
                "segmentation": segmentation,
                "reliability": reliability_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
