"""Evaluate the frozen E08+C03+R02 stack on audited Cheng overlap-control data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import advanced_calibration_common as calibration_common
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_failure_analysis as failure_analysis
import segmentation_overfit_gate as overfit
import segmentation_system
import stage6_external_common as external
import torch
import train_classifier as classifier_training
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x01_cheng_overlap_control"
ARTIFACTS = EXPERIMENT / "artifacts"
EXTRACTED = ROOT / "data/external/cheng_figshare_v8/extracted"
AUDIT_RESULT = ARTIFACTS / "dataset_audit.json"
DATASET_MANIFEST = ARTIFACTS / "dataset_manifest.csv"
E05_RUN = ROOT / "runs/segmentation_v2_weighted/20260924-055049-892543-weighted"
E06_RUN = ROOT / "runs/segmentation_v2_boundary_loss/20260924-063051-647842-boundary"
E07_RUN = (
    ROOT / "runs/segmentation_v2_glioma_specialized/20260924-105229-241365-specialist"
)
V1_CHECKPOINT = (
    ROOT
    / "runs/segmentation/20260923-072439-397263-full/top3_checkpoints/epoch_056_macro_dice_0.762662.pt"
)
CLASSIFIER_CHECKPOINT = (
    ROOT / "runs/classifier/20260921-170607-243133-baseline/best_model.pt"
)
BRISC_REFERENCE_PREDICTIONS = (
    ROOT
    / "experiments/x00_r02_deployable_model/artifacts/operational_reference_predictions.csv"
)
LABELS = ("glioma", "meningioma", "pituitary")
CLASS_NAMES = tuple(classifier_training.CLASSES)
LABEL_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
EXPECTED_HASHES = {
    "classifier": "3bc044027b83b2de225ebde3fb817be8827d2b427c599824697b13ff7a727535",
    "v1": "16dc8745993eef959d0f2924c9147a45320673f6fc0b465ab08f8ad7b2f723de",
    "e05": "e17e7f5afec1c56e9faf12e12db263eff4a3ebc184cfcd7bdfcfff0c1a977c6a",
    "e06": "b98afabc8118471e01f560ff6a727ce25c08001574bf12005d77d1be773f43ab",
    "e07": "bb68ce4f0a9203d74a5ec9dea2e83468d3d833ac285a13ebbdb1d793a0d3bbe0",
}
BOOTSTRAPS = 2000
SEED = 42
EPSILON = 1e-12


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_checkpoint(run: Path) -> Path:
    result = read_json(run / "result.json")
    return run / "top3_checkpoints" / result["top3"][0]["checkpoint"]


def load_cheng(index: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(EXTRACTED / f"{index}.mat", "r") as handle:
        image = np.asarray(handle["cjdata/image"]).T
        mask = np.asarray(handle["cjdata/tumorMask"]).T
    return image, mask


class ChengDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position: int):
        row = self.rows[position]
        image, mask = load_cheng(int(row["source_index"]))
        segmenter, geometry, _window = external.preprocess_segmenter(image)
        classifier, _class_geometry, _class_window = external.preprocess_classifier(
            image
        )
        target, mask_geometry = external.preprocess_reference_mask(
            mask, allow_empty=False
        )
        if geometry != mask_geometry:
            raise ValueError(f"Image/mask transform mismatch: {row['source_index']}")
        return (
            torch.from_numpy(segmenter),
            torch.from_numpy(classifier),
            torch.from_numpy(target.astype(np.float32)),
            position,
        )


def load_single_segmenter(run: Path, expected_hash: str, device: torch.device):
    checkpoint = selected_checkpoint(run)
    if sha256(checkpoint) != expected_hash:
        raise ValueError(f"Segmenter hash mismatch: {checkpoint}")
    model = segmentation_system.load_selected_system(
        run, checkpoint, device, expected_hash
    )
    if isinstance(model, segmentation_system.RoutedSegmenter):
        raise TypeError("Unexpected path-routed model in single-component slot")
    model.eval()
    return model


def load_raw_segmenter(checkpoint: Path, expected_hash: str, device: torch.device):
    if sha256(checkpoint) != expected_hash:
        raise ValueError(f"Segmenter hash mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def load_models(device: torch.device):
    external.verify_frozen_artifacts()
    if sha256(CLASSIFIER_CHECKPOINT) != EXPECTED_HASHES["classifier"]:
        raise ValueError("Classifier checkpoint hash mismatch")
    classifier_state = torch.load(
        CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=True
    )
    classifier = resnet18(weights=None, num_classes=4).to(device)
    classifier.load_state_dict(classifier_state["model_state"])
    classifier.eval()
    models = {
        "e05": load_single_segmenter(E05_RUN, EXPECTED_HASHES["e05"], device),
        "e06": load_single_segmenter(E06_RUN, EXPECTED_HASHES["e06"], device),
        "e07_specialist": load_raw_segmenter(
            selected_checkpoint(E07_RUN), EXPECTED_HASHES["e07"], device
        ),
        "v1_fallback": load_raw_segmenter(V1_CHECKPOINT, EXPECTED_HASHES["v1"], device),
    }
    return classifier, models


def mask_scores(prediction: np.ndarray, reference: np.ndarray):
    axes = (1, 2)
    intersection = np.sum(prediction & reference, axis=axes)
    predicted = np.sum(prediction, axis=axes)
    target = np.sum(reference, axis=axes)
    dice = 2 * intersection / np.maximum(predicted + target, 1)
    iou = intersection / np.maximum(predicted + target - intersection, 1)
    return dice.astype(np.float64), iou.astype(np.float64)


def boundary_metrics(
    prediction: np.ndarray, reference: np.ndarray, mm_per_pixel: float
):
    structure = np.ones((3, 3), dtype=bool)
    predicted_boundary = prediction ^ binary_erosion(prediction, structure=structure)
    reference_boundary = reference ^ binary_erosion(reference, structure=structure)
    if not predicted_boundary.any():
        return 0.0, None
    prediction_close = binary_dilation(
        predicted_boundary, structure=structure, iterations=2
    )
    reference_close = binary_dilation(
        reference_boundary, structure=structure, iterations=2
    )
    precision = float(
        (predicted_boundary & reference_close).sum() / predicted_boundary.sum()
    )
    recall = float(
        (reference_boundary & prediction_close).sum() / reference_boundary.sum()
    )
    boundary_f1 = 2 * precision * recall / max(precision + recall, EPSILON)
    pred_points = np.column_stack(np.where(predicted_boundary)).astype(np.float64)
    ref_points = np.column_stack(np.where(reference_boundary)).astype(np.float64)
    pred_tree = cKDTree(pred_points)
    ref_tree = cKDTree(ref_points)
    hausdorff_pixels = max(
        float(ref_tree.query(pred_points, k=1)[0].max()),
        float(pred_tree.query(ref_points, k=1)[0].max()),
    )
    return boundary_f1, hausdorff_pixels * mm_per_pixel


def equal_frequency_calibration(probability, target, bins=15):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    order = np.argsort(probability, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    rows = []
    ece = 0.0
    mce = 0.0
    for number, members in enumerate(groups, 1):
        confidence = float(probability[members].mean())
        observed = float(target[members].mean())
        gap = abs(confidence - observed)
        weight = len(members) / len(order)
        ece += weight * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": number,
                "count": len(members),
                "mean_probability": confidence,
                "observed_frequency": observed,
                "absolute_gap": gap,
            }
        )
    return {"ece": float(ece), "mce": float(mce), "bins": rows}


def percentile_ci(values) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def subject_index(rows: list[dict]) -> tuple[list[str], dict[str, np.ndarray]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[row["patient_id"]].append(index)
    keys = sorted(members)
    return keys, {key: np.asarray(members[key], dtype=np.int64) for key in keys}


def cluster_bootstrap_indices(keys, members, generator):
    sampled = generator.choice(keys, size=len(keys), replace=True)
    return np.concatenate([members[key] for key in sampled])


def classifier_metrics(labels: np.ndarray, probabilities: np.ndarray):
    predicted = probabilities.argmax(axis=1)
    onehot = np.eye(4, dtype=np.float64)[labels]
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    calibration = equal_frequency_calibration(confidence, correct)
    metrics = {
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(
                np.clip(probabilities[np.arange(len(labels)), labels], EPSILON, 1)
            ).mean()
        ),
        "brier": float(np.square(probabilities - onehot).sum(axis=1).mean()),
        "top_label_ece": calibration["ece"],
        "top_label_mce": calibration["mce"],
        "recall": {},
        "one_vs_rest_auroc": {},
        "confusion": {},
        "reliability_bins": calibration["bins"],
    }
    for name in LABELS:
        true_index = LABEL_INDEX[name]
        member = labels == true_index
        metrics["recall"][name] = float((predicted[member] == true_index).mean())
        metrics["one_vs_rest_auroc"][name] = float(
            roc_auc_score(member, probabilities[:, true_index])
        )
        metrics["confusion"][name] = {
            predicted_name: int(np.sum(member & (predicted == predicted_index)))
            for predicted_index, predicted_name in enumerate(CLASS_NAMES)
        }
    return metrics


def subject_classifier_evaluation(rows, probabilities):
    keys, members = subject_index(rows)
    labels = []
    subject_probability = []
    for key in keys:
        indices = members[key]
        label_values = {LABEL_INDEX[rows[index]["label"]] for index in indices}
        if len(label_values) != 1:
            raise ValueError("Patient has multiple true categories")
        labels.append(next(iter(label_values)))
        subject_probability.append(probabilities[indices].mean(axis=0))
    labels = np.asarray(labels, dtype=np.int64)
    subject_probability = np.asarray(subject_probability, dtype=np.float64)
    metrics = classifier_metrics(labels, subject_probability)
    generator = np.random.default_rng(SEED)
    bootstrap = defaultdict(list)
    valid = 0
    attempts = 0
    while valid < BOOTSTRAPS and attempts < 10000:
        attempts += 1
        indices = generator.integers(0, len(keys), size=len(keys))
        selected_labels = labels[indices]
        if len(set(selected_labels.tolist())) != 3:
            continue
        value = classifier_metrics(selected_labels, subject_probability[indices])
        for key in ("accuracy", "nll", "brier", "top_label_ece", "top_label_mce"):
            bootstrap[key].append(value[key])
        for name in LABELS:
            bootstrap[f"recall_{name}"].append(value["recall"][name])
            bootstrap[f"auroc_{name}"].append(value["one_vs_rest_auroc"][name])
        valid += 1
    if valid != BOOTSTRAPS:
        raise ValueError("Could not obtain 2,000 valid classifier bootstrap draws")
    metrics["bootstrap_ci95"] = {
        key: percentile_ci(value) for key, value in bootstrap.items()
    }
    metrics["subjects"] = len(keys)
    return metrics


def mean_with_patient_bootstrap(values, rows, category=None):
    keys, members = subject_index(rows)
    if category is not None:
        keys = [key for key in keys if rows[int(members[key][0])]["label"] == category]
    patient_values = np.asarray([np.mean(values[members[key]]) for key in keys])
    generator = np.random.default_rng(SEED)
    samples = [
        float(patient_values[generator.integers(0, len(keys), size=len(keys))].mean())
        for _ in range(BOOTSTRAPS)
    ]
    return {
        "subject_macro_mean": float(patient_values.mean()),
        "bootstrap_ci95": percentile_ci(samples),
        "subjects": len(keys),
        "slices": int(sum(len(members[key]) for key in keys)),
    }


def pixel_region_contributions(probability, target, region):
    selected_probability = probability[region].astype(np.float64)
    selected_target = target[region].astype(np.float64)
    clipped = np.clip(selected_probability, EPSILON, 1 - EPSILON)
    nll = float(
        np.mean(
            -selected_target * np.log(clipped)
            - (1 - selected_target) * np.log(1 - clipped)
        )
    )
    brier = float(np.square(selected_probability - selected_target).mean())
    bins = np.minimum((selected_probability * 15).astype(np.int64), 14)
    count = np.bincount(bins, minlength=15).astype(np.float64)
    denominator = len(selected_probability)
    return {
        "nll": nll,
        "brier": brier,
        "weight": count / denominator,
        "confidence": np.bincount(bins, weights=selected_probability, minlength=15)
        / denominator,
        "observed": np.bincount(bins, weights=selected_target, minlength=15)
        / denominator,
    }


def summarize_pixel_contributions(items, rows):
    keys, members = subject_index(rows)
    arrays = {
        name: np.asarray([item[name] for item in items])
        for name in ("nll", "brier", "weight", "confidence", "observed")
    }
    patient = {
        name: np.asarray([arrays[name][members[key]].mean(axis=0) for key in keys])
        for name in arrays
    }

    def aggregate(indices):
        weight = patient["weight"][indices].mean(axis=0)
        confidence_sum = patient["confidence"][indices].mean(axis=0)
        observed_sum = patient["observed"][indices].mean(axis=0)
        occupied = weight > 0
        confidence = np.divide(confidence_sum, weight, out=np.zeros(15), where=occupied)
        observed = np.divide(observed_sum, weight, out=np.zeros(15), where=occupied)
        gaps = np.abs(confidence - observed)
        return {
            "nll": float(patient["nll"][indices].mean()),
            "brier": float(patient["brier"][indices].mean()),
            "ece": float(np.sum(weight[occupied] * gaps[occupied])),
            "mce": float(gaps[occupied].max(initial=0)),
        }

    point = aggregate(np.arange(len(keys)))
    generator = np.random.default_rng(SEED)
    bootstrap = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        indices = generator.integers(0, len(keys), size=len(keys))
        value = aggregate(indices)
        for name in value:
            bootstrap[name].append(value[name])
    return {
        **point,
        "bootstrap_ci95": {
            name: percentile_ci(value) for name, value in bootstrap.items()
        },
        "weighting": "each patient equal; each slice equal within patient",
    }


def reliability_metrics(scores, success, rows):
    scores = np.asarray(scores, dtype=np.float64)
    success = np.asarray(success, dtype=bool)
    risk = 1 - scores
    failure = ~success
    training_success_prevalence = 598 / 680
    baseline_brier = float(np.mean(np.square(training_success_prevalence - success)))
    brier = float(np.mean(np.square(scores - success)))
    calibration = equal_frequency_calibration(scores, success)
    metrics = {
        "failure_detection_auroc": float(roc_auc_score(failure, risk)),
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "brier_skill": 1 - brier / baseline_brier,
        "bins": calibration["bins"],
    }
    keys, members = subject_index(rows)
    generator = np.random.default_rng(SEED)
    bootstrap = defaultdict(list)
    valid = 0
    attempts = 0
    while valid < BOOTSTRAPS and attempts < 10000:
        attempts += 1
        indices = cluster_bootstrap_indices(keys, members, generator)
        selected_failure = failure[indices]
        if not selected_failure.any() or selected_failure.all():
            continue
        selected_success = success[indices]
        selected_scores = scores[indices]
        selected_brier = float(np.mean(np.square(selected_scores - selected_success)))
        selected_baseline = float(
            np.mean(np.square(training_success_prevalence - selected_success))
        )
        selected_calibration = equal_frequency_calibration(
            selected_scores, selected_success
        )
        bootstrap["failure_detection_auroc"].append(
            float(roc_auc_score(selected_failure, 1 - selected_scores))
        )
        bootstrap["ece"].append(selected_calibration["ece"])
        bootstrap["mce"].append(selected_calibration["mce"])
        bootstrap["brier"].append(selected_brier)
        bootstrap["brier_skill"].append(1 - selected_brier / selected_baseline)
        valid += 1
    if valid != BOOTSTRAPS:
        raise ValueError("Could not obtain 2,000 valid reliability bootstrap draws")
    metrics["bootstrap_ci95"] = {
        name: percentile_ci(value) for name, value in bootstrap.items()
    }
    metrics["per_category"] = {}
    for category in LABELS:
        indices = np.asarray([row["label"] == category for row in rows])
        category_failure = failure[indices]
        category_scores = scores[indices]
        if category_failure.any() and not category_failure.all():
            point = float(roc_auc_score(category_failure, 1 - category_scores))
            category_rows = [row for row in rows if row["label"] == category]
            category_keys, category_members = subject_index(category_rows)
            category_generator = np.random.default_rng(SEED)
            values = []
            for _ in range(BOOTSTRAPS):
                draw = cluster_bootstrap_indices(
                    category_keys, category_members, category_generator
                )
                selected = category_failure[draw]
                if selected.any() and not selected.all():
                    values.append(
                        float(roc_auc_score(selected, 1 - category_scores[draw]))
                    )
            metrics["per_category"][category] = {
                "failure_detection_auroc": point,
                "bootstrap_ci95": percentile_ci(values),
                "valid_bootstrap_draws": len(values),
            }
        else:
            metrics["per_category"][category] = "NOT_ESTIMABLE"
    return metrics


def plot_reliability(path: Path, bins, title: str):
    figure, axis = plt.subplots(figsize=(6.4, 5.8))
    x = [item["mean_probability"] for item in bins]
    y = [item["observed_frequency"] for item in bins]
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axis.plot(x, y, marker="o", label="Observed")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        title=title,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_visual_panel(path: Path, rows, calibrated, targets, dice):
    selected = []
    for category in LABELS:
        members = [index for index, row in enumerate(rows) if row["label"] == category]
        ordered = sorted(
            members, key=lambda index: (dice[index], int(rows[index]["source_index"]))
        )
        selected.extend(ordered[:3] + ordered[-3:])
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    for output_index, index in enumerate(selected):
        image, _mask = load_cheng(int(rows[index]["source_index"]))
        display, _ = external.display_uint8(image)
        model_input, _geometry, _window = external.preprocess_segmenter(image)
        base = model_input[0]
        reference = targets[index]
        prediction = calibrated[index] >= 0.5
        axes[output_index, 0].imshow(display, cmap="gray")
        axes[output_index, 0].set_title(
            f"{rows[index]['label']} #{rows[index]['source_index']} source"
        )
        axes[output_index, 1].imshow(base, cmap="gray")
        axes[output_index, 1].imshow(
            np.ma.masked_where(~reference, reference), cmap="Reds", alpha=0.55
        )
        axes[output_index, 1].set_title("Reference (red)")
        axes[output_index, 2].imshow(base, cmap="gray")
        axes[output_index, 2].imshow(
            np.ma.masked_where(~prediction, prediction), cmap="Blues", alpha=0.55
        )
        axes[output_index, 2].set_title(f"Prediction (cyan), Dice {dice[index]:.3f}")
        for axis in axes[output_index]:
            axis.axis("off")
    figure.suptitle("X01 deterministic best/worst overlap-control panel", y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def brisc_domain_reference(classifier, device):
    c03_result = read_json(ROOT / "experiments/c03_isotonic/result.json")
    cache_run, order, raw_probability, targets, _labels = calibration_common.load_cache(
        c03_result["verification"]["cache_manifest_sha256"]
    )
    if len(order) != 680:
        raise ValueError("BRISC development reference membership changed")
    reliability_lookup = {
        row["path"]: float(row["operational_probability"])
        for row in read_csv(BRISC_REFERENCE_PREDICTIONS)
    }
    reliability = np.asarray([reliability_lookup[row["path"]] for row in order])
    embeddings = []
    intensity = []
    captured = []

    def hook(_module, _input, output):
        captured.append(output.detach().float().flatten(1).cpu().numpy())

    handle = classifier.avgpool.register_forward_hook(hook)
    loader = DataLoader(
        classifier_training.BrainImages(order, training=False),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        for images, _labels_batch in loader:
            images = images.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                classifier(images)
            embeddings.append(captured.pop(0))
    handle.remove()
    for row in order:
        with Image.open(ROOT / row["path"]) as source:
            tensor, _geometry, _window = external.preprocess_segmenter(
                np.asarray(source.convert("L"))
            )
        nonzero = tensor[tensor > 0]
        intensity.append(
            [
                float(tensor.mean()),
                float(tensor.std()),
                float(np.mean(nonzero)) if len(nonzero) else 0.0,
            ]
        )
    return {
        "cache_run": cache_run.as_posix(),
        "order": order,
        "raw_probability": raw_probability,
        "targets": targets,
        "reliability": reliability,
        "embeddings": np.concatenate(embeddings),
        "intensity": np.asarray(intensity),
    }


def domain_gap(
    rows,
    external_intensity,
    external_embeddings,
    external_targets,
    external_reliability,
    brisc,
):
    brisc_targets = brisc["targets"]
    ext_foreground = external_targets.mean(axis=(1, 2))
    brisc_foreground = brisc_targets.mean(axis=(1, 2))
    embedding_wasserstein = [
        wasserstein_distance(
            external_embeddings[:, index], brisc["embeddings"][:, index]
        )
        for index in range(external_embeddings.shape[1])
    ]
    result = {
        "reference": "BRISC development validation only; sealed test pixels unopened",
        "intensity_wasserstein": {
            "model_input_mean": float(
                wasserstein_distance(external_intensity[:, 0], brisc["intensity"][:, 0])
            ),
            "model_input_sd": float(
                wasserstein_distance(external_intensity[:, 1], brisc["intensity"][:, 1])
            ),
            "nonzero_mean": float(
                wasserstein_distance(external_intensity[:, 2], brisc["intensity"][:, 2])
            ),
        },
        "foreground_area_fraction_wasserstein": float(
            wasserstein_distance(ext_foreground, brisc_foreground)
        ),
        "encoder_representation_shift": {
            "centroid_euclidean": float(
                np.linalg.norm(
                    external_embeddings.mean(axis=0) - brisc["embeddings"].mean(axis=0)
                )
            ),
            "dimension_mean_wasserstein": float(np.mean(embedding_wasserstein)),
            "dimension_median_wasserstein": float(np.median(embedding_wasserstein)),
            "encoder": "frozen classifier ResNet18 avgpool (512 dimensions)",
        },
        "reliability_score_wasserstein": float(
            wasserstein_distance(external_reliability, brisc["reliability"])
        ),
        "policy_review_rate": {
            "cheng": float(
                np.mean(external_reliability < external.RELIABILITY_POLICY_THRESHOLD)
            ),
            "brisc_operational_reference": float(
                np.mean(brisc["reliability"] < external.RELIABILITY_POLICY_THRESHOLD)
            ),
        },
        "counts": {
            "cheng_slices": len(rows),
            "brisc_validation_slices": len(brisc["order"]),
        },
    }
    return result


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
        ChengDataset(rows),
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
                routed[torch.from_numpy(glioma_route).to(device)] = (
                    specialist.float().sigmoid()[
                        torch.from_numpy(glioma_route).to(device)
                    ]
                )
            ensemble = (e05.float().sigmoid() + e06.float().sigmoid() + routed) / 3.0
            batch_raw = ensemble[:, 0].cpu().numpy().astype(np.float32)
            batch_calibrated = external.apply_operational_c03(batch_raw).astype(
                np.float32
            )
            raw_probability[index_values] = batch_raw
            calibrated[index_values] = batch_calibrated
            targets[index_values] = references
            class_probability[index_values] = probabilities
            embeddings[index_values] = captured.pop(0)
            for position, index in enumerate(index_values):
                route[index] = (
                    "glioma_specialist" if glioma_route[position] else "v1_fallback"
                )
                reliability[index] = external.operational_reliability(
                    batch_calibrated[position]
                )["score"]
                values = segmenter_images[position, 0].float().cpu().numpy()
                nonzero = values[values > 0]
                intensity[index] = [
                    values.mean(),
                    values.std(),
                    nonzero.mean() if len(nonzero) else 0.0,
                ]
    hook_handle.remove()
    if not all(
        np.isfinite(array).all()
        for array in (
            raw_probability,
            calibrated,
            class_probability,
            embeddings,
            reliability,
            intensity,
        )
    ):
        raise ValueError("Inference produced a non-finite output")
    return (
        raw_probability,
        calibrated,
        targets,
        class_probability,
        embeddings,
        reliability,
        intensity,
        route,
    )


def main() -> None:
    config = read_json(EXPERIMENT / "config.json")
    audit = read_json(AUDIT_RESULT)
    if (
        config["status"] != "AUDIT_COMPLETE"
        or audit["audit_outcome"] != "ELIGIBLE_OVERLAP_CONTROL"
    ):
        raise ValueError("X01 does not have a passed audit-only state")
    if config["segmentation_threshold"] != 0.5:
        raise ValueError("Segmentation threshold changed")
    if config["reliability_policy_threshold"] != "0.9490250173235835":
        raise ValueError("Reliability threshold representation changed")
    rows = read_csv(DATASET_MANIFEST)
    if len(rows) != 3064 or len({row["source_index"] for row in rows}) != 3064:
        raise ValueError("Audited Cheng manifest membership changed")
    frozen = external.verify_frozen_artifacts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, models = load_models(device)
    (
        raw_probability,
        calibrated,
        targets,
        class_probability,
        embeddings,
        reliability,
        intensity,
        route,
    ) = run_inference(rows, classifier, models, device)

    hard = calibrated >= 0.5
    raw_hard = raw_probability >= 0.5
    dice, iou = mask_scores(hard, targets)
    raw_dice, raw_iou = mask_scores(raw_hard, targets)
    predicted_class = class_probability.argmax(axis=1)
    classifier_subject = subject_classifier_evaluation(rows, class_probability)

    boundary_f1 = np.empty(len(rows), dtype=np.float64)
    hausdorff_mm = np.full(len(rows), np.nan, dtype=np.float64)
    per_case = []
    failure_rows = []
    for index, row in enumerate(rows):
        width = int(row["width"])
        mm_per_pixel = 0.49 * width / 128
        boundary_f1[index], hausdorff = boundary_metrics(
            hard[index], targets[index], mm_per_pixel
        )
        if hausdorff is not None:
            hausdorff_mm[index] = hausdorff
        reliability_status = (
            "STANDARD"
            if reliability[index] >= external.RELIABILITY_POLICY_THRESHOLD
            else "REVIEW_REQUIRED"
        )
        record = {
            "source_index": row["source_index"],
            "patient_id": row["patient_id"],
            "category": row["label"],
            "official_cv_fold": row["official_cv_fold"],
            "classifier_predicted": CLASS_NAMES[predicted_class[index]],
            **{
                f"classifier_probability_{name}": float(
                    class_probability[index, position]
                )
                for position, name in enumerate(CLASS_NAMES)
            },
            "e07_route": route[index],
            "raw_dice": float(raw_dice[index]),
            "raw_iou": float(raw_iou[index]),
            "calibrated_dice": float(dice[index]),
            "calibrated_iou": float(iou[index]),
            "reference_pixels": int(targets[index].sum()),
            "predicted_pixels": int(hard[index].sum()),
            "boundary_f1_tolerance_2": float(boundary_f1[index]),
            "hausdorff_mm": None
            if np.isnan(hausdorff_mm[index])
            else float(hausdorff_mm[index]),
            "r02_reliability": float(reliability[index]),
            "policy_status": reliability_status,
            "failure_dice_below_0_50": bool(dice[index] < 0.5),
        }
        per_case.append(record)
        if dice[index] < 0.5:
            failure_rows.append(
                {
                    **record,
                    **failure_analysis.component_statistics(
                        hard[index], targets[index]
                    ),
                }
            )
    write_csv(ARTIFACTS / "per_case_predictions.csv", per_case)
    write_csv(ARTIFACTS / "failure_taxonomy.csv", failure_rows)

    segmentation = {
        "slice_macro": {
            "dice": float(dice.mean()),
            "iou": float(iou.mean()),
            "raw_dice": float(raw_dice.mean()),
            "raw_iou": float(raw_iou.mean()),
            "boundary_f1_tolerance_2": float(boundary_f1.mean()),
            "hausdorff_mm_nonempty_predictions": float(np.nanmean(hausdorff_mm)),
        },
        "subject_macro": {
            "dice": mean_with_patient_bootstrap(dice, rows),
            "iou": mean_with_patient_bootstrap(iou, rows),
            "boundary_f1_tolerance_2": mean_with_patient_bootstrap(boundary_f1, rows),
        },
        "per_category": {},
        "zero_dice": int(np.sum(dice == 0)),
        "empty_prediction": int(np.sum(hard.sum(axis=(1, 2)) == 0)),
        "dice_below_0_50": int(np.sum(dice < 0.5)),
        "whole_volume": "NOT_ESTIMABLE: slice order and per-slice orientation are unavailable",
    }
    for category in LABELS:
        indices = np.asarray([row["label"] == category for row in rows])
        segmentation["per_category"][category] = {
            "dice": mean_with_patient_bootstrap(dice, rows, category),
            "iou": mean_with_patient_bootstrap(iou, rows, category),
            "slice_mean_dice": float(dice[indices].mean()),
            "slice_mean_iou": float(iou[indices].mean()),
            "boundary_f1_tolerance_2": mean_with_patient_bootstrap(
                boundary_f1, rows, category
            ),
            "failures_below_0_50": int(np.sum(dice[indices] < 0.5)),
            "zero_dice": int(np.sum(dice[indices] == 0)),
        }

    regions = {
        "lesion_centered": np.stack(
            [
                calibration_common.lesion_regions(targets[index : index + 1])[0]
                for index in range(len(rows))
            ]
        ),
    }
    regions["background"] = ~regions["lesion_centered"]
    regions["whole_image"] = np.ones_like(targets, dtype=bool)
    pixel_calibration = {}
    for region_name, region_masks in regions.items():
        raw_items = [
            pixel_region_contributions(
                raw_probability[index], targets[index], region_masks[index]
            )
            for index in range(len(rows))
        ]
        calibrated_items = [
            pixel_region_contributions(
                calibrated[index], targets[index], region_masks[index]
            )
            for index in range(len(rows))
        ]
        pixel_calibration[region_name] = {
            "raw": summarize_pixel_contributions(raw_items, rows),
            "c03_operational": summarize_pixel_contributions(calibrated_items, rows),
        }

    success = dice >= 0.5
    reliability_result = reliability_metrics(reliability, success, rows)
    order = np.argsort(reliability)
    top_count = math.ceil(0.20 * len(rows))
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
        "p01_total_cost": int(
            (review & ~success).sum()
            + (review & success).sum()
            + 10 * (~review & ~success).sum()
        ),
    }

    taxonomy = failure_analysis.summarize(failure_rows)
    brisc = brisc_domain_reference(classifier, device)
    gap = domain_gap(rows, intensity, embeddings, targets, reliability, brisc)
    render_visual_panel(
        ARTIFACTS / "visual_audit_panel.png", rows, calibrated, targets, dice
    )
    plot_reliability(
        ARTIFACTS / "classifier_reliability.png",
        classifier_subject["reliability_bins"],
        "X01 patient-level classifier reliability",
    )
    plot_reliability(
        ARTIFACTS / "case_reliability.png",
        reliability_result["bins"],
        "X01 case reliability transfer",
    )

    gates = {
        "classifier_recall_at_least_0_85": {
            category: classifier_subject["recall"][category] >= 0.85
            for category in LABELS
        },
        "classifier_top_label_ece_at_most_0_05": classifier_subject["top_label_ece"]
        <= 0.05,
        "segmentation_dice": {
            "glioma_at_least_0_55": segmentation["per_category"]["glioma"]["dice"][
                "subject_macro_mean"
            ]
            >= 0.55,
            "meningioma_at_least_0_75": segmentation["per_category"]["meningioma"][
                "dice"
            ]["subject_macro_mean"]
            >= 0.75,
            "pituitary_at_least_0_75": segmentation["per_category"]["pituitary"][
                "dice"
            ]["subject_macro_mean"]
            >= 0.75,
        },
        "reliability": {
            "auroc_at_least_0_75": reliability_result["failure_detection_auroc"]
            >= 0.75,
            "auroc_ci_lower_at_least_0_70": reliability_result["bootstrap_ci95"][
                "failure_detection_auroc"
            ][0]
            >= 0.70,
            "ece_at_most_0_05": reliability_result["ece"] <= 0.05,
            "brier_skill_positive": reliability_result["brier_skill"] > 0,
            "brier_skill_ci_lower_positive": reliability_result["bootstrap_ci95"][
                "brier_skill"
            ][0]
            > 0,
            "top20_failure_capture_at_least_half": reliability_result["top20_risk"][
                "failures_captured"
            ]
            / max(reliability_result["top20_risk"]["failures_total"], 1)
            >= 0.50,
        },
    }
    all_gate_values = []
    for value in gates.values():
        all_gate_values.extend(value.values() if isinstance(value, dict) else [value])

    result = {
        "schema_version": 1,
        "experiment_id": "x01_cheng_overlap_control",
        "status": "EVALUATION_COMPLETE_VISUAL_REVIEW_PENDING",
        "audit_outcome": "ELIGIBLE_OVERLAP_CONTROL",
        "independent_external_evidence": False,
        "scope": "Cheng/Figshare source-overlap control; not independent external validation and not clinical validation",
        "population": {
            "slices": len(rows),
            "patients": len({row["patient_id"] for row in rows}),
            "categories": dict(Counter(row["label"] for row in rows)),
        },
        "classifier": classifier_subject,
        "segmentation": segmentation,
        "pixel_calibration": pixel_calibration,
        "reliability_transfer": reliability_result,
        "failure_taxonomy": taxonomy,
        "domain_gap": gap,
        "gates": {
            "status": "PASS" if all(all_gate_values) else "FAIL",
            "checks": gates,
            "not_independent": True,
        },
        "verification": {
            "frozen_hashes": {**frozen, **EXPECTED_HASHES},
            "audit_result_sha256": sha256(AUDIT_RESULT),
            "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
            "device": str(device),
            "bootstrap_repetitions": BOOTSTRAPS,
            "bootstrap_seed": SEED,
            "bootstrap_unit": "patient_id",
            "segmentation_threshold": 0.5,
            "reliability_threshold": "0.9490250173235835",
            "brisc_files_opened": {"development_validation": 680, "sealed_test": 0},
            "external_cases_evaluated": 3064,
            "neural_retraining": False,
            "calibrator_refit": False,
            "reliability_refit": False,
            "threshold_changed": False,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "limitations": [
            "Cheng is an upstream source of BRISC and therefore cannot establish independent transport.",
            "Some Cheng slices are source-equivalent candidates for rows in the sealed BRISC test metadata; no BRISC test file was opened.",
            "Per-slice orientation and reliable 3D ordering are unavailable, so volume and depth are not estimable.",
            "All findings are retrospective research measurements and do not establish clinical validity.",
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
                "segmentation": segmentation,
                "classifier": classifier_subject,
                "reliability": reliability_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
