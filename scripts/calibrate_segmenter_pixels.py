"""Cross-fit pixel temperature scaling for the frozen BRISC segmenter."""

import csv
import hashlib
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import torch
import train_segmenter as segmenter
from monai.losses import DiceLoss
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize_scalar
from scipy.special import expit
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "20260923-072439-397263-full"
RUN = ROOT / "runs" / "segmentation" / RUN_NAME
CHECKPOINT = RUN / "top3_checkpoints" / "epoch_056_macro_dice_0.762662.pt"
FINDINGS = ROOT / "audit" / "segmentation_training_findings.json"
CLASSIFIER_FOLDS = (
    ROOT
    / "runs"
    / "classifier"
    / "20260921-170607-243133-baseline"
    / "classifier_calibration_folds.csv"
)
OUTPUT_JSON = ROOT / "audit" / "segmenter_pixel_calibration.json"
OUTPUT_MD = ROOT / "audit" / "segmenter_pixel_calibration.md"
CALIBRATION_FINDINGS = ROOT / "audit" / "confidence_calibration_findings.json"
FOREGROUND_DIAGRAM = RUN / "segmenter_foreground_reliability_diagram.png"
WHOLE_DIAGRAM = RUN / "segmenter_whole_image_reliability_diagram.png"
FOLD_RESULTS = RUN / "segmenter_pixel_calibration_fold_results.csv"
OOF_RESULTS = RUN / "segmenter_pixel_calibration_oof_metrics.csv"

LABELS = segmenter.LABELS
REGIONS = ("lesion_centered", "background", "whole_image")
FOLDS = 5
SEED = 42
BOOTSTRAP_REPETITIONS = 2000
BINS = 15
DILATION_RADIUS = 8.0
ECE_LIMITS = {"lesion_centered": 0.05, "background": 0.02, "whole_image": 0.02}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_path(value):
    return value.replace("\\", "/").casefold()


def lesion_region(reference, radius=DILATION_RADIUS):
    reference = np.asarray(reference, dtype=bool)
    if reference.ndim != 2 or not reference.any() or reference.all():
        raise ValueError("Reference mask must be a nonempty 2D foreground mask")
    region = distance_transform_edt(~reference) <= radius
    if not np.all(region[reference]):
        raise ValueError("Lesion-centered region does not include the reference mask")
    if region.all():
        raise ValueError("Lesion-centered region consumed the whole model space")
    return region


def pixel_probabilities(logits, temperature):
    if np.any(~np.isfinite(temperature)) or np.any(np.asarray(temperature) <= 0):
        raise ValueError("Every temperature must be finite and positive")
    return expit(np.asarray(logits, dtype=np.float64) / temperature)


def pixel_nll(logits, targets, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, scaled) - targets * scaled))


def fit_temperature(logits, targets):
    """Fit whole-image pixel NLL; all images have the same 128x128 weight."""

    def objective(log_temperature):
        return pixel_nll(logits, targets, np.exp(log_temperature))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"Pixel temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    if not 0.05 <= temperature <= 20.0:
        raise ValueError("Pixel temperature left its predeclared bounds")
    return temperature


def region_mask(regions, index, name):
    if name == "lesion_centered":
        return regions[index]
    if name == "background":
        return ~regions[index]
    if name == "whole_image":
        return np.ones_like(regions[index], dtype=bool)
    raise ValueError(f"Unknown evaluation region: {name}")


def region_metrics(probabilities, scaled_logits, targets, regions, indices, name):
    """Image-balanced pixel metrics with 15 fixed equal-width bins."""
    indices = np.asarray(indices, dtype=np.int64)
    bin_weight = np.zeros(BINS, dtype=np.float64)
    bin_confidence = np.zeros(BINS, dtype=np.float64)
    bin_observed = np.zeros(BINS, dtype=np.float64)
    per_image_nll = np.empty(len(indices), dtype=np.float64)
    per_image_brier = np.empty(len(indices), dtype=np.float64)
    pixel_counts = np.empty(len(indices), dtype=np.int64)
    for output_index, image_index in enumerate(indices):
        mask = region_mask(regions, image_index, name)
        probability = probabilities[image_index][mask].astype(np.float64)
        logit = scaled_logits[image_index][mask].astype(np.float64)
        target = targets[image_index][mask].astype(np.float64)
        if not len(probability):
            raise ValueError(f"Empty {name} evaluation region")
        pixel_counts[output_index] = len(probability)
        per_image_nll[output_index] = np.mean(
            np.logaddexp(0.0, logit) - target * logit
        )
        per_image_brier[output_index] = np.mean(np.square(probability - target))
        bin_index = np.minimum((probability * BINS).astype(np.int64), BINS - 1)
        image_pixel_weight = 1.0 / (len(indices) * len(probability))
        counts = np.bincount(bin_index, minlength=BINS)
        bin_weight += counts * image_pixel_weight
        bin_confidence += (
            np.bincount(bin_index, weights=probability, minlength=BINS)
            * image_pixel_weight
        )
        bin_observed += (
            np.bincount(bin_index, weights=target, minlength=BINS)
            * image_pixel_weight
        )
    occupied = bin_weight > 0
    mean_confidence = np.divide(
        bin_confidence,
        bin_weight,
        out=np.full(BINS, np.nan),
        where=occupied,
    )
    observed_frequency = np.divide(
        bin_observed,
        bin_weight,
        out=np.full(BINS, np.nan),
        where=occupied,
    )
    gaps = np.abs(mean_confidence - observed_frequency)
    ece = float(np.sum(bin_weight[occupied] * gaps[occupied]))
    mce = float(np.max(gaps[occupied]))
    bins = []
    edges = np.linspace(0.0, 1.0, BINS + 1)
    for bin_index in range(BINS):
        bins.append(
            {
                "bin": bin_index + 1,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "occupied": bool(occupied[bin_index]),
                "image_balanced_weight": float(bin_weight[bin_index]),
                "mean_confidence": (
                    float(mean_confidence[bin_index]) if occupied[bin_index] else None
                ),
                "observed_foreground_frequency": (
                    float(observed_frequency[bin_index]) if occupied[bin_index] else None
                ),
                "absolute_gap": float(gaps[bin_index]) if occupied[bin_index] else None,
            }
        )
    summary = {
        "image_count": len(indices),
        "pixel_nll": float(per_image_nll.mean()),
        "pixel_brier": float(per_image_brier.mean()),
        "pixel_ece": ece,
        "pixel_mce": mce,
        "binning": "15 fixed equal-width bins over [0,1] with image-balanced pixel weights",
        "pixel_count": {
            "minimum_per_image": int(pixel_counts.min()),
            "median_per_image": float(np.median(pixel_counts)),
            "maximum_per_image": int(pixel_counts.max()),
            "total": int(pixel_counts.sum()),
        },
        "bins": bins,
    }
    return summary, per_image_nll, per_image_brier


def compact_metrics(summary):
    return {
        key: summary[key]
        for key in ("image_count", "pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")
    }


def load_fold_assignment(rows):
    fold_rows = read_csv(CLASSIFIER_FOLDS)
    by_path = {row["path"]: row for row in fold_rows}
    if len(by_path) != len(fold_rows):
        raise ValueError("Duplicate path in reused classifier folds")
    assignment = np.full(len(rows), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        source = by_path.get(row["path"])
        if source is None:
            raise ValueError(f"Segmentation validation path has no existing fold: {row['path']}")
        if source["label"] != row["label"] or source["similarity_group"] != row["similarity_group"]:
            raise ValueError(f"Reused fold metadata mismatch: {row['path']}")
        assignment[index] = int(source["fold"]) - 1
    group_folds = {}
    for row, fold in zip(rows, assignment, strict=True):
        group_folds.setdefault(row["similarity_group"], set()).add(int(fold))
    if any(len(value) != 1 for value in group_folds.values()):
        raise ValueError("A segmentation similarity group crosses reused folds")
    if set(assignment) != set(range(FOLDS)):
        raise ValueError("The reused assignment does not contain five folds")
    return assignment


def verify_validation_sources(rows):
    manifest = {
        normalized_path(row["relative_path"]): row
        for row in read_csv(smoke.RAW / "manifest.csv")
    }
    checked = 0
    for row in rows:
        for field in ("path", "mask_path"):
            path = smoke.dataset_path(row[field])
            relative = normalized_path(path.relative_to(smoke.RAW).as_posix())
            published = manifest.get(relative)
            if published is None:
                raise ValueError(f"File missing from BRISC manifest: {row[field]}")
            digest = sha256(path)
            if digest != published["sha256"]:
                raise ValueError(f"BRISC manifest SHA256 mismatch: {row[field]}")
            if field == "path" and digest != row["sha256"]:
                raise ValueError(f"Frozen split image SHA256 mismatch: {row['path']}")
            checked += 1
    return checked


def infer(model, rows, device):
    dataset = segmenter.FullPairs(rows)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    logits_store = np.empty((len(rows), 128, 128), dtype=np.float32)
    targets_store = np.empty((len(rows), 128, 128), dtype=bool)
    dice_loss = DiceLoss(sigmoid=True)
    dice_values = np.empty(len(rows), dtype=np.float64)
    iou_values = np.empty(len(rows), dtype=np.float64)
    label_indices = np.empty(len(rows), dtype=np.int64)
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for images, targets, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(images)
            loss = segmenter.segmentation_loss(logits, targets_device, dice_loss)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite segmenter validation output")
            hard = logits >= 0.0
            reference = targets_device >= 0.5
            intersection = (hard & reference).sum(dim=(1, 2, 3)).float()
            predicted_area = hard.sum(dim=(1, 2, 3)).float()
            reference_area = reference.sum(dim=(1, 2, 3)).float()
            dice = 2 * intersection / (predicted_area + reference_area).clamp_min(1)
            iou = intersection / (
                predicted_area + reference_area - intersection
            ).clamp_min(1)
            total_loss += loss.item() * len(images)
            for position, dataset_index in enumerate(indices.tolist()):
                logits_store[dataset_index] = logits[position, 0].float().cpu().numpy()
                targets_store[dataset_index] = reference[position, 0].cpu().numpy()
                dice_values[dataset_index] = float(dice[position].item())
                iou_values[dataset_index] = float(iou[position].item())
                label_indices[dataset_index] = int(labels[position].item())
    metrics = segmenter.summarize_scores(dice_values, iou_values, label_indices)
    metrics["val_loss"] = total_loss / len(dataset)
    return logits_store, targets_store, dice_values, iou_values, label_indices, metrics


def verify_prior_predictions(rows, dice, iou, targets, logits, tolerance):
    prior_rows = read_csv(RUN / "validation_predictions.csv")
    prior = {normalized_path(row["path"]): row for row in prior_rows}
    if len(prior) != 680:
        raise ValueError("Expected 680 frozen segmentation predictions")
    maximum = 0.0
    for index, row in enumerate(rows):
        old = prior.get(normalized_path(row["path"]))
        if old is None:
            raise ValueError(f"Missing frozen segmentation prediction: {row['path']}")
        differences = (
            abs(dice[index] - float(old["dice"])),
            abs(iou[index] - float(old["iou"])),
        )
        maximum = max(maximum, *differences)
        if maximum > tolerance:
            raise ValueError(f"Frozen per-image metric mismatch: {row['path']}")
        if int(targets[index].sum()) != int(old["reference_pixels"]):
            raise ValueError(f"Frozen reference pixel mismatch: {row['path']}")
        if int((logits[index] >= 0).sum()) != int(old["predicted_pixels"]):
            raise ValueError(f"Frozen predicted pixel mismatch: {row['path']}")
    return maximum


def cross_fit(logits, targets, rows, assignment):
    temperatures = np.empty(len(rows), dtype=np.float64)
    fold_results = []
    groups = np.asarray([row["similarity_group"] for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    for fold in range(FOLDS):
        evaluation = assignment == fold
        fitting = ~evaluation
        temperature = fit_temperature(logits[fitting], targets[fitting])
        temperatures[evaluation] = temperature
        counts = dict(sorted(Counter(labels[evaluation]).items()))
        if set(counts) != set(LABELS):
            raise ValueError(f"Calibration fold {fold + 1} lacks a tumor type")
        fold_results.append(
            {
                "fold": fold + 1,
                "temperature": temperature,
                "fit_images": int(fitting.sum()),
                "evaluation_images": int(evaluation.sum()),
                "fit_similarity_groups": len(set(groups[fitting])),
                "evaluation_similarity_groups": len(set(groups[evaluation])),
                "evaluation_glioma": counts["glioma"],
                "evaluation_meningioma": counts["meningioma"],
                "evaluation_pituitary": counts["pituitary"],
            }
        )
    if not np.isfinite(temperatures).all():
        raise ValueError("At least one image received no OOF temperature")
    return temperatures, fold_results


def paired_group_bootstrap(raw_brier, calibrated_brier, groups):
    groups = np.asarray(groups)
    unique_groups = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    samples = {region: [] for region in REGIONS}
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        for region in REGIONS:
            samples[region].append(
                float(
                    calibrated_brier[region][indices].mean()
                    - raw_brier[region][indices].mean()
                )
            )
    return {
        region: {
            "point_estimate": float(
                calibrated_brier[region].mean() - raw_brier[region].mean()
            ),
            "bootstrap_mean": float(np.mean(values)),
            "ci95_lower": float(np.percentile(values, 2.5)),
            "ci95_upper": float(np.percentile(values, 97.5)),
        }
        for region, values in samples.items()
    }


def render_diagram(path, title, raw, calibrated):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, name, metrics, color in (
        (axes[0], "Raw segmenter probability", raw, "#c44e52"),
        (axes[1], "OOF temperature scaled", calibrated, "#4c72b0"),
    ):
        occupied = [item for item in metrics["bins"] if item["occupied"]]
        x = [item["mean_confidence"] for item in occupied]
        y = [item["observed_foreground_frequency"] for item in occupied]
        weights = [item["image_balanced_weight"] for item in occupied]
        sizes = [30 + 600 * weight for weight in weights]
        axis.plot([0, 1], [0, 1], "--", color="#555555")
        axis.plot(x, y, color=color, linewidth=1.5)
        axis.scatter(x, y, s=sizes, color=color, edgecolor="white", linewidth=0.7)
        axis.set_xlim(-0.005, 1.005)
        axis.set_ylim(-0.005, 1.005)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Mean predicted foreground probability")
        axis.set_ylabel("Observed foreground frequency")
        axis.set_title(
            f"{name}\nECE={metrics['pixel_ece']:.4f}, NLL={metrics['pixel_nll']:.4f}"
        )
        axis.grid(alpha=0.22)
    figure.suptitle(
        f"{title}\n15 equal-width bins; each validation image has equal total weight",
        fontsize=13,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def mask_digest(mask):
    return hashlib.sha256(np.packbits(mask.reshape(-1)).tobytes()).hexdigest()


def markdown_report(result):
    lines = [
        "# Segmenter pixel confidence calibration",
        "",
        "A single positive temperature was fitted through five-fold similarity-aware cross-fitting on 680 frozen BRISC segmentation-validation images. Reported calibrated metrics use held-out fold temperatures only.",
        "",
        "## Temperatures",
        "",
        "| Fold | Temperature | Fit images | Evaluation images |",
        "|---:|---:|---:|---:|",
    ]
    for fold in result["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['temperature']:.8f} | "
            f"{fold['fit_images']} | {fold['evaluation_images']} |"
        )
    lines.extend(
        [
            "",
            f"Proposed all-validation temperature: `{result['final_temperature_all_validation']:.8f}`. It was not used for reported metrics.",
            "",
            "## Region metrics",
            "",
            "| Region | Metric | Raw | OOF calibrated | Change |",
            "|---|---|---:|---:|---:|",
        ]
    )
    labels = {
        "lesion_centered": "Lesion-centered foreground",
        "background": "Background",
        "whole_image": "Whole image",
    }
    for region in REGIONS:
        raw = result["metrics"]["aggregate"][region]["raw"]
        calibrated = result["metrics"]["aggregate"][region]["oof_calibrated"]
        for key, name in (
            ("pixel_nll", "NLL"),
            ("pixel_brier", "Brier"),
            ("pixel_ece", "ECE"),
            ("pixel_mce", "MCE"),
        ):
            lines.append(
                f"| {labels[region]} | {name} | {raw[key]:.8f} | "
                f"{calibrated[key]:.8f} | {calibrated[key] - raw[key]:+.8f} |"
            )
    lines.extend(
        [
            "",
            "## Brier paired bootstrap",
            "",
            "| Region | Point change | 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for region in REGIONS:
        item = result["metrics"]["brier_paired_similarity_group_bootstrap"][region]
        lines.append(
            f"| {labels[region]} | {item['point_estimate']:+.8f} | "
            f"[{item['ci95_lower']:+.8f}, {item['ci95_upper']:+.8f}] |"
        )
    lines.extend(
        [
            "",
            "## Foreground-region ECE by tumor type",
            "",
            "| Type | Raw | OOF calibrated |",
            "|---|---:|---:|",
        ]
    )
    for label in LABELS:
        item = result["metrics"]["per_category"][label]["lesion_centered"]
        lines.append(
            f"| {label} | {item['raw']['pixel_ece']:.8f} | "
            f"{item['oof_calibrated']['pixel_ece']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"**{result['gate']['status']}**",
            "",
        ]
    )
    for name, passed in result["gate"]["checks"].items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            f"Foreground diagram: `{result['artifacts']['foreground_reliability_diagram']}`",
            "",
            f"Whole-image diagram: `{result['artifacts']['whole_image_reliability_diagram']}`",
            "",
            "No model was retrained, no test image or test prediction was accessed, the 0.50 threshold was unchanged, and no application integration was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config = read_json(RUN / "config.json")
    result = read_json(RUN / "result.json")
    findings = read_json(FINDINGS)
    split = RUN / "split_snapshot.csv"
    if config["image_size"] != 128 or config["test_evaluated"]:
        raise ValueError("Unexpected accepted segmenter configuration")
    if sha256(split) != config["split_sha256"]:
        raise ValueError("Segmenter split SHA256 mismatch")
    expected_checkpoint = findings["top3_checkpoints"][0]
    if CHECKPOINT.name != expected_checkpoint["file"]:
        raise ValueError("Checkpoint is not the accepted rank-1 checkpoint")
    if sha256(CHECKPOINT).upper() != expected_checkpoint["sha256"]:
        raise ValueError("Accepted segmenter checkpoint SHA256 mismatch")
    rows = smoke.read_csv(split)
    parts = segmenter.partition_masked_rows(rows)
    validation_rows = parts["validation"]
    if len(validation_rows) != 680:
        raise ValueError("Expected 680 segmentation-validation images")
    source_files_verified = verify_validation_sources(validation_rows)
    assignment = load_fold_assignment(validation_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != result["best_epoch"] or checkpoint["config"] != config:
        raise ValueError("Checkpoint metadata differs from accepted run configuration")
    model.load_state_dict(checkpoint["model_state"])
    logits, targets, dice, iou, _label_indices, recomputed_metrics = infer(
        model, validation_rows, device
    )
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    expected_metrics = result["best_validation_metrics"]
    maximum_metric_difference = max(
        abs(recomputed_metrics[key] - expected_metrics[key])
        for key in expected_metrics
    )
    if maximum_metric_difference > tolerance:
        raise ValueError(
            f"Recomputed segmenter metric mismatch: {maximum_metric_difference}"
        )
    maximum_per_image_difference = verify_prior_predictions(
        validation_rows, dice, iou, targets, logits, tolerance
    )
    regions = np.stack([lesion_region(target) for target in targets])

    temperatures, fold_results = cross_fit(
        logits, targets, validation_rows, assignment
    )
    final_temperature = fit_temperature(logits, targets)
    raw_scaled_logits = logits.astype(np.float64)
    oof_scaled_logits = raw_scaled_logits / temperatures[:, None, None]
    final_scaled_logits = raw_scaled_logits / final_temperature
    raw_probabilities = expit(raw_scaled_logits)
    oof_probabilities = expit(oof_scaled_logits)
    final_probabilities = expit(final_scaled_logits)

    raw_hard = raw_probabilities >= 0.5
    oof_hard = oof_probabilities >= 0.5
    final_hard = final_probabilities >= 0.5
    oof_changed_pixels = int(np.count_nonzero(raw_hard != oof_hard))
    final_changed_pixels = int(np.count_nonzero(raw_hard != final_hard))
    oof_changed_images = int(np.any(raw_hard != oof_hard, axis=(1, 2)).sum())
    final_changed_images = int(np.any(raw_hard != final_hard, axis=(1, 2)).sum())

    aggregate = {}
    raw_brier = {}
    calibrated_brier = {}
    per_image = {}
    all_indices = np.arange(len(validation_rows))
    for region in REGIONS:
        raw_summary, raw_nll_values, raw_brier_values = region_metrics(
            raw_probabilities,
            raw_scaled_logits,
            targets,
            regions,
            all_indices,
            region,
        )
        calibrated_summary, calibrated_nll_values, calibrated_brier_values = (
            region_metrics(
                oof_probabilities,
                oof_scaled_logits,
                targets,
                regions,
                all_indices,
                region,
            )
        )
        aggregate[region] = {
            "raw": raw_summary,
            "oof_calibrated": calibrated_summary,
        }
        raw_brier[region] = raw_brier_values
        calibrated_brier[region] = calibrated_brier_values
        per_image[region] = {
            "raw_nll": raw_nll_values,
            "raw_brier": raw_brier_values,
            "calibrated_nll": calibrated_nll_values,
            "calibrated_brier": calibrated_brier_values,
        }

    per_category = {}
    labels_array = np.asarray([row["label"] for row in validation_rows])
    for label in LABELS:
        indices = np.flatnonzero(labels_array == label)
        per_category[label] = {}
        for region in REGIONS:
            raw_summary, _, _ = region_metrics(
                raw_probabilities,
                raw_scaled_logits,
                targets,
                regions,
                indices,
                region,
            )
            calibrated_summary, _, _ = region_metrics(
                oof_probabilities,
                oof_scaled_logits,
                targets,
                regions,
                indices,
                region,
            )
            per_category[label][region] = {
                "raw": compact_metrics(raw_summary),
                "oof_calibrated": compact_metrics(calibrated_summary),
            }

    groups = [row["similarity_group"] for row in validation_rows]
    bootstrap = paired_group_bootstrap(
        raw_brier, calibrated_brier, groups
    )
    checks = {}
    for region in REGIONS:
        raw = aggregate[region]["raw"]
        calibrated = aggregate[region]["oof_calibrated"]
        checks[f"{region}_nll_improved"] = calibrated["pixel_nll"] < raw["pixel_nll"]
        checks[f"{region}_brier_improved"] = (
            calibrated["pixel_brier"] < raw["pixel_brier"]
        )
        checks[f"{region}_ece_at_most_{ECE_LIMITS[region]:.2f}"] = (
            calibrated["pixel_ece"] <= ECE_LIMITS[region]
        )
        checks[f"{region}_brier_bootstrap_ci95_upper_below_zero"] = (
            bootstrap[region]["ci95_upper"] < 0.0
        )
    checks["oof_hard_masks_bitwise_identical"] = oof_changed_pixels == 0
    checks["final_temperature_hard_masks_bitwise_identical"] = final_changed_pixels == 0
    gate_passed = all(checks.values())

    render_diagram(
        FOREGROUND_DIAGRAM,
        "Lesion-centered foreground-region pixel reliability",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    render_diagram(
        WHOLE_DIAGRAM,
        "Whole-image pixel reliability",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    write_csv(FOLD_RESULTS, fold_results)
    oof_rows = []
    for index, row in enumerate(validation_rows):
        record = {
            "path": row["path"],
            "label": row["label"],
            "similarity_group": row["similarity_group"],
            "fold": int(assignment[index]) + 1,
            "oof_temperature": temperatures[index],
            "lesion_centered_pixels": int(regions[index].sum()),
            "raw_mask_sha256": mask_digest(raw_hard[index]),
            "oof_mask_sha256": mask_digest(oof_hard[index]),
        }
        for region in REGIONS:
            for metric in ("nll", "brier"):
                record[f"raw_{region}_{metric}"] = per_image[region][f"raw_{metric}"][index]
                record[f"oof_calibrated_{region}_{metric}"] = per_image[region][
                    f"calibrated_{metric}"
                ][index]
        oof_rows.append(record)
    write_csv(OOF_RESULTS, oof_rows)

    group_counts = Counter(groups)
    output = {
        "schema_version": 1,
        "status": "SEGMENTER_PIXEL_TEMPERATURE_SCALING_COMPLETE",
        "scope": "Post-hoc BRISC internal-validation calibration; not clinical validation",
        "method": {
            "name": "single positive pixel temperature scaling",
            "fit_objective": "whole-image binary pixel NLL averaged equally over images",
            "cross_fitting": (
                "five held-out folds; exact path-to-fold assignment reused from the "
                "accepted classifier calibration"
            ),
            "fold_assignment_origin": "StratifiedGroupKFold, shuffle=True, seed=42",
            "binning": {
                "count": BINS,
                "type": "fixed equal-width over [0,1]",
                "rationale": (
                    "dense pixel probabilities concentrate near 0 and 1; equal-width bins "
                    "retain visibility of the smaller middle-confidence range"
                ),
                "weighting": "every image has equal total weight within each region",
            },
            "regions": {
                "lesion_centered": (
                    "accepted 128x128 reference mask plus pixels within Euclidean distance "
                    "8 in model space; clipped at image boundaries"
                ),
                "background": "complement of lesion_centered region",
                "whole_image": "all 128x128 pixels",
            },
            "bootstrap": {
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": SEED,
                "unit": "similarity group with all member images retained",
                "fold_stratified": False,
                "same_structure_as_classifier": True,
            },
        },
        "validation_images": len(validation_rows),
        "category_counts": dict(sorted(Counter(labels_array).items())),
        "similarity_groups": len(set(groups)),
        "multi_image_similarity_groups": sum(value > 1 for value in group_counts.values()),
        "largest_similarity_group": max(group_counts.values()),
        "folds": fold_results,
        "final_temperature_all_validation": final_temperature,
        "final_temperature_role": {
            "status": "proposed operational value only",
            "used_for_reported_metrics": False,
            "eligible_for_application_integration": gate_passed,
        },
        "metrics": {
            "aggregate": aggregate,
            "per_category": per_category,
            "brier_paired_similarity_group_bootstrap": bootstrap,
        },
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "ece_limits": ECE_LIMITS,
            "rule": "All checks must pass; there is no metric tie-break.",
        },
        "verification": {
            "segmenter_run": RUN_NAME,
            "checkpoint_sha256": sha256(CHECKPOINT),
            "split_sha256": sha256(split),
            "reused_fold_assignments_sha256": sha256(CLASSIFIER_FOLDS),
            "validation_predictions_sha256": sha256(RUN / "validation_predictions.csv"),
            "source_images_and_masks_verified_against_manifest": source_files_verified,
            "inference_device": str(device),
            "maximum_recomputed_metric_difference": maximum_metric_difference,
            "maximum_recomputed_per_image_difference": maximum_per_image_difference,
            "oof_changed_mask_pixels": oof_changed_pixels,
            "oof_changed_mask_images": oof_changed_images,
            "final_temperature_changed_mask_pixels": final_changed_pixels,
            "final_temperature_changed_mask_images": final_changed_images,
            "test_images_accessed": 0,
            "test_predictions_accessed": 0,
            "model_retrained": False,
            "threshold": 0.5,
            "threshold_changed": False,
            "application_integration_performed": False,
            "script_sha256": sha256(Path(__file__)),
        },
        "artifacts": {
            "foreground_reliability_diagram": str(FOREGROUND_DIAGRAM),
            "whole_image_reliability_diagram": str(WHOLE_DIAGRAM),
            "fold_results": str(FOLD_RESULTS),
            "oof_metrics": str(OOF_RESULTS),
        },
        "limitations": [
            "The segmenter checkpoint was selected using this validation partition before calibration.",
            "The proposed all-validation temperature is not used for reported metrics.",
            "Reference masks define evaluation regions only and are unavailable at inference time.",
            "Similarity groups are not verified patient identifiers.",
            "Patient independence and cross-dataset generalization remain unverified.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(markdown_report(output), encoding="utf-8")

    consolidated = {
        "schema_version": 1,
        "scope": "Internal BRISC confidence calibration findings; not clinical validation",
        "classifier": {
            "status": "CLOSED_RAW_PROBABILITIES_ACCEPTED",
            "finding": (
                "Raw probabilities are used for four-class and binary tumor/no-tumor "
                "outputs. Multiclass and binary temperature scaling produced no "
                "statistically stable improvement."
            ),
            "positive_result": (
                "The unscaled classifier confidence was already well calibrated on "
                "internal validation."
            ),
            "raw_multiclass_ece": 0.0083915474361794,
            "raw_binary_ece": 0.0020682140151788366,
            "further_classifier_calibration_allowed": False,
        },
        "segmenter_pixel_calibration": {
            "status": output["gate"]["status"],
            "result_path": str(OUTPUT_JSON),
            "proposed_temperature": final_temperature,
            "accepted_for_application": gate_passed,
        },
    }
    CALIBRATION_FINDINGS.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=config["mlflow_run_id"]):
        mlflow.log_params(
            {
                "pixel_calibration_method": "scalar_temperature_scaling",
                "pixel_calibration_objective": "image_balanced_whole_image_nll",
                "pixel_calibration_folds": FOLDS,
                "pixel_calibration_bins": BINS,
                "pixel_calibration_roi_radius": DILATION_RADIUS,
            }
        )
        mlflow.log_metrics(
            {
                "pixel_calibration_proposed_temperature": final_temperature,
                "pixel_calibration_foreground_raw_ece": aggregate["lesion_centered"]["raw"]["pixel_ece"],
                "pixel_calibration_foreground_oof_ece": aggregate["lesion_centered"]["oof_calibrated"]["pixel_ece"],
                "pixel_calibration_whole_raw_ece": aggregate["whole_image"]["raw"]["pixel_ece"],
                "pixel_calibration_whole_oof_ece": aggregate["whole_image"]["oof_calibrated"]["pixel_ece"],
                "pixel_calibration_gate_pass": float(gate_passed),
            }
        )
        for artifact in (
            OUTPUT_JSON,
            OUTPUT_MD,
            CALIBRATION_FINDINGS,
            FOREGROUND_DIAGRAM,
            WHOLE_DIAGRAM,
            FOLD_RESULTS,
            OOF_RESULTS,
        ):
            mlflow.log_artifact(str(artifact))

    print("Per-fold temperatures:")
    for item in fold_results:
        print(f"  Fold {item['fold']}: {item['temperature']:.8f}")
    print(f"Proposed all-validation temperature: {final_temperature:.8f}")
    for region in REGIONS:
        raw = aggregate[region]["raw"]
        calibrated = aggregate[region]["oof_calibrated"]
        print(
            f"{region}: NLL {raw['pixel_nll']:.8f} -> {calibrated['pixel_nll']:.8f}; "
            f"Brier {raw['pixel_brier']:.8f} -> {calibrated['pixel_brier']:.8f}; "
            f"ECE {raw['pixel_ece']:.8f} -> {calibrated['pixel_ece']:.8f}; "
            f"MCE {raw['pixel_mce']:.8f} -> {calibrated['pixel_mce']:.8f}"
        )
    print(f"OOF/final changed mask pixels: {oof_changed_pixels}/{final_changed_pixels}")
    print(f"Gate: {output['gate']['status']}")


if __name__ == "__main__":
    main()
