"""Shared frozen-data utilities for advanced E08 calibration experiments."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = (
    ROOT
    / "runs/classifier/20260921-170607-243133-baseline"
    / "classifier_calibration_folds.csv"
)
REGIONS = ("lesion_centered", "background", "whole_image")
LABELS = ("glioma", "meningioma", "pituitary")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


def load_cache(expected_manifest_sha256):
    run = Path(
        (ROOT / "runs/segmentation_v2_ensemble/latest.txt")
        .read_text(encoding="utf-8")
        .strip()
    ).resolve()
    manifest = run / "manifest.json"
    if sha256(manifest) != expected_manifest_sha256:
        raise ValueError("E08 calibration-cache manifest hash mismatch")
    order = read_csv(run / "validation_order.csv")
    with np.load(run / "ensemble_probabilities_targets.npz") as archive:
        probabilities = archive["probabilities"].astype(np.float64)
        targets = archive["targets"].astype(bool)
        labels = archive["labels"].astype(np.int64)
    if probabilities.shape != (680, 128, 128) or targets.shape != probabilities.shape:
        raise ValueError("E08 calibration-cache array shape mismatch")
    if len(order) != 680 or len({row["path"] for row in order}) != 680:
        raise ValueError("E08 calibration-cache order mismatch")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("E08 calibration cache contains invalid probabilities")
    return run, order, probabilities, targets, labels


def fold_assignment(rows):
    published = {row["path"]: row for row in read_csv(FOLDS_PATH)}
    assignment = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        source = published.get(row["path"])
        if source is None or source["similarity_group"] != row["similarity_group"]:
            raise ValueError(f"Calibration fold metadata mismatch: {row['path']}")
        assignment[index] = int(source["fold"]) - 1
    if set(assignment) != set(range(5)):
        raise ValueError("Calibration assignment does not contain five folds")
    group_folds = {}
    for row, fold in zip(rows, assignment, strict=True):
        group_folds.setdefault(row["similarity_group"], set()).add(int(fold))
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A similarity group crosses calibration folds")
    return assignment


def lesion_regions(targets, radius=8.0):
    regions = np.empty_like(targets, dtype=bool)
    for index, target in enumerate(targets):
        if not target.any() or target.all():
            raise ValueError("Invalid reference mask for lesion-centered region")
        regions[index] = distance_transform_edt(~target) <= radius
    return regions


def region_mask(regions, index, name):
    if name == "lesion_centered":
        return regions[index]
    if name == "background":
        return ~regions[index]
    if name == "whole_image":
        return np.ones_like(regions[index], dtype=bool)
    raise ValueError(f"Unknown calibration region: {name}")


def equal_frequency_binary(probabilities, targets, bins=15):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    order = np.argsort(probabilities, kind="stable")
    groups = np.array_split(order, bins)
    output = []
    ece = 0.0
    mce = 0.0
    for index, members in enumerate(groups, start=1):
        confidence = float(probabilities[members].mean())
        observed = float(targets[members].mean())
        gap = abs(confidence - observed)
        weight = len(members) / len(order)
        ece += weight * gap
        mce = max(mce, gap)
        output.append(
            {
                "bin": index,
                "count": len(members),
                "weight": weight,
                "minimum_probability": float(probabilities[members].min()),
                "maximum_probability": float(probabilities[members].max()),
                "mean_probability": confidence,
                "observed_frequency": observed,
                "absolute_gap": gap,
            }
        )
    return float(ece), float(mce), output


def region_metrics(probabilities, logits, targets, regions, indices, name):
    indices = np.asarray(indices, dtype=np.int64)
    per_nll = np.empty(len(indices), dtype=np.float64)
    per_brier = np.empty(len(indices), dtype=np.float64)
    pooled_probability = []
    pooled_target = []
    for output_index, image_index in enumerate(indices):
        mask = region_mask(regions, image_index, name)
        probability = probabilities[image_index][mask]
        logit = logits[image_index][mask]
        target = targets[image_index][mask].astype(np.float64)
        per_nll[output_index] = np.mean(np.logaddexp(0, logit) - target * logit)
        per_brier[output_index] = np.mean(np.square(probability - target))
        pooled_probability.append(probability)
        pooled_target.append(target)
    pooled_probability = np.concatenate(pooled_probability)
    pooled_target = np.concatenate(pooled_target)
    ece, mce, bins = equal_frequency_binary(pooled_probability, pooled_target)
    return (
        {
            "image_count": len(indices),
            "pixel_count": len(pooled_probability),
            "pixel_nll": float(per_nll.mean()),
            "pixel_brier": float(per_brier.mean()),
            "pixel_ece": ece,
            "pixel_mce": mce,
            "binning": "15 equal-frequency pooled-pixel bins",
            "bins": bins,
        },
        per_nll,
        per_brier,
    )


def paired_group_bootstrap(raw_values, calibrated_values, groups, repetitions=2000):
    groups = np.asarray(groups)
    unique = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(42)
    values = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        values[iteration] = np.mean(calibrated_values[indices] - raw_values[indices])
    return {
        "mean_change": float(np.mean(calibrated_values - raw_values)),
        "ci95_lower": float(np.percentile(values, 2.5)),
        "ci95_upper": float(np.percentile(values, 97.5)),
    }


def render_binary_diagram(path, title, raw, calibrated):
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    for summary, label, marker in (
        (raw, "Raw", "o"),
        (calibrated, "OOF calibrated", "s"),
    ):
        axis.plot(
            [item["mean_probability"] for item in summary["bins"]],
            [item["observed_frequency"] for item in summary["bins"]],
            marker=marker,
            label=f"{label} (ECE={summary['pixel_ece']:.4f})",
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean probability", ylabel="Observed foreground frequency", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
