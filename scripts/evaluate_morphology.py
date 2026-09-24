"""Evaluate preregistered closing-then-opening variants on frozen v1 masks."""

import csv
import json
from pathlib import Path

import analyze_glioma_failures as analysis
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import segmentation_visual_audit as visual
import torch
import train_segmenter as segmenter
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e02_morphology"
ARTIFACTS = EXPERIMENT / "artifacts"
RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
CHECKPOINT = RUN / "top3_checkpoints/epoch_056_macro_dice_0.762662.pt"
LABELS = segmenter.LABELS


def closing_then_opening(mask, kernel_size):
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    closed = ndimage.binary_closing(mask, structure=structure, border_value=0)
    return ndimage.binary_opening(closed, structure=structure, border_value=0)


def mask_scores(prediction, reference):
    intersection = int((prediction & reference).sum())
    denominator = int(prediction.sum() + reference.sum())
    union = int((prediction | reference).sum())
    return (
        float(2 * intersection / max(1, denominator)),
        float(intersection / max(1, union)),
    )


def means(rows, prefix):
    values = {
        label: float(
            np.mean(
                [row[f"{prefix}_dice"] for row in rows if row["category"] == label]
            )
        )
        for label in LABELS
    }
    values["overall"] = float(np.mean([row[f"{prefix}_dice"] for row in rows]))
    values["macro_type"] = float(np.mean([values[label] for label in LABELS]))
    return values


def paired_group_interval(rows, label, prefix):
    members = [row for row in rows if row["category"] == label]
    groups = {}
    for row in members:
        groups.setdefault(row["similarity_group"], []).append(row)
    keys = sorted(groups)
    generator = np.random.default_rng(42)
    values = []
    for _ in range(2000):
        sampled = generator.choice(keys, size=len(keys), replace=True)
        deltas = [
            item[f"{prefix}_dice"] - item["baseline_dice"]
            for key in sampled
            for item in groups[key]
        ]
        values.append(float(np.mean(deltas)))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_registry(status):
    path = ROOT / "experiments/registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    item = next(
        entry for entry in registry["experiments"] if entry["id"] == "e02_morphology"
    )
    item["status"] = status
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def main():
    config_path = EXPERIMENT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E02 is not in its registered pre-run state")
    if visual.sha256(CHECKPOINT).lower() != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint SHA256 mismatch")
    split = RUN / "split_snapshot.csv"
    if smoke.digest(split) != config["split_sha256"]:
        raise ValueError("Frozen split SHA256 mismatch")
    validation_rows = segmenter.partition_masked_rows(smoke.read_csv(split))["validation"]
    if any(row["original_split"] != "train" for row in validation_rows):
        raise ValueError("Original test data entered E02")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"]
    )
    _, records, predictions, references, _ = analysis.infer_all(
        model, validation_rows, device
    )
    prior = {
        analysis.normalized_path(row["path"]): row
        for row in smoke.read_csv(RUN / "validation_predictions.csv")
    }
    sources = {
        analysis.normalized_path(row["path"]): row for row in validation_rows
    }
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_difference = 0.0
    rows = []
    for record in records:
        key = analysis.normalized_path(record["path"])
        old = prior[key]
        difference = max(
            abs(record["dice"] - float(old["dice"])),
            abs(record["iou"] - float(old["iou"])),
        )
        maximum_difference = max(maximum_difference, difference)
        if difference > tolerance:
            raise ValueError(f"Frozen metric mismatch: {record['path']}")
        output = {
            "path": record["path"],
            "filename": record["filename"],
            "category": record["label"],
            "similarity_group": sources[key]["similarity_group"],
            "baseline_dice": record["dice"],
            "baseline_iou": record["iou"],
        }
        for kernel_size in (3, 5):
            prefix = f"kernel_{kernel_size}"
            transformed = closing_then_opening(predictions[key], kernel_size)
            dice, iou = mask_scores(transformed, references[key])
            output.update(
                {
                    f"{prefix}_dice": dice,
                    f"{prefix}_dice_delta": dice - record["dice"],
                    f"{prefix}_iou": iou,
                    f"{prefix}_iou_delta": iou - record["iou"],
                    f"{prefix}_pixels": int(transformed.sum()),
                    f"{prefix}_changed_pixels": int(
                        np.count_nonzero(transformed != predictions[key])
                    ),
                    f"{prefix}_mask_changed": not np.array_equal(
                        transformed, predictions[key]
                    ),
                }
            )
        rows.append(output)
    if len(rows) != 680:
        raise ValueError("E02 did not evaluate all validation cases")
    baseline = means(rows, "baseline")
    variants = {}
    passing = []
    for kernel_size in (3, 5):
        prefix = f"kernel_{kernel_size}"
        candidate = means(rows, prefix)
        delta = {key: candidate[key] - baseline[key] for key in baseline}
        intervals = {
            label: paired_group_interval(rows, label, prefix) for label in LABELS
        }
        checks = {
            "glioma": delta["glioma"]
            >= config["gate"]["glioma_mean_dice_delta_minimum"],
            "meningioma": delta["meningioma"]
            >= config["gate"]["meningioma_mean_dice_delta_minimum"],
            "pituitary": delta["pituitary"]
            >= config["gate"]["pituitary_mean_dice_delta_minimum"],
        }
        passed = all(checks.values())
        variants[prefix] = {
            "kernel_size": kernel_size,
            "mean_dice": candidate,
            "mean_dice_delta": delta,
            "paired_group_bootstrap_ci95_delta": intervals,
            "gate_checks": checks,
            "gate_passed": passed,
            "changed_cases": sum(row[f"{prefix}_mask_changed"] for row in rows),
            "changed_pixels": sum(row[f"{prefix}_changed_pixels"] for row in rows),
            "improved_cases": sum(
                row[f"{prefix}_dice_delta"] > tolerance for row in rows
            ),
            "worsened_cases": sum(
                row[f"{prefix}_dice_delta"] < -tolerance for row in rows
            ),
        }
        if passed:
            passing.append(prefix)
    selected = (
        min(
            passing,
            key=lambda prefix: (
                -variants[prefix]["mean_dice"]["glioma"],
                variants[prefix]["kernel_size"],
            ),
        )
        if passing
        else None
    )
    passed = selected is not None
    status = "PASS" if passed else "FAIL"
    ARTIFACTS.mkdir(exist_ok=True)
    write_csv(ARTIFACTS / "per_case_metrics.csv", rows)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row_index, kernel_size in enumerate((3, 5)):
        prefix = f"kernel_{kernel_size}"
        for axis, label in zip(axes[row_index], LABELS, strict=True):
            values = [
                row[f"{prefix}_dice_delta"]
                for row in rows
                if row["category"] == label
            ]
            axis.hist(values, bins=30, color="#2f77b4", alpha=0.86)
            axis.axvline(0, color="black", linewidth=1)
            axis.set_title(
                f"{kernel_size}×{kernel_size} {label}: mean Δ "
                f"{variants[prefix]['mean_dice_delta'][label]:+.4f}"
            )
            axis.set_xlabel("Morphology Dice − baseline Dice")
            axis.set_ylabel("Cases")
    figure.savefig(ARTIFACTS / "dice_delta_distribution.png", dpi=160)
    plt.close(figure)
    result = {
        "schema_version": 1,
        "experiment_id": "e02_morphology",
        "status": status,
        "scope": "Frozen BRISC internal validation; not clinical validation",
        "device": str(device),
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "threshold_changed": False,
        "model_changed": False,
        "maximum_stored_metric_difference": maximum_difference,
        "baseline_mean_dice": baseline,
        "variants": variants,
        "selected_variant": selected,
        "gate_passed": passed,
        "gate": config["gate"],
        "comparison_tolerance": tolerance,
        "limitations": [
            "Internal validation was reused for development.",
            "Patient independence is unverified.",
            "Morphology can erase small true regions or bridge distinct regions.",
        ],
    }
    (EXPERIMENT / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    decision = "ACCEPTED" if passed else "REJECTED"
    lines = ["# Decision", "", f"**Decision:** {decision}", ""]
    for prefix in ("kernel_3", "kernel_5"):
        delta = variants[prefix]["mean_dice_delta"]
        lines.append(
            f"{prefix} changed glioma/meningioma/pituitary Dice by "
            f"{delta['glioma']:+.4f}/{delta['meningioma']:+.4f}/{delta['pituitary']:+.4f}; "
            f"gate {'passed' if variants[prefix]['gate_passed'] else 'failed'}."
        )
    lines.extend(
        [
            "",
            (
                f"Selected `{selected}` by the preregistered ranking."
                if selected
                else "Neither fixed kernel passed. The morphology intervention is closed and its gate is not relaxed."
            ),
            "",
            "No model weight, probability threshold, or sealed-test case was changed or accessed.",
        ]
    )
    (EXPERIMENT / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    config["status"] = "COMPLETE"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    update_registry("COMPLETE")
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    summary = [
        "# Morphology result",
        "",
        "| Variant | Category | Baseline | Candidate | Delta | 95% group-bootstrap CI | Gate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for prefix in ("kernel_3", "kernel_5"):
        for label in LABELS:
            item = variants[prefix]
            interval = item["paired_group_bootstrap_ci95_delta"][label]
            summary.append(
                f"| {prefix} | {label} | {baseline[label]:.4f} | "
                f"{item['mean_dice'][label]:.4f} | {item['mean_dice_delta'][label]:+.4f} | "
                f"({interval[0]:+.4f}, {interval[1]:+.4f}) | "
                f"{'pass' if item['gate_checks'][label] else 'fail'} |"
            )
    summary.extend(["", f"**Conjunctive decision: {status}.**"])
    (ARTIFACTS / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
