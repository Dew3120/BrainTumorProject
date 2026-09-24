"""Evaluate preregistered largest-component filtering on frozen v1 masks."""

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
EXPERIMENT = ROOT / "experiments/e01_lcc_filter"
ARTIFACTS = EXPERIMENT / "artifacts"
CONFIG = EXPERIMENT / "config.json"
RESULT = EXPERIMENT / "result.json"
DECISION = EXPERIMENT / "decision.md"
RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
CHECKPOINT = RUN / "top3_checkpoints/epoch_056_macro_dice_0.762662.pt"
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)
LABELS = segmenter.LABELS


def keep_largest_component(mask):
    mask = np.asarray(mask, dtype=bool)
    labeled, count = ndimage.label(mask, structure=STRUCTURE_8)
    if count <= 1:
        return mask.copy(), 0, 0
    sizes = np.bincount(labeled.ravel())[1:]
    largest_size = int(sizes.max())
    candidates = np.flatnonzero(sizes == largest_size) + 1

    def top_left(component):
        point = np.argwhere(labeled == component).min(axis=0)
        return int(point[0]), int(point[1])

    selected = min(candidates, key=top_left)
    filtered = labeled == selected
    return filtered, int(count - 1), int(mask.sum() - filtered.sum())


def mask_scores(prediction, reference):
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    intersection = int((prediction & reference).sum())
    denominator = int(prediction.sum() + reference.sum())
    union = int((prediction | reference).sum())
    dice = 2 * intersection / max(1, denominator)
    iou = intersection / max(1, union)
    return float(dice), float(iou)


def category_means(rows, prefix):
    result = {}
    for label in LABELS:
        values = [row[f"{prefix}_dice"] for row in rows if row["category"] == label]
        result[label] = float(np.mean(values))
    result["overall"] = float(np.mean([row[f"{prefix}_dice"] for row in rows]))
    result["macro_type"] = float(np.mean([result[label] for label in LABELS]))
    return result


def paired_group_bootstrap(rows, label):
    members = [row for row in rows if row["category"] == label]
    groups = {}
    for row in members:
        groups.setdefault(row["similarity_group"], []).append(row)
    keys = sorted(groups)
    generator = np.random.default_rng(42)
    estimates = []
    for _ in range(2000):
        sampled = generator.choice(keys, size=len(keys), replace=True)
        deltas = [
            item["filtered_dice"] - item["baseline_dice"]
            for key in sampled
            for item in groups[key]
        ]
        estimates.append(float(np.mean(deltas)))
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_registry(status):
    path = ROOT / "experiments/registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    item = next(
        entry for entry in registry["experiments"] if entry["id"] == "e01_lcc_filter"
    )
    item["status"] = status
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E01 is not in its registered pre-run state")
    if visual.sha256(CHECKPOINT).lower() != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint SHA256 mismatch")
    split = RUN / "split_snapshot.csv"
    if smoke.digest(split) != config["split_sha256"]:
        raise ValueError("Frozen split SHA256 mismatch")
    parts = segmenter.partition_masked_rows(smoke.read_csv(split))
    validation_rows = parts["validation"]
    if any(row["original_split"] != "train" for row in validation_rows):
        raise ValueError("Original test data entered E01")
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
    source = {
        analysis.normalized_path(row["path"]): row for row in validation_rows
    }
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    rows = []
    maximum_difference = 0.0
    for record in records:
        key = analysis.normalized_path(record["path"])
        old = prior[key]
        difference = max(
            abs(record["dice"] - float(old["dice"])),
            abs(record["iou"] - float(old["iou"])),
        )
        maximum_difference = max(maximum_difference, difference)
        if difference > tolerance:
            raise ValueError(f"Frozen per-case metric mismatch: {record['path']}")
        filtered, removed_components, removed_pixels = keep_largest_component(
            predictions[key]
        )
        filtered_dice, filtered_iou = mask_scores(filtered, references[key])
        rows.append(
            {
                "path": record["path"],
                "filename": record["filename"],
                "category": record["label"],
                "similarity_group": source[key]["similarity_group"],
                "baseline_dice": record["dice"],
                "filtered_dice": filtered_dice,
                "dice_delta": filtered_dice - record["dice"],
                "baseline_iou": record["iou"],
                "filtered_iou": filtered_iou,
                "iou_delta": filtered_iou - record["iou"],
                "baseline_pixels": int(predictions[key].sum()),
                "filtered_pixels": int(filtered.sum()),
                "removed_components": removed_components,
                "removed_pixels": removed_pixels,
                "mask_changed": not np.array_equal(predictions[key], filtered),
            }
        )
    if len(rows) != 680:
        raise ValueError("E01 did not evaluate all 680 validation masks")
    baseline = category_means(rows, "baseline")
    filtered = category_means(rows, "filtered")
    deltas = {key: filtered[key] - baseline[key] for key in baseline}
    intervals = {
        label: paired_group_bootstrap(rows, label) for label in (*LABELS,)
    }
    gate_checks = {
        "glioma": deltas["glioma"] >= config["gate"]["glioma_mean_dice_delta_minimum"],
        "meningioma": deltas["meningioma"]
        >= config["gate"]["meningioma_mean_dice_delta_minimum"],
        "pituitary": deltas["pituitary"]
        >= config["gate"]["pituitary_mean_dice_delta_minimum"],
    }
    passed = all(gate_checks.values())
    status = "PASS" if passed else "FAIL"
    ARTIFACTS.mkdir(exist_ok=True)
    per_case_path = ARTIFACTS / "per_case_metrics.csv"
    write_csv(per_case_path, rows)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, label in zip(axes, LABELS, strict=True):
        values = [row["dice_delta"] for row in rows if row["category"] == label]
        axis.hist(values, bins=30, color="#2f77b4", alpha=0.86)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(f"{label}: mean Δ {deltas[label]:+.4f}")
        axis.set_xlabel("Filtered Dice − baseline Dice")
        axis.set_ylabel("Cases")
    figure.savefig(ARTIFACTS / "dice_delta_distribution.png", dpi=160)
    plt.close(figure)
    result = {
        "schema_version": 1,
        "experiment_id": "e01_lcc_filter",
        "status": status,
        "scope": "Frozen BRISC internal validation; not clinical validation",
        "device": str(device),
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "threshold_changed": False,
        "model_changed": False,
        "maximum_stored_metric_difference": maximum_difference,
        "baseline_mean_dice": baseline,
        "filtered_mean_dice": filtered,
        "mean_dice_delta": deltas,
        "paired_group_bootstrap_ci95_delta": intervals,
        "mask_change_summary": {
            "changed_cases": sum(row["mask_changed"] for row in rows),
            "unchanged_cases": sum(not row["mask_changed"] for row in rows),
            "improved_cases": sum(row["dice_delta"] > tolerance for row in rows),
            "worsened_cases": sum(row["dice_delta"] < -tolerance for row in rows),
            "equal_cases": sum(abs(row["dice_delta"]) <= tolerance for row in rows),
            "components_removed": sum(row["removed_components"] for row in rows),
            "pixels_removed": sum(row["removed_pixels"] for row in rows),
        },
        "gate": config["gate"],
        "gate_checks": gate_checks,
        "gate_passed": passed,
        "limitations": [
            "Internal validation was reused for development.",
            "Patient independence is unverified.",
            "Filtering can delete true multifocal or satellite tumor regions.",
        ],
    }
    RESULT.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    decision = "ACCEPTED" if passed else "REJECTED"
    DECISION.write_text(
        "\n".join(
            [
                "# Decision",
                "",
                f"**Decision:** {decision}",
                "",
                (
                    f"Largest-component filtering changed {result['mask_change_summary']['changed_cases']} of 680 masks. "
                    f"Glioma mean Dice changed by {deltas['glioma']:+.4f}; meningioma by {deltas['meningioma']:+.4f}; "
                    f"pituitary by {deltas['pituitary']:+.4f}."
                ),
                "",
                (
                    "The local gate passed. This operation remains a development candidate; it does not independently pass the parent H1 gate or establish generalization."
                    if passed
                    else "The conjunctive local gate failed and will not be relaxed. This filtering rule is closed as a rejected intervention."
                ),
                "",
                "No model weight, probability threshold, or sealed-test case was changed or accessed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config["status"] = "COMPLETE"
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    update_registry("COMPLETE")
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    summary = [
        "# Largest-component filtering result",
        "",
        "| Category | Baseline Dice | Filtered Dice | Delta | 95% group-bootstrap CI | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        summary.append(
            f"| {label} | {baseline[label]:.4f} | {filtered[label]:.4f} | "
            f"{deltas[label]:+.4f} | ({intervals[label][0]:+.4f}, {intervals[label][1]:+.4f}) | "
            f"{'pass' if gate_checks[label] else 'fail'} |"
        )
    summary.extend(
        [
            "",
            f"**Conjunctive decision: {status}.**",
            "",
            "Filtering removes disconnected components from a hard mask. It cannot recover a missed lesion and can remove real multifocal disease. Results are internal development evidence only.",
        ]
    )
    (ARTIFACTS / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
