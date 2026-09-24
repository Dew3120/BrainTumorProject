"""Evaluate preregistered 128/256 inference on the frozen v1 segmenter."""

import csv
import json
from contextlib import nullcontext
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
from torch.nn import functional
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e04_multiscale"
ARTIFACTS = EXPERIMENT / "artifacts"
RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
CHECKPOINT = RUN / "top3_checkpoints/epoch_056_macro_dice_0.762662.pt"
LABELS = segmenter.LABELS


def write_json_lf(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def write_text_lf(path, value):
    Path(path).write_bytes(value.replace("\r\n", "\n").encode("utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resize_probability(probability, size=128):
    return functional.interpolate(
        probability,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )


def combine_probabilities(probability_128, probability_256):
    resized = resize_probability(probability_256, probability_128.shape[-1])
    return (probability_128 + resized) / 2, resized


def mask_scores(prediction, reference):
    intersection = int((prediction & reference).sum())
    denominator = int(prediction.sum() + reference.sum())
    union = int((prediction | reference).sum())
    return (
        float(2 * intersection / max(1, denominator)),
        float(intersection / max(1, union)),
    )


def mean_dice(rows, prefix):
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


def paired_group_interval(rows, label):
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
            item["multiscale_dice"] - item["baseline_dice"]
            for key in sampled
            for item in groups[key]
        ]
        values.append(float(np.mean(deltas)))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def gate_checks(delta, gate):
    return {
        "glioma": delta["glioma"] >= gate["glioma_mean_dice_delta_minimum"],
        "meningioma": (
            delta["meningioma"] >= gate["meningioma_mean_dice_delta_minimum"]
        ),
        "pituitary": (
            delta["pituitary"] >= gate["pituitary_mean_dice_delta_minimum"]
        ),
    }


def update_registry(status):
    path = ROOT / "experiments/registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    item = next(
        entry for entry in registry["experiments"] if entry["id"] == "e04_multiscale"
    )
    item["status"] = status
    write_json_lf(path, registry)


def predict(model, inputs, device):
    inputs = inputs.to(device, non_blocking=True)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        logits = model(inputs)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite frozen-model logits")
    return logits.float().sigmoid()


def main():
    config_path = EXPERIMENT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E04 is not in its registered pre-run state")
    if config["scales"] != [128, 256] or config["threshold"] != 0.5:
        raise ValueError("E04 scale or threshold registration mismatch")
    if visual.sha256(CHECKPOINT).lower() != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint SHA256 mismatch")
    split = RUN / "split_snapshot.csv"
    if smoke.digest(split) != config["split_sha256"]:
        raise ValueError("Frozen split SHA256 mismatch")
    validation_rows = segmenter.partition_masked_rows(smoke.read_csv(split))["validation"]
    if any(row["original_split"] != "train" for row in validation_rows):
        raise ValueError("Original test data entered E04")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"]
    )
    model.eval()
    dataset_128 = segmenter.FullPairs(validation_rows, size=128)
    dataset_256 = segmenter.FullPairs(validation_rows, size=256)
    loader_128 = DataLoader(
        dataset_128,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loader_256 = DataLoader(
        dataset_256,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prior = {
        analysis.normalized_path(row["path"]): row
        for row in smoke.read_csv(RUN / "validation_predictions.csv")
    }
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_difference = 0.0
    rows = []
    with torch.inference_mode():
        for batch_128, batch_256 in zip(loader_128, loader_256, strict=True):
            inputs_128, targets_128, _, indices_128 = batch_128
            inputs_256, _, _, indices_256 = batch_256
            if not torch.equal(indices_128, indices_256):
                raise ValueError("Multiscale loaders lost case alignment")
            probability_128 = predict(model, inputs_128, device)
            probability_256 = predict(model, inputs_256, device)
            combined, resized_256 = combine_probabilities(
                probability_128, probability_256
            )
            baseline_masks = probability_128 >= config["threshold"]
            scale_256_masks = resized_256 >= config["threshold"]
            multiscale_masks = combined >= config["threshold"]
            references = targets_128 >= 0.5
            probability_difference = (probability_128 - resized_256).abs().mean(
                dim=(1, 2, 3)
            )
            for position, dataset_index in enumerate(indices_128.tolist()):
                source = validation_rows[dataset_index]
                key = analysis.normalized_path(source["path"])
                reference = references[position, 0].numpy().astype(bool)
                baseline_mask = baseline_masks[position, 0].cpu().numpy().astype(bool)
                scale_256_mask = (
                    scale_256_masks[position, 0].cpu().numpy().astype(bool)
                )
                multiscale_mask = (
                    multiscale_masks[position, 0].cpu().numpy().astype(bool)
                )
                baseline_dice, baseline_iou = mask_scores(baseline_mask, reference)
                scale_256_dice, scale_256_iou = mask_scores(scale_256_mask, reference)
                multiscale_dice, multiscale_iou = mask_scores(
                    multiscale_mask, reference
                )
                stored = prior[key]
                difference = max(
                    abs(baseline_dice - float(stored["dice"])),
                    abs(baseline_iou - float(stored["iou"])),
                )
                maximum_difference = max(maximum_difference, difference)
                if difference > tolerance:
                    raise ValueError(f"Frozen metric mismatch: {source['path']}")
                rows.append(
                    {
                        "path": source["path"],
                        "filename": source["filename"],
                        "category": source["label"],
                        "similarity_group": source["similarity_group"],
                        "baseline_dice": baseline_dice,
                        "baseline_iou": baseline_iou,
                        "scale_256_dice": scale_256_dice,
                        "scale_256_iou": scale_256_iou,
                        "multiscale_dice": multiscale_dice,
                        "multiscale_dice_delta": multiscale_dice - baseline_dice,
                        "multiscale_iou": multiscale_iou,
                        "multiscale_iou_delta": multiscale_iou - baseline_iou,
                        "baseline_pixels": int(baseline_mask.sum()),
                        "scale_256_pixels_at_128": int(scale_256_mask.sum()),
                        "multiscale_pixels": int(multiscale_mask.sum()),
                        "changed_pixels": int(
                            np.count_nonzero(multiscale_mask != baseline_mask)
                        ),
                        "mask_changed": not np.array_equal(
                            multiscale_mask, baseline_mask
                        ),
                        "mean_absolute_probability_difference": float(
                            probability_difference[position].item()
                        ),
                    }
                )
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E04 did not evaluate every validation case exactly once")

    baseline = mean_dice(rows, "baseline")
    scale_256 = mean_dice(rows, "scale_256")
    candidate = mean_dice(rows, "multiscale")
    delta = {key: candidate[key] - baseline[key] for key in baseline}
    intervals = {label: paired_group_interval(rows, label) for label in LABELS}
    checks = gate_checks(delta, config["gate"])
    passed = all(checks.values())
    status = "PASS" if passed else "FAIL"
    ARTIFACTS.mkdir(exist_ok=True)
    write_csv(ARTIFACTS / "per_case_metrics.csv", rows)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, label in zip(axes, LABELS, strict=True):
        values = [
            row["multiscale_dice_delta"]
            for row in rows
            if row["category"] == label
        ]
        axis.hist(values, bins=30, color="#2f77b4", alpha=0.86)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(f"{label}: mean Δ {delta[label]:+.4f}")
        axis.set_xlabel("Multiscale Dice − baseline Dice")
        axis.set_ylabel("Cases")
    figure.savefig(ARTIFACTS / "dice_delta_distribution.png", dpi=160)
    plt.close(figure)

    result = {
        "schema_version": 1,
        "experiment_id": "e04_multiscale",
        "status": status,
        "scope": "Frozen BRISC internal validation; not clinical validation",
        "device": str(device),
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "threshold_changed": False,
        "model_changed": False,
        "maximum_stored_metric_difference": maximum_difference,
        "baseline_mean_dice": baseline,
        "scale_256_mean_dice_descriptive": scale_256,
        "multiscale_mean_dice": candidate,
        "mean_dice_delta": delta,
        "paired_group_bootstrap_ci95_delta": intervals,
        "gate_checks": checks,
        "gate_passed": passed,
        "gate": config["gate"],
        "changed_cases": sum(row["mask_changed"] for row in rows),
        "changed_pixels": sum(row["changed_pixels"] for row in rows),
        "improved_cases": sum(
            row["multiscale_dice_delta"] > tolerance for row in rows
        ),
        "worsened_cases": sum(
            row["multiscale_dice_delta"] < -tolerance for row in rows
        ),
        "mean_absolute_probability_difference": float(
            np.mean([row["mean_absolute_probability_difference"] for row in rows])
        ),
        "comparison_tolerance": tolerance,
        "limitations": [
            "Internal validation was reused for development.",
            "Patient independence is unverified.",
            "The model was trained at 128 pixels; 256-pixel inference is an out-of-training-scale intervention.",
        ],
    }
    write_json_lf(EXPERIMENT / "result.json", result)
    decision = "ACCEPTED" if passed else "REJECTED"
    decision_text = "\n".join(
        [
            "# Two-scale inference decision",
            "",
            f"**Decision:** {decision}",
            "",
            (
                "The 128/256 probability ensemble passed every preregistered category gate."
                if passed
                else "The 128/256 probability ensemble failed the preregistered conjunctive gate and is closed without relaxing it."
            ),
            "",
            f"Glioma Dice delta: `{delta['glioma']:+.4f}` (required `+0.0200`).",
            f"Meningioma Dice delta: `{delta['meningioma']:+.4f}` (minimum `-0.0100`).",
            f"Pituitary Dice delta: `{delta['pituitary']:+.4f}` (minimum `-0.0100`).",
            "",
            "No model weight or probability threshold changed, and no sealed-test image was accessed.",
            "",
        ]
    )
    write_text_lf(EXPERIMENT / "decision.md", decision_text)
    config["status"] = "COMPLETE"
    write_json_lf(config_path, config)
    update_registry("COMPLETE")
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)

    summary = [
        "# Two-scale inference result",
        "",
        "| Category | Baseline Dice | 256-only Dice | Ensemble Dice | Delta | 95% group-bootstrap CI | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        interval = intervals[label]
        summary.append(
            f"| {label} | {baseline[label]:.4f} | {scale_256[label]:.4f} | "
            f"{candidate[label]:.4f} | {delta[label]:+.4f} | "
            f"({interval[0]:+.4f}, {interval[1]:+.4f}) | "
            f"{'pass' if checks[label] else 'fail'} |"
        )
    summary.extend(["", f"**Conjunctive decision: {status}.**", ""])
    write_text_lf(ARTIFACTS / "summary.md", "\n".join(summary))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
