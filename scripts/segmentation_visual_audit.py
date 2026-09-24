"""Recompute validation predictions and render deterministic segmentation audit panels."""

import argparse
import csv
import hashlib
import json
import textwrap
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import torch
import train_segmenter as train
from matplotlib import font_manager
from monai.losses import DiceLoss
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "audit/segmentation_visual_review_guide.md"
LABELS = train.LABELS


def normalized_path(value):
    return value.replace("\\", "/").casefold()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def select_cases(records, worst_per_category=6, zero_count=6, best_per_category=6):
    groups = {}
    for label in LABELS:
        members = [row for row in records if row["label"] == label]
        groups[f"worst_{label}"] = sorted(
            members, key=lambda row: (row["dice"], normalized_path(row["path"]))
        )[:worst_per_category]
        groups[f"best_{label}"] = sorted(
            members, key=lambda row: (-row["dice"], normalized_path(row["path"]))
        )[:best_per_category]
    zero = [row for row in records if row["dice"] == 0]
    groups["zero_dice"] = sorted(zero, key=lambda row: normalized_path(row["path"]))[
        :zero_count
    ]
    if any(
        len(group) != worst_per_category
        for key, group in groups.items()
        if key.startswith("worst_")
    ):
        raise ValueError("Insufficient cases for a worst-category row")
    if any(
        len(group) != best_per_category
        for key, group in groups.items()
        if key.startswith("best_")
    ):
        raise ValueError("Insufficient cases for a best-category row")
    if len(groups["zero_dice"]) != zero_count:
        raise ValueError("Insufficient zero-Dice cases")
    return groups


def overlay(image, mask, color, alpha=0.58):
    base = image.convert("RGB")
    mask_array = np.asarray(mask, dtype=bool)
    result = np.asarray(base, dtype=np.float32).copy()
    color_array = np.asarray(color, dtype=np.float32)
    result[mask_array] = result[mask_array] * (1 - alpha) + color_array * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def tensor_image(value):
    array = np.clip(value.squeeze().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def tensor_mask(value):
    array = value.squeeze().cpu().numpy() >= 0.5
    return Image.fromarray(array)


def load_font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=size)


def render_panel(destination, title, group_rows, row_lookup, prediction_lookup):
    columns = 6
    panel_size = 144
    tile_width = panel_size * 3 + 18
    tile_height = 218
    title_height = 48
    group_height = 32
    width = columns * tile_width
    height = title_height + len(group_rows) * (group_height + tile_height)
    canvas = Image.new("RGB", (width, height), "#202124")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    group_font = load_font(18)
    text_font = load_font(13)
    small_font = load_font(11)
    draw.text((12, 8), title, fill="white", font=title_font)
    for group_index, (group_name, records) in enumerate(group_rows):
        group_y = title_height + group_index * (group_height + tile_height)
        draw.rectangle((0, group_y, width, group_y + group_height), fill="#34373b")
        draw.text((10, group_y + 5), group_name, fill="#f1f3f4", font=group_font)
        for column, record in enumerate(records):
            row = row_lookup[record["path"]]
            dataset = overfit.SizedPairs([row], 128)
            image_tensor, reference_tensor = dataset[0]
            image = tensor_image(image_tensor)
            reference = tensor_mask(reference_tensor)
            prediction = Image.fromarray(prediction_lookup[record["path"]])
            panels = (
                image.convert("RGB"),
                overlay(image, reference, (235, 64, 52)),
                overlay(image, prediction, (0, 210, 220)),
            )
            tile_x = column * tile_width
            tile_y = group_y + group_height
            draw.rectangle(
                (tile_x, tile_y, tile_x + tile_width - 2, tile_y + tile_height - 2),
                outline="#5f6368",
                width=1,
            )
            filename_lines = textwrap.wrap(record["filename"], width=48) or [
                record["filename"]
            ]
            draw.text(
                (tile_x + 5, tile_y + 3),
                filename_lines[0],
                fill="white",
                font=text_font,
            )
            draw.text(
                (tile_x + 5, tile_y + 21),
                f"{record['label']}  Dice {record['dice']:.4f}  IoU {record['iou']:.4f}",
                fill="#d7dadc",
                font=small_font,
            )
            image_y = tile_y + 42
            for panel_index, panel in enumerate(panels):
                resized = panel.resize(
                    (panel_size, panel_size),
                    Image.Resampling.BILINEAR
                    if panel_index == 0
                    else Image.Resampling.NEAREST,
                )
                canvas.paste(
                    resized, (tile_x + 3 + panel_index * (panel_size + 3), image_y)
                )
            labels = ("Input", "Reference (red)", "Prediction (cyan)")
            for panel_index, label in enumerate(labels):
                draw.text(
                    (
                        tile_x + 4 + panel_index * (panel_size + 3),
                        image_y + panel_size + 3,
                    ),
                    label,
                    fill="#d7dadc",
                    font=small_font,
                )
    canvas.save(destination)


def inference(model, rows, device, threshold):
    dataset = train.FullPairs(rows)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    dice_loss = DiceLoss(sigmoid=True)
    model.eval()
    records = []
    predictions = {}
    dice_values = []
    iou_values = []
    labels = []
    total_loss = 0.0
    with torch.inference_mode():
        for images, targets, label_indices, indices in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(images)
            loss = train.segmentation_loss(logits, targets, dice_loss)
            probabilities = logits.float().sigmoid()
            predicted = probabilities >= threshold
            reference = targets >= 0.5
            intersection = (predicted & reference).sum(dim=(1, 2, 3)).float()
            predicted_area = predicted.sum(dim=(1, 2, 3)).float()
            reference_area = reference.sum(dim=(1, 2, 3)).float()
            dice = 2 * intersection / (predicted_area + reference_area).clamp_min(1)
            iou = intersection / (
                predicted_area + reference_area - intersection
            ).clamp_min(1)
            total_loss += loss.item() * len(images)
            dice_values.extend(dice.cpu().tolist())
            iou_values.extend(iou.cpu().tolist())
            labels.extend(label_indices.tolist())
            for position, dataset_index in enumerate(indices.tolist()):
                row = rows[dataset_index]
                predictions[row["path"]] = predicted[position, 0].cpu().numpy()
                records.append(
                    {
                        "path": row["path"],
                        "filename": row["filename"],
                        "label": row["label"],
                        "dice": float(dice[position].item()),
                        "iou": float(iou[position].item()),
                        "reference_pixels": int(reference_area[position].item()),
                        "predicted_pixels": int(predicted_area[position].item()),
                    }
                )
    metrics = train.summarize_scores(
        np.asarray(dice_values), np.asarray(iou_values), np.asarray(labels)
    )
    metrics["val_loss"] = total_loss / len(dataset)
    return metrics, records, predictions


def write_selection(path, group_rows):
    selected = []
    seen = set()
    for panel, groups in group_rows.items():
        for group, records in groups:
            for rank, record in enumerate(records, start=1):
                key = normalized_path(record["path"])
                selected.append(
                    {
                        "panel": panel,
                        "group": group,
                        "rank": rank,
                        **record,
                        "duplicate_of_earlier_tile": key in seen,
                    }
                )
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--worst-per-category", type=int, default=6)
    parser.add_argument("--zero-dice", type=int, default=6)
    parser.add_argument("--best-per-category", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.threshold != 0.5:
        raise ValueError("The accepted segmentation threshold is fixed at 0.5")
    run = args.run.resolve()
    checkpoint = args.checkpoint.resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    expected_checkpoint = result["top3"][0]
    if checkpoint.name != expected_checkpoint["checkpoint"]:
        raise ValueError("Checkpoint is not the accepted rank-1 epoch")
    if sha256(checkpoint) != args.checkpoint_sha256.upper():
        raise ValueError("Checkpoint SHA256 mismatch")
    split = run / "split_snapshot.csv"
    if smoke.digest(split) != config["split_sha256"]:
        raise ValueError("Split snapshot SHA256 mismatch")
    rows = smoke.read_csv(split)
    parts = train.partition_masked_rows(rows)
    validation_rows = parts["validation"]
    test_paths = {normalized_path(row["path"]) for row in parts["test"]}
    if any(normalized_path(row["path"]) in test_paths for row in validation_rows):
        raise ValueError("Validation/test path overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != result["best_epoch"] or state["config"] != config:
        raise ValueError("Checkpoint metadata differs from the selected run")
    model.load_state_dict(state["model_state"])
    metrics, records, predictions = inference(
        model, validation_rows, device, args.threshold
    )
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    expected_metrics = result["best_validation_metrics"]
    metric_delta = max(
        abs(metrics[key] - expected_metrics[key]) for key in expected_metrics
    )
    if metric_delta > tolerance:
        raise ValueError(f"Recomputed metric mismatch: {metric_delta}")
    prior = {
        normalized_path(row["path"]): row
        for row in smoke.read_csv(run / "validation_predictions.csv")
    }
    for record in records:
        old = prior[normalized_path(record["path"])]
        if (
            abs(record["dice"] - float(old["dice"])) > tolerance
            or abs(record["iou"] - float(old["iou"])) > tolerance
            or record["reference_pixels"] != int(old["reference_pixels"])
            or record["predicted_pixels"] != int(old["predicted_pixels"])
        ):
            raise ValueError(f"Per-image metric mismatch: {record['path']}")
    groups = select_cases(
        records,
        worst_per_category=args.worst_per_category,
        zero_count=args.zero_dice,
        best_per_category=args.best_per_category,
    )
    worst_groups = [
        ("Worst glioma", groups["worst_glioma"]),
        ("Worst meningioma", groups["worst_meningioma"]),
        ("Worst pituitary", groups["worst_pituitary"]),
        ("Zero-Dice validation cases", groups["zero_dice"]),
    ]
    best_groups = [
        ("Best glioma", groups["best_glioma"]),
        ("Best meningioma", groups["best_meningioma"]),
        ("Best pituitary", groups["best_pituitary"]),
    ]
    row_lookup = {row["path"]: row for row in validation_rows}
    render_panel(
        run / "worst_validation_by_category.png",
        "Worst held-out segmentation validation cases",
        worst_groups,
        row_lookup,
        predictions,
    )
    render_panel(
        run / "best_validation_by_category.png",
        "Best held-out segmentation validation cases",
        best_groups,
        row_lookup,
        predictions,
    )
    write_selection(
        run / "visual_audit_selections.csv",
        {"worst": worst_groups, "best": best_groups},
    )
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=config["mlflow_run_id"]):
        for artifact in (
            run / "worst_validation_by_category.png",
            run / "best_validation_by_category.png",
            run / "visual_audit_selections.csv",
            GUIDE,
        ):
            mlflow.log_artifact(str(artifact))
    print(
        json.dumps(
            {
                "status": "VISUAL_AUDIT_SUPPORT_COMPLETE",
                "device": str(device),
                "validation_images_recomputed": len(records),
                "maximum_metric_difference": metric_delta,
                "worst_tiles": sum(len(rows) for _, rows in worst_groups),
                "best_tiles": sum(len(rows) for _, rows in best_groups),
                "test_images_accessed": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
