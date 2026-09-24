"""Retrain the v1 segmenter with preregistered signed-distance boundary loss."""

import argparse
import csv
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import torch
from monai.losses import DiceLoss
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset

LABELS = ("glioma", "meningioma", "pituitary")
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
EXPECTED_COUNTS = {"train": 2719, "validation": 680, "test": 860}


class FullPairs(Dataset):
    def __init__(self, rows, size=128):
        self.pairs = overfit.SizedPairs(rows, size)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, mask = self.pairs[index]
        return image, mask, LABEL_INDEX[self.rows[index]["label"]], index


def signed_distance_map(target):
    """Return positive exterior and negative interior Euclidean pixel distance."""
    foreground = np.asarray(target, dtype=bool)
    if foreground.ndim != 2 or not foreground.any() or foreground.all():
        raise ValueError("Signed distance requires a nonempty, non-full 2D mask")
    outside = distance_transform_edt(~foreground)
    inside = distance_transform_edt(foreground)
    return (outside - inside).astype(np.float32)


def prepare_distance_cache(rows, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    dataset = FullPairs(rows)
    records = []
    for index, row in enumerate(rows):
        _, target, _, _ = dataset[index]
        distance = signed_distance_map(target[0].numpy() >= 0.5)
        path = destination / f"{index:05d}.npy"
        np.save(path, distance, allow_pickle=False)
        records.append(
            {
                "index": index,
                "path": row["path"],
                "mask_path": row["mask_path"],
                "cache_file": path.name,
                "minimum": float(distance.min()),
                "maximum": float(distance.max()),
                "bytes": path.stat().st_size,
                "sha256": smoke.digest(path),
            }
        )
        if (index + 1) % 500 == 0 or index + 1 == len(rows):
            print(f"Prepared boundary distances: {index + 1}/{len(rows)}", flush=True)
    return records


class BoundaryPairs(Dataset):
    def __init__(self, rows, cache_directory):
        self.base = FullPairs(rows)
        self.rows = rows
        self.cache_directory = Path(cache_directory)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, target, label_index, dataset_index = self.base[index]
        distance = np.load(
            self.cache_directory / f"{index:05d}.npy", allow_pickle=False
        )
        if distance.shape != tuple(target.shape[-2:]) or not np.isfinite(distance).all():
            raise ValueError(f"Invalid signed-distance cache entry: {index}")
        return (
            image,
            target,
            torch.from_numpy(distance.copy()).unsqueeze(0),
            label_index,
            dataset_index,
        )


def partition_masked_rows(rows):
    """Validate the frozen development partition while keeping test rows sealed."""
    parts = {
        name: [
            row
            for row in rows
            if row["draft_split"] == name and row.get("mask_path", "")
        ]
        for name in ("train", "validation", "test")
    }
    counts = {name: len(part) for name, part in parts.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected masked-pair counts: {counts}")
    if any(row["original_split"] != "train" for row in parts["train"]):
        raise ValueError("Development training contains original test data")
    if any(row["original_split"] != "train" for row in parts["validation"]):
        raise ValueError("Development validation contains original test data")
    if any(row["original_split"] != "test" for row in parts["test"]):
        raise ValueError("Sealed test membership changed")
    for name, part in parts.items():
        if {row["label"] for row in part} != set(LABELS):
            raise ValueError(f"Missing tumor category in {name}")
    for key in ("path", "sha256", "pixel_sha256", "similarity_group"):
        values = {name: {row[key] for row in part} for name, part in parts.items()}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if values[left] & values[right]:
                raise ValueError(f"Partition overlap in {key}: {left}/{right}")
    return parts


def summarize_scores(dice, iou, labels):
    result = {
        "val_dice_image_mean": float(np.mean(dice)),
        "val_iou_image_mean": float(np.mean(iou)),
    }
    type_dice = []
    type_iou = []
    for index, name in enumerate(LABELS):
        member = labels == index
        if not member.any():
            raise ValueError(f"Validation batch collection is missing {name}")
        result[f"val_dice_{name}"] = float(np.mean(dice[member]))
        result[f"val_iou_{name}"] = float(np.mean(iou[member]))
        result[f"val_count_{name}"] = int(member.sum())
        type_dice.append(result[f"val_dice_{name}"])
        type_iou.append(result[f"val_iou_{name}"])
    result["val_dice_macro_type_mean"] = float(np.mean(type_dice))
    result["val_iou_macro_type_mean"] = float(np.mean(type_iou))
    return result


def qualifies_for_top3(entries, score, limit=3):
    return len(entries) < limit or score > min(item["score"] for item in entries)


def retain_top3(entries, candidate, limit=3):
    entries.append(candidate)
    entries.sort(key=lambda item: (-item["score"], item["epoch"]))
    removed = entries[limit:]
    del entries[limit:]
    for item in removed:
        path = Path(item["path"])
        if path.exists():
            path.unlink()
    return entries


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loss_components(logits, targets, signed_distances, dice_loss):
    """Return equal-coefficient BCE, Dice, and Kervadec boundary terms."""
    logits = logits.float()
    targets = targets.float()
    signed_distances = signed_distances.float()
    if logits.shape != targets.shape or targets.shape != signed_distances.shape:
        raise ValueError("Boundary-loss tensors have different shapes")
    dice = dice_loss(logits, targets)
    bce = overfit.per_image_balanced_bce(logits, targets)
    boundary = (logits.sigmoid() * signed_distances).flatten(1).mean(1).mean()
    return {"balanced_bce": bce, "foreground_dice": dice, "boundary": boundary}


def segmentation_loss(logits, targets, signed_distances, dice_loss):
    components = loss_components(logits, targets, signed_distances, dice_loss)
    return sum(components.values())


def train_epoch(model, loader, optimizer, scaler, dice_loss, device):
    model.train()
    total_loss = 0.0
    dice_values = []
    component_totals = {
        "train_balanced_bce": 0.0,
        "train_foreground_dice_loss": 0.0,
        "train_boundary_loss": 0.0,
    }
    for images, targets, signed_distances, _, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        signed_distances = signed_distances.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        components = loss_components(logits, targets, signed_distances, dice_loss)
        loss = sum(components.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(
            not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise RuntimeError("Non-finite training gradient")
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(images)
        component_totals["train_balanced_bce"] += (
            components["balanced_bce"].item() * len(images)
        )
        component_totals["train_foreground_dice_loss"] += (
            components["foreground_dice"].item() * len(images)
        )
        component_totals["train_boundary_loss"] += (
            components["boundary"].item() * len(images)
        )
        with torch.inference_mode():
            batch_dice, _ = smoke.mask_scores(logits.float(), targets)
            dice_values.extend(batch_dice.cpu().tolist())
    return {
        "train_loss": total_loss / len(loader.dataset),
        "train_dice_batch_state_mean": float(np.mean(dice_values)),
        **{
            key: value / len(loader.dataset)
            for key, value in component_totals.items()
        },
    }


def validate(model, loader, dice_loss, device, include_records=False):
    model.eval()
    total_loss = 0.0
    all_dice = []
    all_iou = []
    all_labels = []
    records = []
    with torch.inference_mode():
        for images, targets, signed_distances, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            signed_distances = signed_distances.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            loss = segmentation_loss(logits, targets, signed_distances, dice_loss)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation output")
            dice, iou = smoke.mask_scores(logits.float(), targets)
            total_loss += loss.item() * len(images)
            all_dice.extend(dice.cpu().tolist())
            all_iou.extend(iou.cpu().tolist())
            all_labels.extend(labels.tolist())
            if include_records:
                predicted_pixels = (logits >= 0).sum(dim=(1, 2, 3)).cpu().tolist()
                reference_pixels = targets.sum(dim=(1, 2, 3)).cpu().int().tolist()
                for position, dataset_index in enumerate(indices.tolist()):
                    row = loader.dataset.rows[dataset_index]
                    records.append(
                        {
                            "path": row["path"],
                            "filename": row["filename"],
                            "label": row["label"],
                            "dice": float(dice[position].item()),
                            "iou": float(iou[position].item()),
                            "reference_pixels": reference_pixels[position],
                            "predicted_pixels": predicted_pixels[position],
                        }
                    )
    metrics = summarize_scores(
        np.asarray(all_dice), np.asarray(all_iou), np.asarray(all_labels)
    )
    metrics["val_loss"] = total_loss / len(loader.dataset)
    return metrics, records


def save_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", type=Path, default=smoke.ROOT / "data/splits/draft_split.csv"
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.epochs != 75 or args.patience != 12 or args.batch_size != 8:
        raise ValueError(
            "This reviewed baseline is fixed at 75 epochs, patience 12, batch 8"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    split = args.split.resolve()
    rows = smoke.read_csv(split)
    parts = partition_masked_rows(rows)
    label_counts = {
        part: {label: sum(row["label"] == label for row in subset) for label in LABELS}
        for part, subset in parts.items()
    }
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    output = (
        smoke.ROOT
        / "runs/segmentation_v2_boundary_loss"
        / (datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "-boundary")
    )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_directory = output / "top3_checkpoints"
    checkpoint_directory.mkdir()
    cache_root = output / "distance_cache"
    train_cache = cache_root / "train"
    validation_cache = cache_root / "validation"
    train_cache_records = prepare_distance_cache(parts["train"], train_cache)
    validation_cache_records = prepare_distance_cache(
        parts["validation"], validation_cache
    )
    save_csv(
        output / "distance_cache_manifest.csv",
        [
            {"partition": "train", **row} for row in train_cache_records
        ]
        + [
            {"partition": "validation", **row}
            for row in validation_cache_records
        ],
    )
    shutil.copy2(split, output / "split_snapshot.csv")
    shutil.copy2(__file__, output / "training_script.py")
    shutil.copy2(overfit.__file__, output / "segmentation_overfit_gate.py")
    shutil.copy2(smoke.__file__, output / "segmentation_smoke_test.py")
    fixed_overlay_rows = []
    for label in LABELS:
        fixed_overlay_rows.extend(
            sorted(
                [row for row in parts["validation"] if row["label"] == label],
                key=lambda row: row["path"],
            )[:3]
        )
    save_csv(output / "fixed_overlay_pairs.csv", fixed_overlay_rows)
    config = {
        "scope": "Experimental boundary-aware BRISC positive-case segmentation training; not clinical validation",
        "split_sha256": smoke.digest(split),
        "training_script_sha256": smoke.digest(__file__),
        "overfit_script_sha256": smoke.digest(overfit.__file__),
        "smoke_script_sha256": smoke.digest(smoke.__file__),
        "lock_sha256": smoke.digest(smoke.ROOT / "uv.lock"),
        "seed": 42,
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "monai": str(smoke.monai.__version__),
        "image_size": 128,
        "model": "MONAI compact 2D U-Net with linear logit head",
        "channels": [16, 32, 64, 128],
        "num_res_units": 2,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "loss": "equal coefficients: per-image balanced BCE + foreground Dice + signed-distance boundary loss",
        "boundary_loss": {
            "signed_distance": "outside EDT minus inside EDT in model pixels",
            "normalization": "none",
            "term": "mean(sigmoid(logit) * signed_distance)",
            "coefficient": 1.0,
        },
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": 0.0001,
        "selection_metric": "validation macro mean of glioma/meningioma/pituitary image-level hard Dice",
        "precision": "CUDA AMP FP16 model forward, FP32 weighted loss and parameters",
        "gradient_scale_initial": 1024.0,
        "gradient_scale_growth_interval": 1000000,
        "augmentation": "none",
        "distance_cache_manifest_sha256": smoke.digest(
            output / "distance_cache_manifest.csv"
        ),
        "num_workers": 2,
        "train_pairs": len(parts["train"]),
        "validation_pairs": len(parts["validation"]),
        "sealed_test_pairs": len(parts["test"]),
        "label_counts": label_counts,
        "top_checkpoints_retained": 3,
        "validation_used_for_selection_and_early_stopping": True,
        "test_evaluated": False,
        "patient_independence_verified": False,
    }
    smoke.save_json(output / "config.json", config)
    train_data = BoundaryPairs(parts["train"], train_cache)
    validation_data = BoundaryPairs(parts["validation"], validation_cache)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        worker_init_fn=seed_worker,
        persistent_workers=True,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        persistent_workers=True,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=1000000)
    dice_loss = DiceLoss(sigmoid=True)
    tracking = smoke.ROOT / "runs"
    artifact_root = tracking / "mlflow_artifacts/segmentation-v2-boundary-loss"
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name("brisc-segmentation-v2-boundary-loss")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            "brisc-segmentation-v2-boundary-loss", artifact_location=artifact_root.as_uri()
        )
    )
    history = []
    top3 = []
    best_score = -1.0
    epochs_without_improvement = 0
    stopped_early = False
    with mlflow.start_run(experiment_id=experiment_id, run_name=output.name) as run:
        config["mlflow_run_id"] = run.info.run_id
        smoke.save_json(output / "config.json", config)
        mlflow.log_params({key: str(value) for key, value in config.items()})
        mlflow.log_artifact(str(output / "config.json"))
        mlflow.log_artifact(str(output / "split_snapshot.csv"))
        for epoch in range(1, args.epochs + 1):
            training_metrics = train_epoch(
                model, train_loader, optimizer, scaler, dice_loss, device
            )
            validation_metrics, _ = validate(
                model, validation_loader, dice_loss, device
            )
            record = {"epoch": epoch, **training_metrics, **validation_metrics}
            history.append(record)
            save_csv(output / "history.csv", history)
            mlflow.log_metrics(
                {key: value for key, value in record.items() if key != "epoch"},
                step=epoch,
            )
            score = validation_metrics["val_dice_macro_type_mean"]
            if qualifies_for_top3(top3, score):
                checkpoint = checkpoint_directory / (
                    f"epoch_{epoch:03d}_macro_dice_{score:.6f}.pt"
                )
                torch.save(
                    {
                        "model_state": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "epoch": epoch,
                        "validation_metrics": validation_metrics,
                        "config": config,
                    },
                    checkpoint,
                )
                top3 = retain_top3(
                    top3,
                    {
                        "epoch": epoch,
                        "score": score,
                        "path": str(checkpoint),
                        "validation_metrics": validation_metrics,
                    },
                )
                smoke.save_json(output / "top3_checkpoints.json", top3)
            if score > best_score + config["early_stopping_min_delta"]:
                best_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(
                f"Epoch {epoch:03d}: train loss={record['train_loss']:.4f}, "
                f"val Dice={record['val_dice_image_mean']:.4f}, macro={score:.4f}, "
                f"gl={record['val_dice_glioma']:.4f}, "
                f"me={record['val_dice_meningioma']:.4f}, "
                f"pi={record['val_dice_pituitary']:.4f}, "
                f"patience={epochs_without_improvement}/{args.patience}",
                flush=True,
            )
            if epochs_without_improvement >= args.patience:
                stopped_early = True
                break
        top3.sort(key=lambda item: (-item["score"], item["epoch"]))
        best = top3[0]
        best_checkpoint = Path(best["path"])
        restored = overfit.make_diagnostic_model("linear", "compact").to(device)
        saved = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
        restored.load_state_dict(saved["model_state"])
        best_metrics, prediction_records = validate(
            restored, validation_loader, dice_loss, device, include_records=True
        )
        metric_delta = max(
            abs(best_metrics[key] - best["validation_metrics"][key])
            for key in best_metrics
        )
        if metric_delta > 1e-6:
            raise RuntimeError(
                f"Best checkpoint metric reload mismatch: {metric_delta}"
            )
        save_csv(output / "validation_predictions.csv", prediction_records)
        top3_rows = [
            {
                "rank": rank,
                "epoch": item["epoch"],
                "checkpoint": Path(item["path"]).name,
                **item["validation_metrics"],
            }
            for rank, item in enumerate(top3, start=1)
        ]
        save_csv(output / "top3_comparison.csv", top3_rows)
        row_by_path = {row["path"]: row for row in parts["validation"]}
        worst_rows = []
        for label in LABELS:
            worst_rows.extend(
                [
                    row_by_path[item["path"]]
                    for item in sorted(
                        [r for r in prediction_records if r["label"] == label],
                        key=lambda item: (item["dice"], item["path"]),
                    )[:3]
                ]
            )
        for name, selected_rows in (
            ("fixed_validation_overlays.png", fixed_overlay_rows),
            ("worst_validation_overlays.png", worst_rows),
        ):
            selected_dataset = overfit.SizedPairs(selected_rows, 128)
            selected_images = torch.stack(
                [selected_dataset[i][0] for i in range(len(selected_dataset))]
            ).to(device)
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.float16),
            ):
                selected_logits = restored(selected_images).float().cpu()
            smoke.contact_sheet(
                selected_rows, selected_dataset, selected_logits, output / name
            )
        figure, axis = plt.subplots(figsize=(10, 6))
        for key, label in (
            ("val_dice_macro_type_mean", "Validation macro type mean"),
            ("val_dice_glioma", "Glioma"),
            ("val_dice_meningioma", "Meningioma"),
            ("val_dice_pituitary", "Pituitary"),
        ):
            axis.plot(
                [row["epoch"] for row in history],
                [row[key] for row in history],
                label=label,
            )
        axis.set(
            xlabel="Epoch",
            ylabel="Validation foreground Dice",
            ylim=(0, 1),
            title="Boundary-aware segmenter validation Dice by tumor type",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "validation_dice_curve.png", dpi=160)
        plt.close(figure)
        result = {
            "status": "EARLY_STOPPED" if stopped_early else "MAX_EPOCHS_COMPLETED",
            "epochs_completed": epoch,
            "best_epoch": best["epoch"],
            "best_validation_metrics": best_metrics,
            "top3": top3_rows,
            "best_checkpoint_reload_max_metric_difference": metric_delta,
            "train_pairs": len(parts["train"]),
            "validation_pairs": len(parts["validation"]),
            "sealed_test_pairs": len(parts["test"]),
            "test_evaluated": False,
            "patient_independence_verified": False,
            "visual_review": "PENDING",
            "interpretation": "Experimental held-out BRISC validation performance only; not clinical performance.",
        }
        smoke.save_json(output / "result.json", result)
        for artifact in (
            "history.csv",
            "top3_checkpoints.json",
            "top3_comparison.csv",
            "validation_predictions.csv",
            "fixed_validation_overlays.png",
            "worst_validation_overlays.png",
            "validation_dice_curve.png",
            "result.json",
        ):
            mlflow.log_artifact(str(output / artifact))
        for item in top3:
            mlflow.log_artifact(item["path"], artifact_path="top3_checkpoints")
        mlflow.set_tag("test_evaluated", "false")
        mlflow.set_tag("best_epoch", str(best["epoch"]))
    (smoke.ROOT / "runs/segmentation_v2_boundary_loss/latest_boundary.txt").write_text(
        str(output), encoding="utf-8"
    )
    print("RUN DIRECTORY:", output, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
