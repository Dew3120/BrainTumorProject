"""Predetermined six-pair memorization check; separate validation is diagnostic only."""

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
import phase4_segmentation_smoke as smoke
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

RULE = {
    "max_epochs": 100,
    "mean_training_dice_required": 0.90,
    "minimum_individual_training_dice_required": 0.80,
    "consecutive_epochs_required": 5,
    "plateau_check_start_epoch": 50,
    "plateau_window": 20,
    "low_dice_ceiling": 0.20,
    "plateau_range_maximum": 0.02,
    "validation_used_for_selection_or_stopping": False,
}


def decision(history):
    if len(history) >= 5 and all(
        row["train_dice"] >= RULE["mean_training_dice_required"]
        and row["train_min_dice"] >= RULE["minimum_individual_training_dice_required"]
        for row in history[-5:]
    ):
        return "MEMORIZATION_PASS"
    if len(history) >= RULE["plateau_check_start_epoch"]:
        recent = [r["train_dice"] for r in history[-RULE["plateau_window"] :]]
        if (
            max(recent) < RULE["low_dice_ceiling"]
            and max(recent) - min(recent) < RULE["plateau_range_maximum"]
        ):
            return "STOP_AND_DIAGNOSE_LOW_PLATEAU"
    if len(history) >= RULE["max_epochs"]:
        return "BUDGET_EXHAUSTED_DIAGNOSE_BEFORE_FULL_TRAINING"
    return "CONTINUE"


def write_history(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_diagnostic_model(head, model_size="tiny"):
    if model_size == "tiny":
        channels = (8, 16, 32)
        model = smoke.make_model()
    elif model_size == "compact":
        channels = (16, 32, 64, 128)
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=(2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
            dropout=0.0,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    if head == "linear":
        # Preserve the encoder and inner decoder; replace the one-channel normalized
        # residual output block with an unrestricted linear logit layer.
        model.model[2] = torch.nn.ConvTranspose2d(
            channels[0] * 2, 1, 3, stride=2, padding=1, output_padding=1
        )
    return model


class SizedPairs(smoke.Pairs):
    """Reuse the verified smoke preprocessing at an explicit square resolution."""

    def __init__(self, rows, size):
        super().__init__(rows)
        self.size = size

    def __getitem__(self, index):
        row = self.rows[index]
        with (
            Image.open(smoke.dataset_path(row["path"])) as image,
            Image.open(smoke.dataset_path(row["mask_path"])) as mask,
        ):
            return smoke.transform_pair(image, mask, size=self.size)


def foreground_statistics(rows, targets):
    """Describe each transformed training mask and derive one fixed BCE weight."""
    if len(rows) != len(targets):
        raise ValueError("Mask metadata and tensors have different lengths")
    per_image = []
    foreground_total = 0
    pixel_total = 0
    for index, (row, target) in enumerate(zip(rows, targets, strict=True), start=1):
        foreground = int(target.sum().item())
        pixels = target.numel()
        if foreground <= 0 or foreground >= pixels:
            raise ValueError(f"Training mask {index} has invalid foreground count")
        foreground_total += foreground
        pixel_total += pixels
        per_image.append(
            {
                "pair_index": index,
                "filename": row["filename"],
                "foreground_pixels": foreground,
                "total_pixels": pixels,
                "foreground_fraction": foreground / pixels,
            }
        )
    background_total = pixel_total - foreground_total
    return per_image, {
        "foreground_pixels": foreground_total,
        "background_pixels": background_total,
        "total_pixels": pixel_total,
        "foreground_fraction": foreground_total / pixel_total,
        "pos_weight": background_total / foreground_total,
    }


def per_image_positive_weights(targets):
    """Return background/foreground for each positive mask in a batch."""
    foreground = targets.sum(dim=(1, 2, 3))
    pixels = targets[0].numel()
    if (foreground <= 0).any() or (foreground >= pixels).any():
        raise ValueError("Per-image weighting requires nonempty, non-full masks")
    return (pixels - foreground) / foreground


def per_image_balanced_bce(logits, targets):
    """Balance foreground/background inside each image, then average images."""
    weights = per_image_positive_weights(targets).view(-1, 1, 1, 1)
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted = raw * torch.where(targets > 0.5, weights, 1.0)
    return weighted.flatten(1).mean(1).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-run",
        type=Path,
        default=smoke.ROOT / "runs/segmentation/20260921-174907-092984-smoke",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--head", choices=("original", "linear"), default="original")
    parser.add_argument("--model-size", choices=("tiny", "compact"), default="tiny")
    parser.add_argument("--image-size", type=int, choices=(128, 256), default=128)
    parser.add_argument(
        "--balanced-bce",
        action="store_true",
        help="Use one fixed pos_weight derived only from all six training masks",
    )
    parser.add_argument(
        "--fixed-pos-weight",
        type=float,
        help="Freeze BCE pos_weight to this positive value instead of recomputing it",
    )
    parser.add_argument(
        "--per-image-pos-weight",
        action="store_true",
        help="Balance foreground/background separately inside every image",
    )
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("Learning rate must be positive and at most 0.01")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.fixed_pos_weight is not None and (
        not args.balanced_bce or args.fixed_pos_weight <= 0
    ):
        raise ValueError(
            "--fixed-pos-weight requires --balanced-bce and a positive value"
        )
    if args.per_image_pos_weight and (
        not args.balanced_bce or args.fixed_pos_weight is not None
    ):
        raise ValueError(
            "--per-image-pos-weight requires --balanced-bce and excludes a fixed weight"
        )
    source = args.smoke_run.resolve()
    config_before = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if smoke.digest(smoke.__file__) != config_before["script_sha256"]:
        raise ValueError(
            "Smoke pipeline changed; inspect before reusing this implementation"
        )
    rows = smoke.read_csv(source / "selected_pairs.csv")
    if len(rows) != 9 or len({r["path"] for r in rows}) != 9:
        raise ValueError("Expected original nine unique smoke pairs")
    subsets = {
        name: [r for r in rows if r["draft_split"] == name]
        for name in ("train", "validation")
    }
    if [len(subsets[p]) for p in subsets] != [6, 3] or len(
        {r["similarity_group"] for r in rows}
    ) != 9:
        raise ValueError("Smoke membership or group separation changed")
    for row in rows:
        if row["original_split"] != "train":
            raise ValueError("Unexpected original test data")
        if (
            smoke.digest(smoke.dataset_path(row["path"])) != row["sha256"]
            or smoke.digest(smoke.dataset_path(row["mask_path"]))
            != row["mask_sha256_verified"]
        ):
            raise ValueError("Image or mask changed since smoke run")
    datasets = {
        name: SizedPairs(subset, args.image_size) for name, subset in subsets.items()
    }
    tensors = {
        name: (
            torch.stack([ds[i][0] for i in range(len(ds))]).cuda(),
            torch.stack([ds[i][1] for i in range(len(ds))]).cuda(),
        )
        for name, ds in datasets.items()
    }
    per_image_foreground, aggregate_foreground = foreground_statistics(
        subsets["train"], tensors["train"][1]
    )
    resolution_comparison = {}
    for size in (128, 256):
        comparison_dataset = SizedPairs(subsets["train"], size)
        comparison_targets = torch.stack(
            [comparison_dataset[i][1] for i in range(len(comparison_dataset))]
        )
        per_image, aggregate = foreground_statistics(
            subsets["train"], comparison_targets
        )
        resolution_comparison[str(size)] = {
            "per_image": per_image,
            "aggregate": aggregate,
        }
    used_pos_weight = (
        args.fixed_pos_weight
        if args.fixed_pos_weight is not None
        else aggregate_foreground["pos_weight"]
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = (
        smoke.ROOT
        / "runs/segmentation"
        / (
            datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            + f"-overfit-{args.image_size}-{args.model_size}"
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "selected_pairs.csv", output / "selected_pairs.csv")
    shutil.copy2(__file__, output / "overfit_script.py")
    shutil.copy2(smoke.__file__, output / "phase4_segmentation_smoke.py")
    model = make_diagnostic_model(args.head, args.model_size).cuda()
    model_channels = [8, 16, 32] if args.model_size == "tiny" else [16, 32, 64, 128]
    config = {
        "purpose": "Training-set memorization diagnostic, not a generalization claim",
        "decision_rule": RULE,
        "initialization": "fresh random seed 42; not resumed smoke weights",
        "same_selected_pairs_sha256": smoke.digest(source / "selected_pairs.csv"),
        "source_smoke_run": source.name,
        "split_sha256": config_before["split_sha256"],
        "overfit_script_sha256": smoke.digest(__file__),
        "smoke_script_sha256": smoke.digest(smoke.__file__),
        "lock_sha256": smoke.digest(smoke.ROOT / "uv.lock"),
        "torch": str(torch.__version__),
        "monai": str(smoke.monai.__version__),
        "seed": 42,
        "learning_rate": args.learning_rate,
        "output_head": args.head,
        "model_size": args.model_size,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "weight_decay": 0.0001,
        "batch_size": 2,
        "image_size": args.image_size,
        "channels": model_channels,
        "num_res_units": 1 if args.model_size == "tiny" else 2,
        "precision": "FP32 for this diagnostic",
        "augmentation": "none",
        "loss": "per-image pos_weight BCE + unchanged foreground soft Dice"
        if args.per_image_pos_weight
        else (
            "fixed aggregate pos_weight BCE + unchanged foreground soft Dice"
            if args.balanced_bce
            else "unweighted BCE + foreground soft Dice"
        ),
        "fixed_pos_weight": None
        if args.per_image_pos_weight
        else (used_pos_weight if args.balanced_bce else 1.0),
        "per_image_pos_weights": per_image_positive_weights(
            tensors["train"][1]
        ).tolist()
        if args.per_image_pos_weight
        else None,
        "weight_source": "each transformed training mask independently"
        if args.per_image_pos_weight
        else (
            "explicit fixed value from the prior 128x128 six-mask aggregate"
            if args.fixed_pos_weight is not None
            else "all six transformed training masks only; fixed before optimization"
        ),
        "aggregate_training_mask_statistics": aggregate_foreground,
        "per_image_training_mask_statistics": per_image_foreground,
        "training_mask_resolution_comparison": resolution_comparison,
        "train_pairs": 6,
        "validation_pairs": 3,
        "metric_definition": "Mean foreground hard Dice per image at sigmoid >= 0.5, in eval mode after each epoch",
        "test_evaluated": False,
        "patient_independence_verified": False,
    }
    smoke.save_json(output / "config.json", config)
    comparison_rows = []
    for index, row in enumerate(subsets["train"]):
        comparison_rows.append(
            {
                "pair_index": index + 1,
                "filename": row["filename"],
                "label": row["label"],
                "foreground_pixels_128": resolution_comparison["128"]["per_image"][
                    index
                ]["foreground_pixels"],
                "foreground_fraction_128": resolution_comparison["128"]["per_image"][
                    index
                ]["foreground_fraction"],
                "foreground_pixels_256": resolution_comparison["256"]["per_image"][
                    index
                ]["foreground_pixels"],
                "foreground_fraction_256": resolution_comparison["256"]["per_image"][
                    index
                ]["foreground_fraction"],
            }
        )
    write_history(output / "foreground_by_resolution.csv", comparison_rows)
    print("PREDECLARED RULE:", json.dumps(RULE), flush=True)
    print(
        "TRAINING MASK STATISTICS:",
        json.dumps(
            {
                "per_image": per_image_foreground,
                "aggregate": aggregate_foreground,
                "comparison_128_and_256": resolution_comparison,
            }
        ),
        flush=True,
    )
    print("RUN DIRECTORY:", output, flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0001
    )
    dice_loss = DiceLoss(sigmoid=True)
    if args.per_image_pos_weight:
        bce = per_image_balanced_bce
    else:
        pos_weight = (
            tensors["train"][1].new_tensor(used_pos_weight)
            if args.balanced_bce
            else None
        )
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(
        TensorDataset(*tensors["train"]), batch_size=2, shuffle=True, num_workers=0
    )
    tracking = smoke.ROOT / "runs"
    artifact_root = tracking / "mlflow_artifacts/segmentation-overfit"
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    exp = mlflow.get_experiment_by_name("brisc-segmentation-overfit")
    exp_id = (
        exp.experiment_id
        if exp
        else mlflow.create_experiment(
            "brisc-segmentation-overfit", artifact_location=artifact_root.as_uri()
        )
    )
    history = []
    with mlflow.start_run(experiment_id=exp_id, run_name=output.name) as run:
        config["mlflow_run_id"] = run.info.run_id
        smoke.save_json(output / "config.json", config)
        mlflow.log_params({k: str(v) for k, v in config.items()})
        mlflow.log_artifact(str(output / "config.json"))
        mlflow.log_artifact(str(output / "foreground_by_resolution.csv"))
        for epoch in range(1, RULE["max_epochs"] + 1):
            model.train()
            total = 0.0
            for images, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = dice_loss(logits, targets) + bce(logits, targets)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite loss: stop and diagnose")
                loss.backward()
                if any(
                    not torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                    if p.grad is not None
                ):
                    raise RuntimeError("Non-finite gradient: stop and diagnose")
                optimizer.step()
                total += loss.item() * len(images)
            model.eval()
            scores, predictions = {}, {}
            with torch.inference_mode():
                for name, (images, targets) in tensors.items():
                    logits = model(images)
                    if not torch.isfinite(logits).all().item():
                        raise RuntimeError("Non-finite evaluation output")
                    dice, iou = smoke.mask_scores(logits, targets)
                    scores[name] = (
                        dice.cpu().tolist(),
                        iou.cpu().tolist(),
                        (dice_loss(logits, targets) + bce(logits, targets)).item(),
                    )
                    predictions[name] = logits
            train_dice, train_iou, train_eval_loss = scores["train"]
            val_dice, val_iou, val_loss = scores["validation"]
            record = {
                "epoch": epoch,
                "optimization_loss": total / 6,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                "train_dice": float(np.mean(train_dice)),
                "train_min_dice": min(train_dice),
                "val_dice": float(np.mean(val_dice)),
                "train_iou": float(np.mean(train_iou)),
                "val_iou": float(np.mean(val_iou)),
                **{
                    f"train_pair_{i + 1}_dice": value
                    for i, value in enumerate(train_dice)
                },
                **{
                    f"train_pair_{item['pair_index']}_foreground_fraction": item[
                        "foreground_fraction"
                    ]
                    for item in per_image_foreground
                },
                **{f"val_pair_{i + 1}_dice": value for i, value in enumerate(val_dice)},
            }
            history.append(record)
            write_history(output / "history.csv", history)
            mlflow.log_metrics(
                {k: v for k, v in record.items() if k != "epoch"}, step=epoch
            )
            print(
                f"Epoch {epoch:03d}: train Dice={record['train_dice']:.4f}, worst train={record['train_min_dice']:.4f}, validation Dice={record['val_dice']:.4f}",
                flush=True,
            )
            status = decision(history)
            if status != "CONTINUE":
                break
        torch.save(
            {
                "model_state": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "epoch": epoch,
                "config": config,
            },
            output / "overfit_model.pt",
        )
        restored = make_diagnostic_model(args.head, args.model_size).cuda()
        restored.load_state_dict(
            torch.load(
                output / "overfit_model.pt", map_location="cpu", weights_only=True
            )["model_state"]
        )
        restored.eval()
        with torch.inference_mode():
            delta = max(
                (restored(tensors[name][0]) - predictions[name]).abs().max().item()
                for name in tensors
            )
        if delta > 1e-6:
            raise RuntimeError("Reload mismatch")
        for name, subset in subsets.items():
            smoke.contact_sheet(
                subset,
                datasets[name],
                predictions[name].cpu(),
                output / f"{name}_overlays.png",
            )
        figure, axis = plt.subplots(figsize=(9, 5))
        for key, label in (
            ("train_dice", "Training mean"),
            ("train_min_dice", "Worst training pair"),
            ("val_dice", "Separate validation mean (3 pairs)"),
        ):
            axis.plot(
                [r["epoch"] for r in history], [r[key] for r in history], label=label
            )
        axis.axhline(0.9, linestyle="--", color="gray", label="Training mean target")
        axis.set(
            xlabel="Epoch",
            ylabel="Foreground Dice",
            ylim=(0, 1),
            title="Six-pair memorization check; not a performance benchmark",
        )
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "dice_curve.png", dpi=160)
        plt.close(figure)
        result = {
            "status": status,
            "decision_rule": RULE,
            "epochs_completed": epoch,
            "optimizer_steps": epoch * 3,
            "final_metrics": history[-1],
            "last_five_train_dice": [r["train_dice"] for r in history[-5:]],
            "last_five_worst_train_dice": [r["train_min_dice"] for r in history[-5:]],
            "checkpoint_reload_max_absolute_difference": delta,
            "same_smoke_pairs_used": True,
            "fixed_pos_weight": config["fixed_pos_weight"],
            "per_image_pos_weights": config["per_image_pos_weights"],
            "model_size": config["model_size"],
            "model_parameter_count": config["model_parameter_count"],
            "aggregate_training_mask_statistics": aggregate_foreground,
            "per_image_training_mask_statistics": per_image_foreground,
            "training_mask_resolution_comparison": resolution_comparison,
            "test_evaluated": False,
            "validation_used_for_selection": False,
            "interpretation": "PASS supports learnability on six known positive images only. Non-pass requires diagnosis, not a conclusion that the architecture is fundamentally broken.",
            "visual_review": "PENDING",
        }
        smoke.save_json(output / "result.json", result)
        for file in (
            "history.csv",
            "result.json",
            "selected_pairs.csv",
            "foreground_by_resolution.csv",
            "dice_curve.png",
            "train_overlays.png",
            "validation_overlays.png",
        ):
            mlflow.log_artifact(str(output / file))
        mlflow.set_tag("diagnostic_outcome", status)
    (smoke.ROOT / "runs/segmentation/latest_overfit.txt").write_text(
        str(output), encoding="utf-8"
    )
    print(status, flush=True)


if __name__ == "__main__":
    main()
