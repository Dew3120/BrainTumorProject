"""Six training pairs, three validation pairs, two epochs: pipeline checks only."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import monai
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/brisc/brisc2025"
DEFAULT_SPLIT = (
    ROOT / "runs/classifier/20260921-170607-243133-baseline/split_snapshot.csv"
)
LABELS = ("glioma", "meningioma", "pituitary")
SIZE = 128
SCOPE = (
    "Experimental positive-case segmentation pipeline check, not a performance estimate. "
    "No test images, no negative-case specificity estimate. Patient independence and "
    "modality consistency remain unverified."
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def dataset_path(value):
    result = (ROOT / value).resolve()
    if not result.is_relative_to(RAW.resolve()):
        raise ValueError("Dataset path escapes the BRISC raw directory")
    return result


def binarize_mask(mask):
    if mask.mode not in ("1", "L"):
        raise ValueError(f"Unsupported mask mode: {mask.mode}")
    value = np.asarray(mask.convert("L"), dtype=np.uint8)
    # Preserve true 0/1 source masks; BRISC grayscale export artifacts use fixed 128.
    binary = value > 0 if value.max() <= 1 else value >= 128
    if not binary.any():
        raise ValueError("Positive-case source mask is empty")
    return Image.fromarray(binary.astype(np.uint8) * 255)


def transform_pair(image, mask, size=SIZE):
    if image.size != mask.size:
        raise ValueError("Image and mask dimensions differ")
    mask = binarize_mask(mask)
    width, height = image.size
    scale = min(size / width, size / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    x = Image.new("L", (size, size), 0)
    y = Image.new("L", (size, size), 0)
    x.paste(image.convert("L").resize(resized, Image.Resampling.BILINEAR), offset)
    y.paste(mask.resize(resized, Image.Resampling.NEAREST), offset)
    target = np.asarray(y, dtype=np.float32) / 255.0
    if not np.isin(target, [0, 1]).all() or not target.any():
        raise ValueError("Mask is non-binary or disappeared after resizing")
    inputs = torch.from_numpy(np.asarray(x, dtype=np.float32).copy() / 255.0).unsqueeze(
        0
    )
    targets = torch.from_numpy(target.copy()).unsqueeze(0)
    return inputs, targets


def select_pairs(split):
    rows = read_csv(split)
    active = {
        part: [r for r in rows if r["draft_split"] == part]
        for part in ("train", "validation", "test")
    }
    for key in ("path", "sha256", "pixel_sha256", "similarity_group"):
        groups = {part: {r[key] for r in subset} for part, subset in active.items()}
        if any(not values or "" in values for values in groups.values()):
            raise ValueError(f"Missing split or {key} metadata")
        for a, b in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if groups[a] & groups[b]:
                raise ValueError(f"Split overlap in {key}: {a}/{b}")
    selected = []
    rng = random.Random(42)
    used = set()
    for part, count in (("train", 2), ("validation", 1)):
        for label in LABELS:
            candidates = sorted(
                [r for r in active[part] if r["label"] == label and r["mask_path"]],
                key=lambda r: r["path"],
            )
            rng.shuffle(candidates)
            chosen = []
            for row in candidates:
                if row["similarity_group"] in used:
                    continue
                chosen.append(dict(row))
                used.add(row["similarity_group"])
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(
                    f"Insufficient distinct similarity groups for {part}/{label}"
                )
            selected.extend(chosen)
    publisher = {
        r["relative_path"].replace("\\", "/"): r["sha256"]
        for r in read_csv(RAW / "manifest.csv")
    }
    for row in selected:
        image = dataset_path(row["path"])
        mask = dataset_path(row["mask_path"])
        if row["original_split"] != "train" or image.stem != mask.stem:
            raise ValueError("Unexpected test pair or mismatched image/mask name")
        image_hash, mask_hash = digest(image), digest(mask)
        if image_hash != row["sha256"]:
            raise ValueError("Image changed since classifier split snapshot")
        if publisher.get(image.relative_to(RAW).as_posix()) != image_hash:
            raise ValueError("Image does not match publisher hash")
        if publisher.get(mask.relative_to(RAW).as_posix()) != mask_hash:
            raise ValueError("Mask does not match publisher hash")
        row["mask_sha256_verified"] = mask_hash
        with Image.open(image) as x, Image.open(mask) as y:
            inputs, target = transform_pair(x, y)
        row["foreground_pixels_128"] = int(target.sum().item())
        if not torch.isfinite(inputs).all():
            raise ValueError("Non-finite image input")
    return selected


class Pairs(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with (
            Image.open(dataset_path(row["path"])) as x,
            Image.open(dataset_path(row["mask_path"])) as y,
        ):
            return transform_pair(x, y)


def make_model():
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
        norm="INSTANCE",
        dropout=0.0,
    )


def mask_scores(logits, target):
    prediction = logits.sigmoid() >= 0.5
    reference = target >= 0.5
    intersection = (prediction & reference).sum(dim=(1, 2, 3)).float()
    pred_area = prediction.sum(dim=(1, 2, 3)).float()
    ref_area = reference.sum(dim=(1, 2, 3)).float()
    dice = 2 * intersection / (pred_area + ref_area).clamp_min(1)
    iou = intersection / (pred_area + ref_area - intersection).clamp_min(1)
    return dice, iou


def contact_sheet(rows, dataset, logits, destination):
    canvas = Image.new("RGB", (900, len(rows) * 350), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        image, target = dataset[index]
        predicted = logits[index, 0].sigmoid().cpu().numpy() >= 0.5
        gray = Image.fromarray(
            np.round(image[0].numpy() * 255).astype(np.uint8)
        ).convert("RGB")
        truth = Image.fromarray((target[0].numpy() * 255).astype(np.uint8))
        estimate = Image.fromarray(predicted.astype(np.uint8) * 255)
        panels = [
            gray,
            Image.composite(
                Image.blend(gray, Image.new("RGB", gray.size, "red"), 0.45), gray, truth
            ),
            Image.composite(
                Image.blend(gray, Image.new("RGB", gray.size, "cyan"), 0.45),
                gray,
                estimate,
            ),
        ]
        draw.text(
            (8, index * 350 + 5),
            f"{row['draft_split']} / {row['label']} / {row['filename']}",
            fill="black",
        )
        for column, (panel, title) in enumerate(
            zip(panels, ("Input", "Reference mask (red)", "Prediction (cyan)"))
        ):
            draw.text((column * 300 + 8, index * 350 + 25), title, fill="black")
            canvas.paste(
                panel.resize((288, 288), Image.Resampling.NEAREST),
                (column * 300 + 6, index * 350 + 48),
            )
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = select_pairs(args.split)
    parts = {
        name: [r for r in rows if r["draft_split"] == name]
        for name in ("train", "validation")
    }
    output = (
        ROOT
        / "runs/segmentation"
        / (datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "-smoke")
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "selected_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(__file__, output / "smoke_script.py")
    config = {
        "scope": SCOPE,
        "split_sha256": digest(args.split),
        "script_sha256": digest(__file__),
        "lock_sha256": digest(ROOT / "uv.lock"),
        "torch": str(torch.__version__),
        "monai": str(monai.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "image_size": 128,
        "batch_size": 2,
        "epochs": 2,
        "seed": 42,
        "learning_rate": 0.001,
        "training_pairs": 6,
        "validation_pairs": 3,
        "channels": [8, 16, 32],
        "loss": "MONAI DiceLoss(sigmoid=True) + BCEWithLogitsLoss",
        "mask_threshold": 128,
        "mask_resize": "nearest-neighbour",
        "image_resize": "bilinear",
        "image_normalization": "grayscale / 255",
        "augmentation": "none",
        "test_evaluated": False,
    }
    save_json(output / "config.json", config)
    print("RUN DIRECTORY:", output, flush=True)
    training = Pairs(parts["train"])
    validation = Pairs(parts["validation"])
    loader = DataLoader(training, batch_size=2, shuffle=True, num_workers=0)
    val_images = torch.stack([validation[i][0] for i in range(3)]).cuda()
    val_targets = torch.stack([validation[i][1] for i in range(3)]).cuda()
    model = make_model().cuda()
    first = next(model.parameters())
    original = first.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    dice_loss = DiceLoss(sigmoid=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")
    history = []
    steps = 0
    tracking = ROOT / "runs"
    artifacts = tracking / "mlflow_artifacts/segmentation-smoke"
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name("brisc-segmentation-smoke")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            "brisc-segmentation-smoke", artifact_location=artifacts.as_uri()
        )
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name=output.name) as run:
        mlflow.log_params(
            {key: value for key, value in config.items() if not isinstance(value, list)}
        )
        config["mlflow_run_id"] = run.info.run_id
        save_json(output / "config.json", config)
        for epoch in (1, 2):
            model.train()
            train_loss = 0.0
            for images, target in loader:
                images, target = images.cuda(), target.cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    if logits.shape != target.shape:
                        raise RuntimeError("Output and mask shape differ")
                    loss = dice_loss(logits, target) + criterion(logits, target)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or any(
                    not torch.isfinite(g).all().item() for g in gradients
                ):
                    raise RuntimeError("Missing or non-finite gradients")
                scaler.step(optimizer)
                scaler.update()
                steps += 1
                train_loss += loss.item() * len(images)
            model.eval()
            with torch.inference_mode():
                val_logits = model(val_images)
                if not torch.isfinite(val_logits).all().item():
                    raise RuntimeError("Non-finite validation outputs")
                val_loss = (
                    dice_loss(val_logits, val_targets)
                    + criterion(val_logits, val_targets)
                ).item()
                dice, iou = mask_scores(val_logits, val_targets)
            record = {
                "epoch": epoch,
                "training_loss": train_loss / 6,
                "validation_loss": val_loss,
                "smoke_validation_dice": dice.mean().item(),
                "smoke_validation_iou": iou.mean().item(),
            }
            history.append(record)
            mlflow.log_metrics(
                {k: v for k, v in record.items() if k != "epoch"}, step=epoch
            )
            print(json.dumps(record), flush=True)
        changed = not torch.equal(original, first.detach())
        if not changed or steps != 6:
            raise RuntimeError("Expected six updates and changed weights")
        torch.save(
            {
                "model_state": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "config": config,
                "epoch": 2,
            },
            output / "smoke_model.pt",
        )
        restored = make_model().cuda()
        restored.load_state_dict(
            torch.load(
                output / "smoke_model.pt", map_location="cpu", weights_only=True
            )["model_state"]
        )
        restored.eval()
        with torch.inference_mode():
            restored_logits = restored(val_images)
        reload_difference = (restored_logits - val_logits).abs().max().item()
        if reload_difference > 1e-6:
            raise RuntimeError("Checkpoint predictions changed after reload")
        contact_sheet(
            parts["validation"],
            validation,
            restored_logits.cpu(),
            output / "validation_overlays.png",
        )
        with torch.inference_mode():
            train_logits = restored(
                torch.stack([training[i][0] for i in range(6)]).cuda()
            )
        contact_sheet(
            parts["train"],
            training,
            train_logits.cpu(),
            output / "training_overlays.png",
        )
        result = {
            "status": "AUTOMATED_SMOKE_CHECKS_PASSED",
            "scope": SCOPE,
            "visual_review": "PENDING: inspect reference/prediction overlays for geometry",
            "training_pairs": 6,
            "validation_pairs": 3,
            "optimizer_steps": steps,
            "weights_updated": changed,
            "checkpoint_reload_max_absolute_difference": reload_difference,
            "selected_pairs_hashes_verified": True,
            "source_files_modified": False,
            "masks_binary_and_nonempty_after_resize": True,
            "history": history,
            "test_evaluated": False,
            "full_training_started": False,
            "next_gate": "Visual review, then separate tiny-set overfit check before full training",
        }
        save_json(output / "result.json", result)
        for filename in (
            "config.json",
            "result.json",
            "selected_pairs.csv",
            "validation_overlays.png",
            "training_overlays.png",
        ):
            mlflow.log_artifact(str(output / filename))
    (ROOT / "runs/segmentation/latest_smoke.txt").write_text(
        str(output), encoding="utf-8"
    )
    print("AUTOMATED_SMOKE_CHECKS_PASSED", flush=True)
    print(
        "Inspect the two overlay sheets. Low Dice after six steps does not fail this pipeline test.",
        flush=True,
    )


if __name__ == "__main__":
    main()
