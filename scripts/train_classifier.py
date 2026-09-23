"""Experimental BRISC classifier. Does not evaluate the held-out test set."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/splits/draft_split.csv"
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SCOPE = (
    "EXPERIMENTAL BRISC internal baseline. data suitability review suitability review remains open. "
    "Patient independence and modality consistency are not verified. "
    "Scores are not clinical validation. Measurements and segmentation are not included."
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_path(row):
    path = (ROOT / row["path"]).resolve()
    if not path.is_relative_to((ROOT / "data/raw/brisc").resolve()):
        raise ValueError(f"Image outside the expected dataset: {row['path']}")
    return path


def inspect_split(split_path=SPLIT):
    with Path(split_path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected = {"train": 3570, "validation": 893, "test": 1000, "review_excluded": 537}
    counts = Counter(row["draft_split"] for row in rows)
    if dict(counts) != expected:
        raise ValueError(f"Split changed. Review it before training: {dict(counts)}")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Repeated image paths in split.")
    if any(row["label"] not in CLASSES for row in rows):
        raise ValueError("Unexpected class label.")
    if any(
        (r["original_split"] == "test") != (r["draft_split"] == "test") for r in rows
    ):
        raise ValueError("Original test membership changed.")
    for key in ("similarity_group", "sha256", "pixel_sha256"):
        sets = {
            part: {row[key] for row in rows if row["draft_split"] == part}
            for part in ("train", "validation", "test")
        }
        if "" in set.union(*sets.values()):
            raise ValueError(f"Missing {key} values.")
        for a, b in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if sets[a] & sets[b]:
                raise ValueError(f"{key} overlap between {a} and {b}.")
    for part in ("train", "validation"):
        subset = [row for row in rows if row["draft_split"] == part]
        if {row["label"] for row in subset} != set(CLASSES):
            raise ValueError(f"Missing classes in {part}.")
        for row in subset:
            if sha256(image_path(row)) != row["sha256"]:
                raise ValueError(f"Image changed since audit: {row['path']}")
    # Test metadata is checked above; no test image pixels are opened.
    print("DATA CHECK PASSED:", dict(counts), flush=True)
    print(SCOPE, flush=True)
    return rows


def prepare_image(image, training=False):
    # Preserve aspect ratio and pad; avoid a center crop that could remove a lesion.
    image = ImageOps.pad(
        image.convert("L"), (224, 224), method=Image.Resampling.BILINEAR, color=0
    ).convert("RGB")
    if training:
        image = transforms.ColorJitter(brightness=0.1, contrast=0.1)(image)
    return transforms.functional.normalize(
        transforms.functional.to_tensor(image), MEAN, STD
    )


class BrainImages(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(image_path(row)) as image:
            tensor = prepare_image(image, self.training)
        return tensor, CLASSES.index(row["label"])


def metrics_for(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    tumor_truth = labels != CLASSES.index("no_tumor")
    tumor_scores = probabilities[:, :3].sum(axis=1)
    tumor_prediction = tumor_scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(
        tumor_truth, tumor_prediction, labels=[False, True]
    ).ravel()
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(
                labels,
                predicted,
                labels=list(range(4)),
                average="macro",
                zero_division=0,
            )
        ),
        "tumor_sensitivity": float(tp / (tp + fn)),
        "no_tumor_specificity": float(tn / (tn + fp)),
        "tumor_roc_auc": float(roc_auc_score(tumor_truth, tumor_scores)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
    report = classification_report(
        labels,
        predicted,
        labels=list(range(4)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    for name in CLASSES:
        metrics[f"recall_{name}"] = float(report[name]["recall"])
    return metrics, report


def run_epoch(model, loader, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    targets, probabilities = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss; stop and inspect this run.")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * len(labels)
            targets.append(labels.cpu().numpy())
            probabilities.append(logits.detach().float().softmax(dim=1).cpu().numpy())
    return (
        loss_sum / len(loader.dataset),
        np.concatenate(targets),
        np.concatenate(probabilities),
    )


def save_plots(output, history, labels, probabilities):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Training")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["macro_f1"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation macro F1", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    plt.close(figure)
    matrix = confusion_matrix(
        labels, probabilities.argmax(axis=1), labels=list(range(4))
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted class",
        ylabel="Dataset label",
        title="Validation only: four-class predictions",
    )
    for i in range(4):
        for j in range(4):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def train(args):
    if not args.experimental:
        raise ValueError(
            "Use --experimental: data suitability review is still open."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Fix the environment before training.")
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("Use at least one epoch and a batch size of at least two.")
    rows = inspect_split()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parts = {
        name: [r for r in rows if r["draft_split"] == name]
        for name in ("train", "validation")
    }
    if args.smoke:
        # A tiny run tests the complete pipeline; these scores are not useful performance estimates.
        parts = {
            name: [
                row
                for label in CLASSES
                for row in [r for r in subset if r["label"] == label][:8]
            ]
            for name, subset in parts.items()
        }
    output = (
        ROOT
        / "runs/classifier"
        / (
            datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            + ("-smoke" if args.smoke else "-baseline")
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    torch.hub.set_dir(str(ROOT / ".cache/torch"))
    shutil.copy2(SPLIT, output / "split_snapshot.csv")
    shutil.copy2(__file__, output / "training_script.py")
    shutil.copy2(ROOT / "uv.lock", output / "uv.lock")
    config = {
        "scope": SCOPE,
        "classes": CLASSES,
        "seed": args.seed,
        "epochs_requested": 1 if args.smoke else args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "patience": 5,
        "selection_metric": "validation_macro_f1",
        "tumor_threshold": 0.5,
        "architecture": "resnet18",
        "initial_weights": "IMAGENET1K_V1",
        "preprocessing": "grayscale -> aspect-preserving 224x224 black padding -> RGB -> ImageNet normalization",
        "augmentation": "training-only brightness/contrast jitter 0.1; no geometric augmentation",
        "mean": MEAN,
        "std": STD,
        "smoke": args.smoke,
        "train_images": len(parts["train"]),
        "validation_images": len(parts["validation"]),
        "split_sha256": sha256(SPLIT),
        "script_sha256": sha256(__file__),
        "lock_sha256": sha256(ROOT / "uv.lock"),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "workers": 0,
        "test_evaluated": False,
        "patient_independence_verified": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
    }
    save_json(output / "config.json", config)
    print("RUN DIRECTORY:", output, flush=True)
    loaders = {
        name: DataLoader(
            BrainImages(subset, name == "train"),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, subset in parts.items()
    }
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    tracking = ROOT / "runs"
    artifacts = tracking / "mlflow_artifacts/classifier"
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name("brisc-classifier-experimental")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            "brisc-classifier-experimental", artifact_location=artifacts.as_uri()
        )
    )
    best_score, stale, history = -1.0, 0, []
    with mlflow.start_run(experiment_id=experiment_id, run_name=output.name) as run:
        mlflow.log_params({k: v for k, v in config.items() if not isinstance(v, list)})
        config["mlflow_run_id"] = run.info.run_id
        save_json(output / "config.json", config)
        for epoch in range(1, config["epochs_requested"] + 1):
            train_loss, _, _ = run_epoch(model, loaders["train"], optimizer, scaler)
            val_loss, labels, probabilities = run_epoch(model, loaders["validation"])
            metrics, _ = metrics_for(labels, probabilities)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }
            history.append(record)
            write_csv(output / "history.csv", history)
            mlflow.log_metrics(
                {k: v for k, v in record.items() if k != "epoch"}, step=epoch
            )
            print(
                f"Epoch {epoch}: train loss={train_loss:.4f}, validation loss={val_loss:.4f}, "
                f"validation macro F1={metrics['macro_f1']:.4f}",
                flush=True,
            )
            if metrics["macro_f1"] > best_score:
                best_score, stale = metrics["macro_f1"], 0
                torch.save(
                    {
                        "model_state": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "config": config,
                        "epoch": epoch,
                    },
                    output / "best_model.pt",
                )
            else:
                stale += 1
            if stale >= config["patience"]:
                print(
                    "Early stopping: no macro F1 improvement for five epochs.",
                    flush=True,
                )
                break
        checkpoint = torch.load(
            output / "best_model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model_state"])
        _, labels, probabilities = run_epoch(model, loaders["validation"])
        metrics, report = metrics_for(labels, probabilities)
        train_counts = Counter(r["label"] for r in parts["train"])
        majority = train_counts.most_common(1)[0][0]
        majority_prediction = np.full(len(labels), CLASSES.index(majority))
        baseline = {
            "class_selected_from_training": majority,
            "accuracy": float(accuracy_score(labels, majority_prediction)),
            "macro_f1": float(
                f1_score(
                    labels,
                    majority_prediction,
                    labels=list(range(4)),
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        result = {
            "status": "SMOKE_PASS" if args.smoke else "EXPERIMENTAL_BASELINE_COMPLETE",
            "scope": SCOPE,
            "best_epoch": checkpoint["epoch"],
            "epochs_completed": len(history),
            "validation": metrics,
            "majority_baseline": baseline,
            "classification_report": report,
            "test_evaluated": False,
        }
        save_json(output / "validation_metrics.json", result)
        predictions = []
        for row, probability in zip(parts["validation"], probabilities):
            prediction = {
                "path": row["path"],
                "label": row["label"],
                "predicted": CLASSES[int(probability.argmax())],
                "tumor_score": float(probability[:3].sum()),
                **{
                    f"score_{name}": float(probability[i])
                    for i, name in enumerate(CLASSES)
                },
            }
            predictions.append(prediction)
        write_csv(output / "validation_predictions.csv", predictions)
        save_plots(output, history, labels, probabilities)
        for filename in (
            "config.json",
            "validation_metrics.json",
            "history.csv",
            "validation_predictions.csv",
            "learning_curves.png",
            "confusion_matrix.png",
        ):
            mlflow.log_artifact(str(output / filename))
    marker = (
        ROOT
        / "runs/classifier"
        / ("latest_smoke.txt" if args.smoke else "latest_baseline.txt")
    )
    marker.write_text(str(output), encoding="utf-8")
    print(result["status"], flush=True)
    print("Results:", output, flush=True)
    print("Final test set was not evaluated.", flush=True)


def predict(args):
    output = Path(args.run).resolve()
    checkpoint = torch.load(
        output / "best_model.pt", map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    if config["classes"] != CLASSES or config["script_sha256"] != sha256(__file__):
        raise ValueError(
            "Class mapping or script changed. Use the saved training_script.py for this run."
        )
    source = Path(args.image).resolve()
    with (output / "split_snapshot.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows = list(csv.DictReader(stream))
    digest = sha256(source)
    # Only approved-for-development examples are allowed in this classifier baseline demo.
    matches = [r for r in rows if r["sha256"] == digest]
    if not matches or any(
        r["draft_split"] not in ("train", "validation") for r in matches
    ):
        raise ValueError(
            "Use a training or validation image from this run; test/review/new uploads are excluded here."
        )
    model = resnet18(weights=None, num_classes=4)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with Image.open(source) as image:
        tensor = prepare_image(image).unsqueeze(0)
    with torch.inference_mode():
        probability = model(tensor).softmax(dim=1)[0].numpy()
    tumor_score = float(probability[:3].sum())
    result = {
        "scope": SCOPE,
        "image": str(source),
        "four_class_prediction": CLASSES[int(probability.argmax())],
        "binary_prediction": "tumor" if tumor_score >= 0.5 else "no_tumor",
        "tumor_score": tumor_score,
        "threshold": 0.5,
        "scores": {name: float(probability[i]) for i, name in enumerate(CLASSES)},
        "note": "Model scores are uncalibrated; this is not a diagnosis or proof of no tumor.",
    }
    print(json.dumps(result, indent=2))
    save_json(output / "last_demo_prediction.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    fitting = commands.add_parser("train")
    fitting.add_argument("--experimental", action="store_true")
    fitting.add_argument("--smoke", action="store_true")
    fitting.add_argument("--epochs", type=int, default=20)
    fitting.add_argument("--batch-size", type=int, default=8)
    fitting.add_argument("--seed", type=int, default=42)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--run", required=True)
    prediction.add_argument("--image", required=True)
    args = parser.parse_args()
    if args.command == "check":
        inspect_split()
    elif args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
