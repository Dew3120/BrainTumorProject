"""Materialize one verified E08 probability cache for calibration studies."""

import csv
import hashlib
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import segmentation_smoke_test as smoke
import segmentation_system as system
import torch
import train_segmenter as train
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
E08 = ROOT / "experiments/e08_ensemble"
BASELINE_RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
COMPONENT_RUNS = {
    "e05_glioma_weighted": ROOT
    / "runs/segmentation_v2_weighted/20260924-055049-892543-weighted",
    "e06_boundary_loss": ROOT
    / "runs/segmentation_v2_boundary_loss/20260924-063051-647842-boundary",
    "e07_glioma_specialized": ROOT
    / "runs/segmentation_v2_glioma_specialized/20260924-105229-241365-specialist",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


def hard_scores(probability, target):
    hard = probability >= 0.5
    intersection = (hard & target).sum(axis=(1, 2), dtype=np.int64)
    predicted = hard.sum(axis=(1, 2), dtype=np.int64)
    reference = target.sum(axis=(1, 2), dtype=np.int64)
    dice = 2 * intersection / np.maximum(predicted + reference, 1)
    iou = intersection / np.maximum(predicted + reference - intersection, 1)
    return dice, iou


def main():
    e08_config = read_json(E08 / "config.json")
    e08_result = read_json(E08 / "result.json")
    if e08_result["status"] != "PASS" or not e08_result["gate_passed"]:
        raise ValueError("E08 is not the locked experimental v2 candidate")
    split = BASELINE_RUN / "split_snapshot.csv"
    if sha256(split) != e08_config["split_sha256"]:
        raise ValueError("E08 calibration-cache split mismatch")
    rows = train.partition_masked_rows(smoke.read_csv(split))["validation"]
    if len(rows) != 680:
        raise ValueError("E08 calibration-cache population mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_config = {
        item["experiment_id"]: item for item in e08_config["components"]
    }
    models = {}
    for name, run in COMPONENT_RUNS.items():
        result = read_json(run / "result.json")
        selected = result["top3"][0]
        checkpoint = run / "top3_checkpoints" / selected["checkpoint"]
        models[name] = system.load_selected_system(
            run, checkpoint, device, component_config[name]["checkpoint_sha256"]
        )
    dataset = train.FullPairs(rows)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    probabilities = np.empty((680, 128, 128), dtype=np.float32)
    targets = np.empty((680, 128, 128), dtype=bool)
    labels = np.empty(680, dtype=np.int8)
    with torch.inference_mode():
        for images, batch_targets, batch_labels, indices in loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_rows = [rows[index] for index in indices.tolist()]
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                maps = [
                    system.logits_for_rows(model, images, batch_rows).float().sigmoid()
                    for model in models.values()
                ]
            average = sum(maps) / 3.0
            for position, index in enumerate(indices.tolist()):
                probabilities[index] = average[position, 0].cpu().numpy()
                targets[index] = batch_targets[position, 0].cpu().numpy() >= 0.5
                labels[index] = int(batch_labels[position].item())
    dice, iou = hard_scores(probabilities, targets)
    expected = {
        normalized_path(row["path"]): row
        for row in read_csv(E08 / "artifacts/per_case_comparison.csv")
    }
    maximum_difference = 0.0
    for index, row in enumerate(rows):
        old = expected[normalized_path(row["path"])]
        maximum_difference = max(
            maximum_difference,
            abs(float(dice[index]) - float(old["ensemble_dice"])),
            abs(float(iou[index]) - float(old["ensemble_iou"])),
        )
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    if maximum_difference > tolerance:
        raise ValueError(f"E08 cache metric mismatch: {maximum_difference}")
    output = (
        ROOT
        / "runs/segmentation_v2_ensemble"
        / (datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "-calibration-cache")
    )
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "ensemble_probabilities_targets.npz"
    np.savez_compressed(cache, probabilities=probabilities, targets=targets, labels=labels)
    order_rows = [
        {
            "index": index,
            "path": row["path"],
            "label": row["label"],
            "similarity_group": row["similarity_group"],
            "dice": float(dice[index]),
            "iou": float(iou[index]),
        }
        for index, row in enumerate(rows)
    ]
    order = output / "validation_order.csv"
    write_csv(order, order_rows)
    config = {
        "schema_version": 1,
        "kind": "e08_equal_probability_ensemble_calibration_cache",
        "source_experiment_result_sha256": sha256(E08 / "result.json"),
        "source_experiment_manifest_sha256": sha256(
            E08 / "artifacts/experiment_manifest.json"
        ),
        "split_sha256": sha256(split),
        "component_checkpoint_sha256": {
            name: component_config[name]["checkpoint_sha256"]
            for name in COMPONENT_RUNS
        },
        "e07_routing_sha256": component_config["e07_glioma_specialized"][
            "routing_sha256"
        ],
        "validation_cases": 680,
        "shape": [680, 128, 128],
        "probability_dtype": str(probabilities.dtype),
        "target_dtype": str(targets.dtype),
        "maximum_recomputed_metric_difference": maximum_difference,
        "inference_device": str(device),
        "test_images_accessed": 0,
    }
    write_json(output / "config.json", config)
    manifest = {
        "schema_version": 1,
        "run": str(output),
        "files": [
            {
                "path": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (cache, order, output / "config.json")
        ],
        "test_images_accessed": 0,
    }
    write_json(output / "manifest.json", manifest)
    latest = ROOT / "runs/segmentation_v2_ensemble/latest.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(output), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
