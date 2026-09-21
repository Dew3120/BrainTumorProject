"""Regression checks for the Phase 3 pipeline; integration uses synthetic images."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/phase3_classifier.py"
spec = importlib.util.spec_from_file_location("phase3_classifier", SCRIPT)
phase3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase3)


def test_preprocessing_preserves_aspect_ratio_with_padding():
    image = Image.new("L", (80, 40), 255)
    value = phase3.prepare_image(image)
    assert value.shape == (3, 224, 224)
    unnormalized = (
        value * torch.tensor(phase3.STD)[:, None, None]
        + torch.tensor(phase3.MEAN)[:, None, None]
    )
    assert torch.allclose(unnormalized[:, 0, :], torch.zeros(3, 224), atol=1e-6)
    assert torch.allclose(unnormalized[:, 112, :], torch.ones(3, 224), atol=1e-6)


def test_binary_prediction_uses_sum_of_tumor_classes():
    probabilities = np.array(
        [
            [0.2, 0.2, 0.2, 0.4],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    metrics, _ = phase3.metrics_for(np.arange(4), probabilities)
    assert metrics["accuracy"] == 0.75
    assert metrics["tumor_sensitivity"] == 1.0
    assert metrics["no_tumor_specificity"] == 1.0
    assert metrics["false_negatives"] == 0


def test_image_path_rejects_outside_dataset():
    with pytest.raises(ValueError, match="outside"):
        phase3.image_path({"path": "../outside.jpg"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_synthetic_training_checkpoint_reload_and_test_rejection(tmp_path, monkeypatch):
    rows = []
    generator = np.random.default_rng(7)
    for split in ("train", "validation"):
        for label in phase3.CLASSES:
            for index in range(8):
                relative = f"data/raw/brisc/{split}_{label}_{index}.png"
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    generator.integers(0, 256, (64, 80), dtype=np.uint8)
                ).save(path)
                rows.append(
                    {
                        "path": relative,
                        "label": label,
                        "draft_split": split,
                        "sha256": phase3.sha256(path),
                    }
                )
    split_path = tmp_path / "split.csv"
    phase3.write_csv(split_path, rows)
    (tmp_path / "uv.lock").write_text("synthetic-test", encoding="utf-8")
    monkeypatch.setattr(phase3, "ROOT", tmp_path)
    monkeypatch.setattr(phase3, "SPLIT", split_path)
    monkeypatch.setattr(phase3, "inspect_split", lambda: rows)
    monkeypatch.setattr(
        phase3.subprocess, "check_output", lambda *a, **kw: "synthetic-test"
    )
    original_resnet = phase3.resnet18
    monkeypatch.setattr(
        phase3,
        "resnet18",
        lambda **kw: original_resnet(
            weights=None, num_classes=kw.get("num_classes", 1000)
        ),
    )
    original_cache = torch.hub.get_dir()
    try:
        phase3.train(
            argparse.Namespace(
                experimental=True, smoke=True, epochs=1, batch_size=8, seed=42
            )
        )
        output = Path(
            (tmp_path / "runs/classifier/latest_smoke.txt").read_text(encoding="utf-8")
        )
        result = json.loads(
            (output / "validation_metrics.json").read_text(encoding="utf-8")
        )
        assert result["status"] == "SMOKE_PASS"
        assert result["test_evaluated"] is False
        with (output / "validation_predictions.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            prediction = next(csv.DictReader(stream))
        arguments = argparse.Namespace(
            run=str(output), image=str(tmp_path / prediction["path"])
        )
        phase3.predict(arguments)
        demo = json.loads(
            (output / "last_demo_prediction.json").read_text(encoding="utf-8")
        )
        assert demo["tumor_score"] == pytest.approx(
            float(prediction["tumor_score"]), abs=0.003
        )
        assert (output / "learning_curves.png").stat().st_size > 0
        assert (output / "confusion_matrix.png").stat().st_size > 0
        # Even a copied test image must be blocked by its hash.
        rows.append(
            {
                "path": "test-copy.png",
                "label": "glioma",
                "draft_split": "test",
                "sha256": phase3.sha256(arguments.image),
            }
        )
        phase3.write_csv(output / "split_snapshot.csv", rows)
        with pytest.raises(ValueError, match="excluded"):
            phase3.predict(arguments)
    finally:
        torch.hub.set_dir(original_cache)
        torch.cuda.empty_cache()
