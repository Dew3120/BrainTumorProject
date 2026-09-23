"""Visual audit selection and rendering helpers must remain deterministic."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segmentation_visual_audit import normalized_path, overlay, select_cases


def record(label, index, dice):
    return {
        "path": f"data/{label}/{index:02d}.png",
        "filename": f"{label}-{index:02d}.png",
        "label": label,
        "dice": dice,
        "iou": dice / 2,
        "reference_pixels": 10,
        "predicted_pixels": 10,
    }


def test_case_selection_is_deterministic_and_keeps_requested_groups():
    rows = []
    for label in ("glioma", "meningioma", "pituitary"):
        rows.extend(record(label, index, index / 10) for index in range(8))
    rows[0]["dice"] = 0
    rows[8]["dice"] = 0
    rows[16]["dice"] = 0
    groups = select_cases(rows, 2, 3, 2)
    assert [row["label"] for row in groups["zero_dice"]] == [
        "glioma",
        "meningioma",
        "pituitary",
    ]
    assert [row["dice"] for row in groups["best_glioma"]] == [0.7, 0.6]
    assert len(groups["worst_meningioma"]) == 2
    assert normalized_path("DATA\\A.PNG") == "data/a.png"


def test_overlay_changes_only_masked_pixels():
    image = Image.fromarray(np.full((4, 4), 100, dtype=np.uint8))
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 2] = True
    result = np.asarray(overlay(image, Image.fromarray(mask), (255, 0, 0)))
    assert np.array_equal(result[0, 0], [100, 100, 100])
    assert result[1, 2, 0] > 100
    assert result[1, 2, 1] < 100
