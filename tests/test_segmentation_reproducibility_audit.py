"""Reproducibility manifest and position-analysis tests."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segmentation_reproducibility_audit import (
    inventory_files,
    mask_position,
    position_summary,
    verify_inventory,
)


def test_manifest_inventory_is_sorted_hashed_and_self_excluding(tmp_path):
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "segmentation_run_manifest.json").write_text("self", encoding="utf-8")
    files = inventory_files(tmp_path)
    assert [item["path"] for item in files] == ["nested/a.txt", "z.txt"]
    assert all(len(item["sha256"]) == 64 for item in files)
    verify_inventory(tmp_path, files)


def test_mask_position_uses_normalized_image_coordinates():
    mask = np.zeros((10, 10), dtype=bool)
    mask[8:, 2:4] = True
    result = mask_position(mask)
    assert result["centroid_y_normalized"] > 0.9
    assert result["fraction_in_lower_half"] == 1.0
    assert result["fraction_in_lower_third"] == 1.0


def test_position_summary_counts_lower_centroids():
    items = [
        {
            "centroid_y_normalized": 0.4,
            "fraction_in_lower_half": 0.2,
            "fraction_in_lower_third": 0.0,
        },
        {
            "centroid_y_normalized": 0.7,
            "fraction_in_lower_half": 0.9,
            "fraction_in_lower_third": 0.6,
        },
    ]
    result = position_summary(items)
    assert result["count"] == 2
    assert result["centroid_below_image_midline_count"] == 1
    assert result["centroid_below_normalized_y_0_60_count"] == 1
    assert result["median_centroid_y_normalized"] == 0.55
