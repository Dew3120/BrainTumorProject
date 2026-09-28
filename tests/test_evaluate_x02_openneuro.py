import numpy as np

from scripts.evaluate_x02_openneuro import (
    boundary_metrics_physical,
    build_evaluation_rows,
    js_divergence,
)


def test_build_evaluation_rows_preserves_primary_membership() -> None:
    subjects = [
        {
            "subject_id": "sub-001",
            "canonical_shape": "8x3x8",
            "image_path": "image.nii.gz",
            "mask_path": "mask.nii.gz",
            "voxel_size_x_mm": "1",
            "voxel_size_y_mm": "2",
            "voxel_size_z_mm": "1",
        }
    ]
    tumor = [{"subject_id": "sub-001", "coronal_index_ras": "1"}]
    rows = build_evaluation_rows(subjects, tumor)
    assert len(rows) == 3
    assert [row["reference_has_tumor"] for row in rows] == [False, True, False]


def test_boundary_metrics_are_exact_for_identical_masks() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 5:12] = True
    f1, hausdorff = boundary_metrics_physical(mask, mask, 2.0, 3.0)
    assert f1 == 1.0
    assert hausdorff == 0.0


def test_js_divergence_is_zero_for_equal_distributions() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert js_divergence(values, values, np.linspace(0, 1, 6)) == 0.0
