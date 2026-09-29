import evaluate_x03_upenn_gbm as upenn
import numpy as np
from evaluate_x03_upenn_gbm import (
    boundary_metrics_physical,
    build_evaluation_rows,
    js_divergence,
    nonrepresentable_failure_statistics,
)


def test_build_evaluation_rows_preserves_axial_primary_membership() -> None:
    subjects = [
        {
            "subject_id": "sub-001",
            "patient_id": "sub-001",
            "canonical_shape": "8x9x3",
            "image_name": "image.nii.gz",
            "mask_name": "mask.nii.gz",
            "voxel_size_x_mm": "1",
            "voxel_size_y_mm": "2",
            "voxel_size_z_mm": "3",
        }
    ]
    tumor = [{"patient_id": "sub-001", "axial_index_ras": "1"}]
    rows = build_evaluation_rows(subjects, tumor)
    assert len(rows) == 3
    assert [row["reference_has_tumor"] for row in rows] == [False, True, False]
    assert rows[1]["case_id"] == "sub-001:axial:001"
    assert rows[1]["width"] == 8
    assert rows[1]["height"] == 9
    assert rows[1]["depth"] == 3


def test_boundary_metrics_are_exact_for_identical_masks() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 5:12] = True
    f1, hausdorff = boundary_metrics_physical(mask, mask, 2.0, 3.0)
    assert f1 == 1.0
    assert hausdorff == 0.0


def test_js_divergence_is_zero_for_equal_distributions() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert js_divergence(values, values, np.linspace(0, 1, 6)) == 0.0


def test_nonrepresentable_target_is_an_explicit_failure() -> None:
    values = nonrepresentable_failure_statistics(np.zeros((8, 8), dtype=bool))
    assert values["primary_failure_mode"] == "target_not_representable_at_128"
    assert values["target_not_representable_at_128"] is True
    assert values["reference_foreground_pixels"] == 0


def test_domain_bootstrap_accepts_distinct_full_and_representable_memberships(
    monkeypatch,
) -> None:
    monkeypatch.setattr(upenn, "BOOTSTRAPS", 10)
    full_rows = [
        {"patient_id": "p1"},
        {"patient_id": "p1"},
        {"patient_id": "p2"},
        {"patient_id": "p2"},
    ]
    representable_rows = full_rows[:3]
    full = upenn.distribution_gap(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.15, 0.25, 0.35]),
        np.linspace(0, 1, 6),
        full_rows,
    )
    representable = upenn.distribution_gap(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.15, 0.25, 0.35]),
        np.linspace(0, 1, 6),
        representable_rows,
    )
    assert full["patient_bootstrap_ci95"]
    assert representable["patient_bootstrap_ci95"]
