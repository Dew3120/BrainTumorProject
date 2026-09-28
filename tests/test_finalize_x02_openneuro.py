import pytest

from scripts.finalize_x02_openneuro import as_bool, subject_mean, verify_target_losses


def test_as_bool_is_strict() -> None:
    assert as_bool("True") is True
    assert as_bool("False") is False
    with pytest.raises(ValueError):
        as_bool("false")


def test_subject_mean_weights_subjects_equally() -> None:
    rows = [
        {"patient_id": "a"},
        {"patient_id": "a"},
        {"patient_id": "b"},
    ]
    assert subject_mean([0.0, 1.0, 1.0], rows) == 0.75


def test_target_loss_contract_requires_exact_membership() -> None:
    rows = [
        {
            "case_id": case,
            "source_tumor_pixels": pixels,
            "post_resize_tumor_pixels": 0,
        }
        for case, pixels in {
            "sub-008:coronal:003": 10,
            "sub-015:coronal:013": 2,
            "sub-017:coronal:000": 9,
            "sub-025:coronal:008": 9,
            "sub-028:coronal:013": 7,
        }.items()
    ]
    verify_target_losses(rows)
