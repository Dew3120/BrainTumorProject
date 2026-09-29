import pytest

from scripts.finalize_x03_upenn_gbm import (
    ARTIFACTS,
    as_bool,
    subject_mean,
    verify_target_losses,
    x03,
)


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
    rows = x03.read_csv(ARTIFACTS / "target_not_representable.csv")
    verify_target_losses(rows)

    rows[0]["native_target_pixels"] = "99"
    with pytest.raises(ValueError, match="artifact changed"):
        verify_target_losses(rows)
