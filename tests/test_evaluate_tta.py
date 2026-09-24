import evaluate_tta as tta
import torch


def test_all_eight_transforms_round_trip_exactly():
    source = torch.arange(49).reshape(1, 1, 7, 7)
    for name in tta.TRANSFORMS:
        transformed = tta.apply_transform(source, name)
        restored = tta.invert_transform(transformed, name)
        assert torch.equal(restored, source), name


def test_all_eight_transforms_are_distinct_for_asymmetric_grid():
    source = torch.zeros((1, 1, 7, 7), dtype=torch.int64)
    source[0, 0, 0, 1] = 1
    source[0, 0, 2, 5] = 2
    source[0, 0, 6, 3] = 3
    outputs = {
        tuple(tta.apply_transform(source, name).flatten().tolist())
        for name in tta.TRANSFORMS
    }
    assert len(outputs) == 8


def test_gate_is_conjunctive():
    gate = {
        "glioma_mean_dice_delta_minimum": 0.03,
        "meningioma_mean_dice_delta_minimum": -0.01,
        "pituitary_mean_dice_delta_minimum": -0.01,
    }
    checks = tta.gate_checks(
        {"glioma": 0.029, "meningioma": 0.02, "pituitary": 0.01}, gate
    )
    assert checks == {"glioma": False, "meningioma": True, "pituitary": True}
    assert not all(checks.values())
