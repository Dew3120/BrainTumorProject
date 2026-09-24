import evaluate_multiscale as multiscale
import torch


def test_resize_probability_preserves_constant_field():
    source = torch.full((2, 1, 256, 256), 0.37)
    resized = multiscale.resize_probability(source, 128)
    assert resized.shape == (2, 1, 128, 128)
    assert torch.allclose(resized, torch.full_like(resized, 0.37), atol=1e-7)


def test_combine_probabilities_uses_equal_arithmetic_weights():
    low = torch.full((1, 1, 128, 128), 0.2)
    high = torch.full((1, 1, 256, 256), 0.8)
    combined, resized = multiscale.combine_probabilities(low, high)
    assert torch.allclose(resized, torch.full_like(low, 0.8), atol=1e-7)
    assert torch.allclose(combined, torch.full_like(low, 0.5), atol=1e-7)


def test_gate_is_conjunctive():
    gate = {
        "glioma_mean_dice_delta_minimum": 0.02,
        "meningioma_mean_dice_delta_minimum": -0.01,
        "pituitary_mean_dice_delta_minimum": -0.01,
    }
    checks = multiscale.gate_checks(
        {"glioma": 0.021, "meningioma": -0.011, "pituitary": 0.0}, gate
    )
    assert checks == {"glioma": True, "meningioma": False, "pituitary": True}
    assert not all(checks.values())
