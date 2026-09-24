import segmentation_system as system
import torch


class ConstantModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, images):
        return torch.full(
            (len(images), 1, images.shape[-2], images.shape[-1]),
            self.value,
            device=images.device,
        )


def test_routed_segmenter_uses_only_frozen_path_assignment():
    routed = system.RoutedSegmenter(
        ConstantModel(3.0),
        ConstantModel(-2.0),
        {"DATA\\SPECIAL.JPG"},
    )
    images = torch.zeros((3, 1, 8, 8))
    rows = [
        {"path": "data/general-a.jpg"},
        {"path": "data/special.jpg"},
        {"path": "data/general-b.jpg"},
    ]

    logits = system.logits_for_rows(routed, images, rows)

    assert torch.all(logits[0] == -2.0)
    assert torch.all(logits[1] == 3.0)
    assert torch.all(logits[2] == -2.0)


def test_plain_segmenter_path_is_unchanged():
    images = torch.zeros((2, 1, 8, 8))
    rows = [{"path": "a"}, {"path": "b"}]
    logits = system.logits_for_rows(ConstantModel(1.25), images, rows)
    assert torch.all(logits == 1.25)
