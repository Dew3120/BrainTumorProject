import finalize_e05_weighted as finalize


def test_normalized_path_is_platform_independent():
    assert finalize.normalized_path("A\\B\\Case.JPG") == "a/b/case.jpg"


def test_category_mean_uses_only_requested_category():
    rows = [
        {"category": "glioma", "candidate_dice": 0.2},
        {"category": "glioma", "candidate_dice": 0.6},
        {"category": "meningioma", "candidate_dice": 0.9},
        {"category": "pituitary", "candidate_dice": 0.7},
    ]
    means = finalize.mean_by_category(rows, "candidate_dice")
    assert means == {"glioma": 0.4, "meningioma": 0.9, "pituitary": 0.7}
