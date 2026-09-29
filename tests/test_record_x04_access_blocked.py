from scripts.record_x04_access_blocked import build_result
from scripts.validate_experiment_tracking import VALID_STATUSES


def test_access_blocked_result_contains_no_metrics_or_access() -> None:
    result = build_result(free_bytes=123, frozen_hashes={"x": "hash"})
    assert result["status"] == "ACCESS_BLOCKED"
    assert result["performance_metrics"].startswith("NOT_ESTIMABLE")
    assert result["dataset_files_accessed"] == 0
    assert result["dataset_images_accessed"] == 0
    assert result["brisc_sealed_test_images_accessed"] == 0
    assert result["segmentation_threshold"] == 0.5
    assert result["reliability_policy_threshold"] == "0.9490250173235835"


def test_access_blocked_is_a_registered_lifecycle_status() -> None:
    assert "ACCESS_BLOCKED" in VALID_STATUSES
