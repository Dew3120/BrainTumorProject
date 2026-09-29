from scripts.synthesize_stage6 import (
    ROOT,
    SOURCE_IDS,
    compact_cohort,
    read_json,
    sha256,
)


def test_access_blocked_cohort_has_no_performance_metrics() -> None:
    cohort = compact_cohort(
        "x04_meningioma_seg_class", {"status": "ACCESS_BLOCKED"}
    )
    assert cohort["status"] == "ACCESS_BLOCKED"
    assert cohort["performance_metrics"] == "NOT_ESTIMABLE"
    assert cohort["dataset_images_accessed"] == 0


def test_completed_synthesis_matches_frozen_source_results() -> None:
    synthesis = read_json(
        ROOT / "experiments/x05_cross_dataset_synthesis/artifacts/synthesis.json"
    )
    assert synthesis["h8"]["verdict"] == "REJECTED_FOR_EVALUATED_INDEPENDENT_COHORTS"
    assert synthesis["verification"]["pooled_cross_dataset_metric"] is False
    assert synthesis["verification"]["brisc_sealed_test_images_accessed"] == 0
    for experiment_id in SOURCE_IDS:
        source_path = ROOT / "experiments" / experiment_id / "result.json"
        assert synthesis["source_artifacts"][experiment_id]["sha256"] == sha256(
            source_path
        )
        assert synthesis["cohorts"][experiment_id] == compact_cohort(
            experiment_id, read_json(source_path)
        )
