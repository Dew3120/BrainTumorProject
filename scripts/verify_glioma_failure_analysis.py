"""Verify the completed glioma failure-decomposition artifact set."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EXPECTED_CHECKPOINT_SHA256 = (
    "16dc8745993eef959d0f2924c9147a45320673f6fc0b465ab08f8ad7b2f723de"
)
EXPECTED_SPLIT_SHA256 = (
    "d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28"
)
REQUIRED_DECOMPOSITION_FIELDS = {
    "dice",
    "iou",
    "reference_pixels",
    "predicted_pixels",
    "area_ratio",
    "overlap_pixels",
    "reference_component_count_raw",
    "prediction_component_count_raw",
    "prediction_largest_component_fraction",
    "reference_centroid_x",
    "reference_centroid_y",
    "prediction_centroid_x",
    "prediction_centroid_y",
    "centroid_distance",
    "reference_bbox_width",
    "reference_bbox_height",
    "prediction_bbox_width",
    "prediction_bbox_height",
    "bbox_overlap_iou",
    "reference_touches_image_edge",
    "prediction_touches_image_edge",
    "original_width",
    "original_height",
    "orientation",
    "edema_visual",
    "contrast_enhancement_visual",
    "ventricle_proximity_visual",
    "multifocal_visual",
    "visual_review_status",
}
ARTIFACTS = (
    "glioma_failure_decomposition.csv",
    "glioma_failure_findings.md",
    "glioma_boundary_sharpness.csv",
    "glioma_case_study.md",
    "glioma_case_study_pairs.png",
    "glioma_subgroup_analysis.md",
    "glioma_visual_review_annotations.csv",
    "glioma_failure_analysis_verification.json",
    "glioma_failure_review_01.png",
    "glioma_failure_review_02.png",
    "glioma_failure_review_03.png",
    "glioma_failure_review_04.png",
    "glioma_failure_review_05.png",
    "glioma_failure_review_06.png",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    decomposition = read_csv(AUDIT / "glioma_failure_decomposition.csv")
    sharpness = read_csv(AUDIT / "glioma_boundary_sharpness.csv")
    annotations = read_csv(AUDIT / "glioma_visual_review_annotations.csv")
    verification = json.loads(
        (AUDIT / "glioma_failure_analysis_verification.json").read_text(
            encoding="utf-8"
        )
    )
    if len(decomposition) != 60 or len(sharpness) != 204 or len(annotations) != 60:
        raise ValueError("Unexpected glioma artifact row counts")
    if not REQUIRED_DECOMPOSITION_FIELDS <= set(decomposition[0]):
        missing = sorted(REQUIRED_DECOMPOSITION_FIELDS - set(decomposition[0]))
        raise ValueError(f"Missing decomposition fields: {missing}")
    if any(row["label"] != "glioma" or float(row["dice"]) >= 0.50 for row in decomposition):
        raise ValueError("Decomposition contains a non-glioma or non-failure case")
    if any(row["visual_review_status"] != "REVIEWED" for row in decomposition):
        raise ValueError("Visual review is incomplete")
    if any("/test/" in row["path"].replace("\\", "/") for row in decomposition):
        raise ValueError("Original test path entered the development audit")
    if {row["path"] for row in decomposition} != {row["path"] for row in annotations}:
        raise ValueError("Visual annotation membership mismatch")
    if verification["status"] != "COMPLETE":
        raise ValueError("Analysis verification is not complete")
    if verification["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint identity mismatch")
    if verification["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Split identity mismatch")
    if verification["maximum_stored_metric_difference"] != 0.0:
        raise ValueError("Stored and recomputed metrics differ")
    if verification["test_images_accessed"] != 0:
        raise ValueError("Sealed test access was recorded")
    case_study = (AUDIT / "glioma_case_study.md").read_text(encoding="utf-8")
    if sum(line.startswith("## Pair ") for line in case_study.splitlines()) != 10:
        raise ValueError("Matched case study does not contain ten pairs")
    literature = (ROOT / "docs/glioma_literature_review.md").read_text(
        encoding="utf-8"
    )
    paper_rows = [
        line
        for line in literature.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().isdigit()
    ]
    if len(paper_rows) != 19:
        raise ValueError(f"Expected 19 literature rows, found {len(paper_rows)}")
    manifest = {
        "status": "PASS",
        "decomposition_cases": len(decomposition),
        "sharpness_cases": len(sharpness),
        "matched_pairs": 10,
        "literature_primary_papers": len(paper_rows),
        "test_images_accessed": 0,
        "artifacts": [
            {
                "path": f"audit/{name}",
                "sha256": sha256(AUDIT / name),
                "bytes": (AUDIT / name).stat().st_size,
            }
            for name in ARTIFACTS
        ]
        + [
            {
                "path": "docs/glioma_literature_review.md",
                "sha256": sha256(ROOT / "docs/glioma_literature_review.md"),
                "bytes": (ROOT / "docs/glioma_literature_review.md").stat().st_size,
            }
        ],
    }
    destination = AUDIT / "glioma_failure_artifact_manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
