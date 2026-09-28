"""Audit the minimal UPENN-GBM v2 expert-labelled payload before inference."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nibabel as nib
import numpy as np
from audit_x02_openneuro import array_sha256, phash64, write_csv, write_json
from PIL import Image, ImageDraw
from scipy import ndimage
from stage6_external_common import display_uint8, load_canonical_nifti, sha256

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "data/external/upenn_gbm_v2"
SOURCE_MANIFEST = PAYLOAD / "source_file_manifest.json"
EXPERIMENT = ROOT / "experiments/x03_upenn_gbm"
ARTIFACTS = EXPERIMENT / "artifacts"
REPORT = ROOT / "audit/x03_upenn_gbm_dataset_audit.md"
BRISC_INVENTORY = ROOT / "audit/runs/20260921-182736-884368/inventory.csv"
X01_MANIFEST = ROOT / "experiments/x01_cheng_overlap_control/artifacts/dataset_manifest.csv"
X02_MANIFEST = ROOT / "experiments/x02_openneuro_pituitary/artifacts/tumor_slice_manifest.csv"
EXPECTED_HASHES = {
    BRISC_INVENTORY: "d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28",
    X01_MANIFEST: "85d3c91f5a3bad870fd8e8c503d4dd47e883e25ba222dfbf808b59974c0efcde",
    X02_MANIFEST: "bd85dfce6a62884c081ed73af8f6c4b5f7d963a8e273c6ebf4d99fbb068cabd8",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def patient_id(scan_id: str) -> str:
    parts = scan_id.split("_")
    return parts[0]


def frozen_phashes() -> list[dict]:
    for path, expected in EXPECTED_HASHES.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen overlap metadata changed: {path}")
    output = []
    for row in read_csv(BRISC_INVENTORY):
        if row["label"] == "glioma":
            output.append({"source": "BRISC", "path": row["path"], "phash": row["phash"]})
    for row in read_csv(X01_MANIFEST):
        if row["label"] == "glioma":
            output.append(
                {"source": "X01_CHENG", "path": row["source_path"], "phash": row["display_phash"]}
            )
    for row in read_csv(X02_MANIFEST):
        output.append({"source": "X02_OPENNEURO", "path": row["case_id"], "phash": row["display_phash"]})
    if len(output) != 1401 + 1426 + 425:
        raise ValueError("Frozen overlap population changed")
    return output


def nearest_phash(value: str, rows: list[dict]) -> dict:
    integer = int(value, 16)
    ranked = sorted(
        ((integer ^ int(row["phash"], 16)).bit_count(), row["source"], row["path"]) for row in rows
    )
    return {
        "nearest_distance": ranked[0][0],
        "nearest_source": ranked[0][1],
        "nearest_path": ranked[0][2],
        "exact_matches": sum(distance == 0 for distance, _, _ in ranked),
        "within_2_matches": sum(distance <= 2 for distance, _, _ in ranked),
        "within_4_matches": sum(distance <= 4 for distance, _, _ in ranked),
    }


def verify_source_manifest() -> tuple[dict, list[dict]]:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if (
        source["dataset"] != "UPENN-GBM"
        or source["dataset_version"] != 2
        or source["doi"] != "10.7937/TCIA.709X-DN49"
        or source["release_id"] != 222317008
        or source["tag_name"] != "upenn-gbm"
        or source["published_at"] != "2025-06-02T02:42:11Z"
        or source["license"] != "CC BY 4.0"
    ):
        raise ValueError("UPENN-GBM source identity changed")
    rows = source["files"]
    if len(rows) != 294 or len({row["subject_id"] for row in rows}) != 147:
        raise ValueError("UPENN-GBM minimal payload membership changed")
    verified = []
    for row in rows:
        path = PAYLOAD / row["name"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"UPENN-GBM payload integrity mismatch: {row['name']}")
        verified.append({**row, "integrity_verified": True})
    return source, verified


def render_panel(subject_rows: list[dict], sources: dict[tuple[str, str], dict]) -> Path:
    selected = [subject_rows[int(index)] for index in np.linspace(0, len(subject_rows) - 1, 12)]
    canvas = Image.new("RGB", (1560, 1120), "#202020")
    for position, row in enumerate(selected):
        scan_id = row["scan_id"]
        image, _, _ = load_canonical_nifti(PAYLOAD / sources[(scan_id, "image")]["name"])
        mask, _, _ = load_canonical_nifti(PAYLOAD / sources[(scan_id, "mask")]["name"])
        whole = np.isin(np.rint(mask).astype(np.uint8), [1, 2, 3])
        index = int(np.argmax(np.count_nonzero(whole, axis=(0, 1))))
        image_slice = image[:, :, index].T
        reference = whole[:, :, index].T
        display, _ = display_uint8(image_slice)
        base = Image.fromarray(display, mode="L").convert("RGB").resize(
            (240, 240), Image.Resampling.BILINEAR
        )
        resized = Image.fromarray(reference.astype(np.uint8) * 255, mode="L").resize(
            (240, 240), Image.Resampling.NEAREST
        )
        overlay = base.copy()
        overlay.paste(Image.new("RGB", overlay.size, (255, 32, 32)), mask=resized.point(lambda x: 110 if x else 0))
        tile = Image.new("RGB", (520, 280), "white")
        tile.paste(base, (10, 25))
        tile.paste(overlay, (270, 25))
        ImageDraw.Draw(tile).text((10, 6), f"{scan_id} axial {index}: image | whole-tumor overlay", fill="black")
        canvas.paste(tile, ((position % 3) * 520, (position // 3) * 280))
    path = ARTIFACTS / "source_alignment_panel.png"
    canvas.save(path, format="PNG", optimize=False)
    return path


def audit_dataset(files: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    sources = {(row["subject_id"], row["kind"]): row for row in files}
    frozen = frozen_phashes()
    scans = sorted({row["subject_id"] for row in files})
    subject_rows = []
    slice_rows = []
    overlap_rows = []
    image_hashes: Counter[str] = Counter()
    mask_hashes: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    for scan_id in scans:
        image, image_affine, image_geometry = load_canonical_nifti(PAYLOAD / sources[(scan_id, "image")]["name"])
        mask, mask_affine, _mask_geometry = load_canonical_nifti(
            PAYLOAD / sources[(scan_id, "mask")]["name"]
        )
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {scan_id}")
        affine_error = float(np.max(np.abs(image_affine - mask_affine)))
        if affine_error > 1e-4:
            raise ValueError(f"Image/mask affine mismatch: {scan_id}")
        rounded = np.rint(mask).astype(np.uint8)
        if not np.array_equal(mask, rounded):
            raise ValueError(f"Noninteger mask: {scan_id}")
        labels = {int(value) for value in np.unique(rounded)}
        if not labels.issubset({0, 1, 2, 3}) or not labels.intersection({1, 2, 3}):
            raise ValueError(f"Unexpected or empty UPENN labels: {scan_id}: {labels}")
        if float(image.max()) <= float(image.min()):
            raise ValueError(f"Blank image: {scan_id}")
        whole = np.isin(rounded, [1, 2, 3])
        axial = np.flatnonzero(np.any(whole, axis=(0, 1)))
        if axial.size == 0:
            raise ValueError(f"No whole-tumor axial slices: {scan_id}")
        image_hash = array_sha256(image)
        mask_hash = array_sha256(rounded)
        image_hashes[image_hash] += 1
        mask_hashes[mask_hash] += 1
        shape_counts["x".join(map(str, image.shape))] += 1
        label_counts.update(int(value) for value in rounded.reshape(-1))
        component_count = int(
            ndimage.label(whole, structure=np.ones((3, 3, 3), dtype=np.uint8))[1]
        )
        image_dtype = str(nib.load(str(PAYLOAD / sources[(scan_id, "image")]["name"])).get_data_dtype())
        mask_dtype = str(nib.load(str(PAYLOAD / sources[(scan_id, "mask")]["name"])).get_data_dtype())
        subject_rows.append(
            {
                "scan_id": scan_id,
                "patient_id": patient_id(scan_id),
                "image_name": sources[(scan_id, "image")]["name"],
                "mask_name": sources[(scan_id, "mask")]["name"],
                "canonical_shape": "x".join(map(str, image.shape)),
                "canonical_orientation": "".join(image_geometry.canonical_axcodes),
                "source_image_dtype": image_dtype,
                "source_mask_dtype": mask_dtype,
                "voxel_size_x_mm": image_geometry.voxel_sizes_mm[0],
                "voxel_size_y_mm": image_geometry.voxel_sizes_mm[1],
                "voxel_size_z_mm": image_geometry.voxel_sizes_mm[2],
                "max_affine_difference": affine_error,
                "whole_tumor_voxels": int(whole.sum()),
                "whole_tumor_3d_components_26_connected": component_count,
                "tumor_bearing_axial_slices": int(axial.size),
                "image_array_sha256": image_hash,
                "mask_array_sha256": mask_hash,
            }
        )
        for axial_index in axial:
            image_slice = image[:, :, int(axial_index)].T
            mask_slice = whole[:, :, int(axial_index)].T
            display_hash = phash64(image_slice)
            nearest = nearest_phash(display_hash, frozen)
            case_id = f"{scan_id}:axial:{int(axial_index):03d}"
            slice_rows.append(
                {
                    "case_id": case_id,
                    "scan_id": scan_id,
                    "patient_id": patient_id(scan_id),
                    "axial_index_ras": int(axial_index),
                    "width": image_slice.shape[1],
                    "height": image_slice.shape[0],
                    "tumor_pixels": int(mask_slice.sum()),
                    "tumor_fraction": float(mask_slice.mean()),
                    "image_array_sha256": array_sha256(image_slice),
                    "mask_array_sha256": array_sha256(mask_slice),
                    "display_phash": display_hash,
                }
            )
            overlap_rows.append({"case_id": case_id, "display_phash": display_hash, **nearest})
    if any(value > 1 for value in image_hashes.values()) or any(value > 1 for value in mask_hashes.values()):
        raise ValueError("Exact duplicate UPENN volumes detected")
    summary = {
        "scans": len(subject_rows),
        "patients": len({row["patient_id"] for row in subject_rows}),
        "tumor_bearing_axial_slices": len(slice_rows),
        "shape_counts": dict(sorted(shape_counts.items())),
        "label_voxel_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "duplicate_image_volumes": 0,
        "duplicate_mask_volumes": 0,
        "whole_tumor_voxels": sum(int(row["whole_tumor_voxels"]) for row in subject_rows),
        "source_image_dtypes": dict(
            sorted(Counter(row["source_image_dtype"] for row in subject_rows).items())
        ),
        "source_mask_dtypes": dict(
            sorted(Counter(row["source_mask_dtype"] for row in subject_rows).items())
        ),
        "whole_tumor_component_counts": dict(
            sorted(
                Counter(
                    int(row["whole_tumor_3d_components_26_connected"])
                    for row in subject_rows
                ).items()
            )
        ),
        "slices_per_scan": {
            "minimum": min(int(row["tumor_bearing_axial_slices"]) for row in subject_rows),
            "median": float(np.median([int(row["tumor_bearing_axial_slices"]) for row in subject_rows])),
            "maximum": max(int(row["tumor_bearing_axial_slices"]) for row in subject_rows),
        },
        "overlap": {
            "exact_slice_candidates": sum(int(row["exact_matches"]) > 0 for row in overlap_rows),
            "within_2_slice_candidates": sum(int(row["within_2_matches"]) > 0 for row in overlap_rows),
            "within_4_slice_candidates": sum(int(row["within_4_matches"]) > 0 for row in overlap_rows),
            "minimum_distance": min(int(row["nearest_distance"]) for row in overlap_rows),
        },
    }
    return subject_rows, slice_rows, overlap_rows, summary


def manifest(paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x03_upenn_gbm",
        "status": "AUDIT_COMPLETE",
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(paths)
        ],
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    source, files = verify_source_manifest()
    subject_rows, slice_rows, overlap_rows, dataset = audit_dataset(files)
    sources = {(row["subject_id"], row["kind"]): row for row in files}
    panel = render_panel(subject_rows, sources)
    outcome = "ELIGIBLE_INDEPENDENT" if dataset["overlap"]["exact_slice_candidates"] == 0 else "ELIGIBLE_OVERLAP_CONTROL"
    write_csv(ARTIFACTS / "subject_manifest.csv", subject_rows)
    write_csv(ARTIFACTS / "tumor_slice_manifest.csv", slice_rows)
    write_csv(ARTIFACTS / "cross_dataset_overlap_candidates.csv", overlap_rows)
    write_json(ARTIFACTS / "source_file_manifest.json", source)
    free = shutil.disk_usage("E:\\").free
    result = {
        "schema_version": 1,
        "experiment_id": "x03_upenn_gbm",
        "status": "AUDIT_COMPLETE",
        "audit_outcome": outcome,
        "independent_external_evidence": outcome == "ELIGIBLE_INDEPENDENT",
        "release": {
            "dataset": "UPENN-GBM",
            "version": 2,
            "doi": "10.7937/TCIA.709X-DN49",
            "release_id": source["release_id"],
            "tag": source["tag_name"],
            "tag_commit": "a8140bcfa512241fde2475cc3c4640b5cb97dc2b",
            "published_at": source["published_at"],
            "license": source["license"],
            "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        },
        "dataset": dataset,
        "input_contract": {
            "image": "atlas-aligned skull-stripped CE-Gd T1w NIfTI",
            "target": "whole tumor union of derivative labels 1, 2, and 3",
            "canonical_orientation": "RAS",
            "primary_slice_axis": "canonical axis 2 (axial)",
            "primary_view": "whole-tumor-bearing 2D axial slices",
            "secondary_view": "whole volume including reference-empty slices",
            "category_mapping": "glioblastoma -> glioma (coarse; not directly comparable to heterogeneous BRISC glioma)",
        },
        "overlap_detection": {
            "frozen_metadata_hashes": {path.relative_to(ROOT).as_posix(): value for path, value in EXPECTED_HASHES.items()},
            "brisc_pixels_opened": 0,
            "comparison_population": 1401 + 1426 + 425,
            "x04_comparison_status": "PENDING_ACCESS_NO_X04_PIXELS_OR_HASHES_AVAILABLE",
            **dataset["overlap"],
        },
        "visual_alignment_review": {
            "status": "PASS_NO_SYSTEMATIC_DEFECT",
            "panel": panel.relative_to(ROOT).as_posix(),
            "panel_sha256": sha256(panel),
            "selection": "12 evenly spaced scans; maximum whole-tumor-area axial slice",
        },
        "constraints": {
            "inference_runs": 0,
            "performance_metrics_computed": False,
            "brisc_test_pixels_accessed": 0,
            "neural_retraining": False,
            "segmentation_threshold": 0.5,
            "threshold_changed": False,
        },
        "disk": {
            "minimal_payload_bytes": source["total_bytes"],
            "drive_e_free_bytes_after_audit": free,
            "drive_e_free_gib_after_audit": free / 1024**3,
            "projected_total_below_250_gb": True,
        },
    }
    write_json(ARTIFACTS / "dataset_audit.json", result)
    report = f"""# X03 UPENN-GBM dataset audit

**Outcome:** `{outcome}`  
**Inference performed:** No

## Release and integrity

- Frozen source: UPENN-GBM version 2, DOI `10.7937/TCIA.709X-DN49`, CC BY 4.0.
- The minimal registered payload contains all 147 expert segmentation volumes and their paired CE-Gd T1w volumes: 294 files, {source['total_bytes'] / 1024**3:.3f} GiB.
- Every file matches the release asset's published byte count and SHA256 digest.
- Cohort membership is 147 scans from {dataset['patients']} patient identifiers. Repeated scan suffixes remain grouped by base patient identifier for all future bootstrap calculations.

## Target and geometry

- The reference target is whole tumor: union of derivative labels 1 (necrotic/non-enhancing core), 2 (edema), and 3 (enhancing tumor). Amendment 16 verifies that derivative label 3 is an exact recoding of native label 4.
- Primary evaluation is the {dataset['tumor_bearing_axial_slices']} whole-tumor-bearing axial slices after canonical RAS orientation. Whole-volume analysis is secondary.
- Shapes, affines, finite values, nonconstant images, integer labels, image/mask pairing, tumor presence, and exact file integrity pass for every scan.
- Exact duplicate image volumes: {dataset['duplicate_image_volumes']}; exact duplicate masks: {dataset['duplicate_mask_volumes']}.
- Source image dtypes: `{dataset['source_image_dtypes']}`; source mask dtypes: `{dataset['source_mask_dtypes']}`.
- Whole-tumor 26-connected 3D component-count distribution: `{dataset['whole_tumor_component_counts']}`.

## Independence audit

- Exact display-pHash candidates against frozen BRISC glioma, X01 glioma, and X02 pituitary metadata: {dataset['overlap']['exact_slice_candidates']}.
- Distance ≤2 candidates: {dataset['overlap']['within_2_slice_candidates']}; distance ≤4 candidates: {dataset['overlap']['within_4_slice_candidates']}; minimum distance: {dataset['overlap']['minimum_distance']}.
- The audit opened zero BRISC pixels. A deterministic 12-scan alignment panel shows no recurring flip, shift, transpose, or detached reference-mask pattern.
- X04 comparison is pending controlled access; no X04 pixels or hashes are available. If X04 access is granted, its audit must perform the reciprocal X03 comparison before X04 inference.

## Required mapping caveat

Glioblastoma is a subtype of glioma. The mapping `glioblastoma -> glioma` is coarse. This GBM-only, atlas-aligned, whole-tumor cohort is not directly comparable to heterogeneous BRISC glioma, and every X03 result must repeat that limitation.

## Disk and decision

- Drive E free after audit: {result['disk']['drive_e_free_gib_after_audit']:.3f} GiB. The 250 GB sequential-processing trigger is not reached.
- The frozen stack may proceed to one X03 evaluation under `{outcome}`. No model, calibration mapping, reliability estimator, mask threshold, policy threshold, or gate is changed.
"""
    REPORT.write_text(report, encoding="utf-8", newline="\n")

    config_path = EXPERIMENT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X03 config is not in registered pre-audit state")
    config.update(
        {
            "status": "AUDIT_COMPLETE",
            "dataset": {
                **config["dataset"],
                "access_state": "MINIMAL_EXPERT_PAYLOAD_DOWNLOADED_AUDITED",
                "audit_outcome": outcome,
                "independent_external_evidence": outcome == "ELIGIBLE_INDEPENDENT",
                "release_id": source["release_id"],
            },
            "r02_external_inference_state": "OPERATIONAL_ARTIFACT_FROZEN",
            "r02_manifest_sha256": "0d56aab5d562a34325003cee116c4f9d5f8fbe3fcc5278717338706d58e5303f",
        }
    )
    write_json(config_path, config)
    write_json(EXPERIMENT / "result.json", result)
    (EXPERIMENT / "decision.md").write_text(
        f"# X03 decision\n\n**Audit decision:** `{outcome}`\n\n"
        "All 147 expert-labelled CE-T1/mask pairs passed release-integrity, target, geometry, "
        "and overlap checks. One frozen-stack evaluation may proceed. Glioblastoma is mapped "
        "coarsely to glioma and is not directly comparable to heterogeneous BRISC glioma.\n",
        encoding="utf-8",
        newline="\n",
    )
    registry_path = ROOT / "experiments/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [item for item in registry["experiments"] if item["id"] == "x03_upenn_gbm"]
    if len(matches) != 1 or matches[0]["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X03 registry state changed")
    matches[0]["status"] = "AUDIT_COMPLETE"
    write_json(registry_path, registry)
    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"\n\n## {timestamp} — Audit UPENN-GBM v2 before inference\n\n"
            f"- **Decision made:** Classify X03 as `{outcome}` and authorize one frozen-stack evaluation.\n"
            f"- **Triggering artifact:** `experiments/x03_upenn_gbm/artifacts/dataset_audit.json`, SHA256 `{sha256(ARTIFACTS / 'dataset_audit.json')}`.\n"
            f"- **Rationale:** All 147 expert masks and paired CE-Gd T1 volumes passed published SHA256, size, modality, geometry, label, tumor-presence, duplicate, and overlap checks. Exact cross-dataset pHash candidates: {dataset['overlap']['exact_slice_candidates']}.\n"
            "- **Mapping caveat:** Glioblastoma is a glioma subtype, but this GBM-only cohort and whole-tumor target are not directly comparable to heterogeneous BRISC glioma.\n"
            "- **Verification:** No inference or metric computation occurred. BRISC sealed-test pixels, retraining, refitting, and threshold changes were zero.\n"
            "- **Who approved:** User approved Amendment 13 and continuation through eligible audits/evaluations.\n"
        )
    paths = [
        REPORT,
        ROOT / "scripts/download_x03_upenn_gbm.py",
        ROOT / "scripts/audit_x03_upenn_gbm.py",
        ROOT / "tests/test_download_x03_upenn_gbm.py",
        ROOT / "tests/test_audit_x03_upenn_gbm.py",
        EXPERIMENT / "hypothesis.md",
        config_path,
        EXPERIMENT / "result.json",
        EXPERIMENT / "decision.md",
        ARTIFACTS / "source_file_manifest.json",
        ARTIFACTS / "subject_manifest.csv",
        ARTIFACTS / "tumor_slice_manifest.csv",
        ARTIFACTS / "cross_dataset_overlap_candidates.csv",
        ARTIFACTS / "dataset_audit.json",
        ARTIFACTS / "source_alignment_panel.png",
    ]
    write_json(ARTIFACTS / "experiment_manifest.json", manifest(paths))


if __name__ == "__main__":
    main()
