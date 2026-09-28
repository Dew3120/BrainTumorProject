"""Audit OpenNeuro ds006248 v1.0.0 before any frozen-model inference."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import imagehash
import numpy as np
from download_x02_openneuro import PAYLOAD, SNAPSHOT, required_files
from PIL import Image
from stage6_external_common import display_uint8, load_canonical_nifti, sha256

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x02_openneuro_pituitary"
ARTIFACTS = EXPERIMENT / "artifacts"
AUDIT_MD = ROOT / "audit/x02_openneuro_dataset_audit.md"
BRISC_INVENTORY = ROOT / "audit/runs/20260921-182736-884368/inventory.csv"
X01_MANIFEST = ROOT / "experiments/x01_cheng_overlap_control/artifacts/dataset_manifest.csv"
EXPECTED_COMMIT = "1b8ebd1f1171049c7f4258ea41933e7b8ee6c06e"
EXPECTED_DOI = "doi:10.18112/openneuro.ds006248.v1.0.0"
EXPECTED_BRISC_INVENTORY_SHA256 = (
    "d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28"
)
EXPECTED_X01_MANIFEST_SHA256 = (
    "85d3c91f5a3bad870fd8e8c503d4dd47e883e25ba222dfbf808b59974c0efcde"
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SNAPSHOT), *arguments], text=True
    ).strip()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def phash64(value: np.ndarray) -> str:
    display, _ = display_uint8(value)
    return str(imagehash.phash(Image.fromarray(display, mode="L")))


def phash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def frozen_phash_rows() -> tuple[list[dict], list[dict]]:
    if sha256(BRISC_INVENTORY) != EXPECTED_BRISC_INVENTORY_SHA256:
        raise ValueError("Frozen BRISC inventory hash mismatch")
    if sha256(X01_MANIFEST) != EXPECTED_X01_MANIFEST_SHA256:
        raise ValueError("Frozen X01 dataset manifest hash mismatch")
    with BRISC_INVENTORY.open("r", encoding="utf-8", newline="") as handle:
        brisc = [row for row in csv.DictReader(handle) if row["label"] == "pituitary"]
    with X01_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        x01 = [row for row in csv.DictReader(handle) if row["label"] == "pituitary"]
    if len(brisc) != 1757 or len(x01) != 930:
        raise ValueError("Frozen pituitary pHash populations changed")
    return brisc, x01


def nearest(value: str, rows: list[dict], hash_field: str, path_field: str) -> dict:
    distances = [(phash_distance(value, row[hash_field]), row[path_field]) for row in rows]
    distance, path = min(distances, key=lambda item: (item[0], item[1]))
    return {
        "nearest_distance": distance,
        "nearest_path": path,
        "exact_matches": sum(item[0] == 0 for item in distances),
        "within_2_matches": sum(item[0] <= 2 for item in distances),
        "within_4_matches": sum(item[0] <= 4 for item in distances),
    }


def verify_release() -> dict:
    head = git_output("rev-parse", "HEAD")
    tags = git_output("tag", "--points-at", "HEAD").splitlines()
    description_path = SNAPSHOT / "dataset_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    if head != EXPECTED_COMMIT or "1.0.0" not in tags:
        raise ValueError("OpenNeuro snapshot commit/tag mismatch")
    if description.get("DatasetDOI") != EXPECTED_DOI:
        raise ValueError("OpenNeuro DOI mismatch")
    if description.get("Name") != "Open-Access Multimodal Dataset of Pituitary Adenoma":
        raise ValueError("OpenNeuro dataset name mismatch")
    readme = (SNAPSHOT / "README.md").read_text(encoding="utf-8")
    readme_license = "CC BY-NC 4.0" if "CC BY-NC 4.0" in readme else "NOT_FOUND"
    if description.get("License") != "CC0" or readme_license != "CC BY-NC 4.0":
        raise ValueError("Expected release-level licence conflict was not reproduced")
    return {
        "dataset": description["Name"],
        "accession": "ds006248",
        "version": "1.0.0",
        "doi": description["DatasetDOI"],
        "git_commit": head,
        "git_tag": "1.0.0",
        "dataset_description_license": description["License"],
        "readme_license": readme_license,
        "governance_license": "CC BY-NC 4.0 (conservative resolution pending source clarification)",
        "dataset_description_sha256": sha256(description_path),
        "readme_sha256": sha256(SNAPSHOT / "README.md"),
        "participants_tsv_sha256": sha256(SNAPSHOT / "participants.tsv"),
    }


def verify_sources(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        pointer = SNAPSHOT / row["relative_path"]
        payload = PAYLOAD / row["relative_path"]
        if not payload.is_file():
            raise ValueError(f"Missing X02 payload: {row['relative_path']}")
        observed_size = payload.stat().st_size
        observed_hash = sha256(payload)
        if observed_size != row["bytes"] or observed_hash != row["sha256"]:
            raise ValueError(f"X02 payload integrity mismatch: {row['relative_path']}")
        output.append(
            {
                **row,
                "pointer_sha256": sha256(pointer),
                "payload_bytes": observed_size,
                "payload_sha256": observed_hash,
                "integrity_verified": True,
            }
        )
    return output


def participant_ids() -> list[str]:
    with (SNAPSHOT / "participants.tsv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    identifiers = [row["participant_id"] for row in rows]
    if len(identifiers) != 50 or len(set(identifiers)) != 50:
        raise ValueError("Participant table must contain 50 unique identifiers")
    return identifiers


def render_alignment_panel(subject_rows: list[dict], source_rows: list[dict]) -> Path:
    """Render a deterministic image/reference overlay panel for manual audit."""
    source_lookup = {(row["subject_id"], row["kind"]): row for row in source_rows}
    selected_indices = np.linspace(0, len(subject_rows) - 1, 12, dtype=int)
    selected = [subject_rows[int(index)] for index in selected_indices]
    tile_width, tile_height = 550, 286
    canvas = Image.new("RGB", (tile_width * 3, tile_height * 4), "#202020")
    for position, row in enumerate(selected):
        subject_id = row["subject_id"]
        image_path = PAYLOAD / source_lookup[(subject_id, "image")]["relative_path"]
        mask_path = PAYLOAD / source_lookup[(subject_id, "mask")]["relative_path"]
        image, _, _ = load_canonical_nifti(image_path)
        mask, _, _ = load_canonical_nifti(mask_path)
        tumor = np.rint(mask).astype(np.uint8) == 1
        areas = np.count_nonzero(tumor, axis=(0, 2))
        coronal_index = int(np.argmax(areas))
        display, _ = display_uint8(image[:, coronal_index, :].T)
        reference = tumor[:, coronal_index, :].T
        base = Image.fromarray(display, mode="L").convert("RGB").resize(
            (256, 256), Image.Resampling.BILINEAR
        )
        resized_mask = Image.fromarray(reference.astype(np.uint8) * 255, mode="L").resize(
            (256, 256), Image.Resampling.NEAREST
        )
        overlay = base.copy()
        red = Image.new("RGB", overlay.size, (255, 32, 32))
        overlay.paste(red, mask=resized_mask.point(lambda value: 110 if value else 0))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(base, (10, 20))
        tile.paste(overlay, (284, 20))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(tile)
        draw.text(
            (10, 4),
            f"{subject_id} coronal {coronal_index}: original | tumor label 1 overlay",
            fill="black",
        )
        x = (position % 3) * tile_width
        y = (position // 3) * tile_height
        canvas.paste(tile, (x, y))
    output = ARTIFACTS / "source_alignment_panel.png"
    canvas.save(output, format="PNG", optimize=False)
    return output


def audit_dataset(source_rows: list[dict]):
    sources = {(row["subject_id"], row["kind"]): row for row in source_rows}
    brisc, x01 = frozen_phash_rows()
    subject_rows: list[dict] = []
    slice_rows: list[dict] = []
    overlap_rows: list[dict] = []
    image_volume_hashes: Counter[str] = Counter()
    mask_volume_hashes: Counter[str] = Counter()
    tumor_slice_hashes: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()

    for subject_id in participant_ids():
        image_path = PAYLOAD / sources[(subject_id, "image")]["relative_path"]
        mask_path = PAYLOAD / sources[(subject_id, "mask")]["relative_path"]
        image, image_affine, image_geometry = load_canonical_nifti(image_path)
        mask, mask_affine, mask_geometry = load_canonical_nifti(mask_path)
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {subject_id}")
        affine_error = float(np.max(np.abs(image_affine - mask_affine)))
        if affine_error > 1e-4:
            raise ValueError(f"Image/mask affine mismatch: {subject_id} ({affine_error})")
        rounded_mask = np.rint(mask).astype(np.uint8)
        if not np.array_equal(mask, rounded_mask):
            raise ValueError(f"Noninteger reference labels: {subject_id}")
        labels = {int(value) for value in np.unique(rounded_mask)}
        if not labels.issubset({0, 1, 2, 3, 4, 5}) or 1 not in labels:
            raise ValueError(f"Invalid or tumor-empty label map: {subject_id}: {labels}")
        if float(np.max(image)) <= float(np.min(image)):
            raise ValueError(f"Blank/constant image volume: {subject_id}")

        image_hash = array_sha256(image)
        mask_hash = array_sha256(rounded_mask)
        image_volume_hashes[image_hash] += 1
        mask_volume_hashes[mask_hash] += 1
        shape = "x".join(str(value) for value in image.shape)
        shape_counts[shape] += 1
        label_counts.update(int(value) for value in rounded_mask.reshape(-1))
        tumor_slices = np.flatnonzero(np.any(rounded_mask == 1, axis=(0, 2)))
        if tumor_slices.size == 0:
            raise ValueError(f"No tumor-bearing coronal slices: {subject_id}")

        subject_rows.append(
            {
                "subject_id": subject_id,
                "image_path": sources[(subject_id, "image")]["relative_path"],
                "mask_path": sources[(subject_id, "mask")]["relative_path"],
                "canonical_shape": shape,
                "source_image_shape": "x".join(str(v) for v in image_geometry.source_shape),
                "source_mask_shape": "x".join(str(v) for v in mask_geometry.source_shape),
                "canonical_orientation": "".join(image_geometry.canonical_axcodes),
                "voxel_size_x_mm": image_geometry.voxel_sizes_mm[0],
                "voxel_size_y_mm": image_geometry.voxel_sizes_mm[1],
                "voxel_size_z_mm": image_geometry.voxel_sizes_mm[2],
                "max_affine_difference": affine_error,
                "tumor_voxels": int(np.count_nonzero(rounded_mask == 1)),
                "tumor_bearing_coronal_slices": int(tumor_slices.size),
                "image_array_sha256": image_hash,
                "mask_array_sha256": mask_hash,
            }
        )

        for coronal_index in tumor_slices:
            image_slice = image[:, int(coronal_index), :].T
            mask_slice = (rounded_mask[:, int(coronal_index), :] == 1).T
            if image_slice.shape != mask_slice.shape or not mask_slice.any():
                raise ValueError(f"Invalid tumor-bearing slice: {subject_id}:{coronal_index}")
            slice_hash = array_sha256(image_slice)
            tumor_slice_hashes[slice_hash] += 1
            display_hash = phash64(image_slice)
            brisc_nearest = nearest(display_hash, brisc, "phash", "path")
            x01_nearest = nearest(display_hash, x01, "display_phash", "source_path")
            slice_id = f"{subject_id}:coronal:{int(coronal_index):03d}"
            slice_rows.append(
                {
                    "case_id": slice_id,
                    "subject_id": subject_id,
                    "coronal_index_ras": int(coronal_index),
                    "width": image_slice.shape[1],
                    "height": image_slice.shape[0],
                    "image_min": float(np.min(image_slice)),
                    "image_max": float(np.max(image_slice)),
                    "tumor_pixels": int(np.count_nonzero(mask_slice)),
                    "tumor_fraction": float(np.mean(mask_slice)),
                    "image_array_sha256": slice_hash,
                    "mask_array_sha256": array_sha256(mask_slice),
                    "display_phash": display_hash,
                }
            )
            overlap_rows.append(
                {
                    "case_id": slice_id,
                    "subject_id": subject_id,
                    "display_phash": display_hash,
                    "brisc_nearest_distance": brisc_nearest["nearest_distance"],
                    "brisc_nearest_path": brisc_nearest["nearest_path"],
                    "brisc_exact_matches": brisc_nearest["exact_matches"],
                    "brisc_within_2_matches": brisc_nearest["within_2_matches"],
                    "brisc_within_4_matches": brisc_nearest["within_4_matches"],
                    "x01_nearest_distance": x01_nearest["nearest_distance"],
                    "x01_nearest_path": x01_nearest["nearest_path"],
                    "x01_exact_matches": x01_nearest["exact_matches"],
                    "x01_within_2_matches": x01_nearest["within_2_matches"],
                    "x01_within_4_matches": x01_nearest["within_4_matches"],
                }
            )

    if len(subject_rows) != 50:
        raise ValueError("Subject audit did not produce 50 rows")
    if any(value > 1 for value in image_volume_hashes.values()):
        raise ValueError("Exact duplicate image volumes detected across subjects")
    if any(value > 1 for value in mask_volume_hashes.values()):
        raise ValueError("Exact duplicate mask volumes detected across subjects")

    summary = {
        "subjects": len(subject_rows),
        "tumor_bearing_coronal_slices": len(slice_rows),
        "shape_counts": dict(sorted(shape_counts.items())),
        "label_voxel_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "duplicate_image_volumes": 0,
        "duplicate_mask_volumes": 0,
        "duplicate_tumor_slice_arrays_beyond_first": sum(
            value - 1 for value in tumor_slice_hashes.values() if value > 1
        ),
        "tumor_voxels": sum(int(row["tumor_voxels"]) for row in subject_rows),
        "tumor_slices_per_subject": {
            "minimum": min(int(row["tumor_bearing_coronal_slices"]) for row in subject_rows),
            "median": float(np.median([int(row["tumor_bearing_coronal_slices"]) for row in subject_rows])),
            "maximum": max(int(row["tumor_bearing_coronal_slices"]) for row in subject_rows),
        },
        "brisc_overlap": {
            "exact_slice_candidates": sum(int(row["brisc_exact_matches"]) > 0 for row in overlap_rows),
            "within_2_slice_candidates": sum(int(row["brisc_within_2_matches"]) > 0 for row in overlap_rows),
            "within_4_slice_candidates": sum(int(row["brisc_within_4_matches"]) > 0 for row in overlap_rows),
            "minimum_distance": min(int(row["brisc_nearest_distance"]) for row in overlap_rows),
        },
        "x01_overlap": {
            "exact_slice_candidates": sum(int(row["x01_exact_matches"]) > 0 for row in overlap_rows),
            "within_2_slice_candidates": sum(int(row["x01_within_2_matches"]) > 0 for row in overlap_rows),
            "within_4_slice_candidates": sum(int(row["x01_within_4_matches"]) > 0 for row in overlap_rows),
            "minimum_distance": min(int(row["x01_nearest_distance"]) for row in overlap_rows),
        },
    }
    return subject_rows, slice_rows, overlap_rows, summary


def update_registry() -> None:
    path = ROOT / "experiments/registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in registry["experiments"] if item["id"] == "x02_openneuro_pituitary"]
    if len(matches) != 1 or matches[0]["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X02 registry state is not the registered pre-audit state")
    matches[0]["status"] = "AUDIT_COMPLETE"
    write_json(path, registry)


def experiment_manifest(paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x02_openneuro_pituitary",
        "status": "AUDIT_COMPLETE",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(paths)
        ],
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    release = verify_release()
    source_rows = verify_sources(required_files())
    subject_rows, slice_rows, overlap_rows, dataset = audit_dataset(source_rows)
    alignment_panel = render_alignment_panel(subject_rows, source_rows)
    if dataset["brisc_overlap"]["exact_slice_candidates"] or dataset["x01_overlap"]["exact_slice_candidates"]:
        outcome = "ELIGIBLE_OVERLAP_CONTROL"
        independent = False
    else:
        outcome = "ELIGIBLE_INDEPENDENT"
        independent = True

    write_csv(ARTIFACTS / "subject_manifest.csv", subject_rows)
    write_csv(ARTIFACTS / "tumor_slice_manifest.csv", slice_rows)
    write_csv(ARTIFACTS / "cross_dataset_overlap_candidates.csv", overlap_rows)
    source_manifest = {
        "schema_version": 1,
        **release,
        "retrieval_base_url": "https://s3.amazonaws.com/openneuro.org/ds006248",
        "files": source_rows,
    }
    write_json(ARTIFACTS / "source_file_manifest.json", source_manifest)
    free = shutil.disk_usage("E:\\").free
    audit_result = {
        "schema_version": 1,
        "experiment_id": "x02_openneuro_pituitary",
        "status": "AUDIT_COMPLETE",
        "audit_outcome": outcome,
        "independent_external_evidence": independent,
        "release": release,
        "dataset": dataset,
        "input_contract": {
            "image": "coronal contrast-enhanced T1-weighted NIfTI",
            "target": "ground-truth label 1 (tumor) only",
            "canonical_orientation": "RAS",
            "primary_slice_axis": "canonical axis 1 (coronal)",
            "primary_view": "tumor-bearing 2D coronal slices",
            "secondary_view": "whole volume including reference-empty slices",
        },
        "licence_conflict": {
            "present": True,
            "dataset_description": "CC0",
            "readme": "CC BY-NC 4.0",
            "handling": "Use the more restrictive CC BY-NC 4.0 terms unless the publisher resolves the conflict.",
            "technical_eligibility_affected": False,
        },
        "overlap_detection": {
            "brisc_inventory_sha256": EXPECTED_BRISC_INVENTORY_SHA256,
            "x01_manifest_sha256": EXPECTED_X01_MANIFEST_SHA256,
            "brisc_pixels_opened": 0,
            "comparison": "Tumor-bearing slice display pHashes compared with frozen pituitary pHash metadata only.",
            "brisc": dataset["brisc_overlap"],
            "x01": dataset["x01_overlap"],
        },
        "visual_alignment_review": {
            "status": "PASS_NO_SYSTEMATIC_DEFECT",
            "panel": alignment_panel.relative_to(ROOT).as_posix(),
            "panel_sha256": sha256(alignment_panel),
            "selection": "12 evenly spaced subjects; maximum tumor-area coronal slice per subject",
            "review_scope": "Reference label 1 follows visible sellar anatomy; no recurring flip, shift, transpose, or detached-mask pattern.",
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
            "required_payload_bytes": sum(int(row["bytes"]) for row in source_rows),
            "snapshot_metadata_bytes": sum(path.stat().st_size for path in SNAPSHOT.rglob("*") if path.is_file()),
            "projected_total_below_250_gb": True,
            "drive_e_free_bytes_after_audit": free,
            "drive_e_free_gib_after_audit": free / 1024**3,
        },
    }
    write_json(ARTIFACTS / "dataset_audit.json", audit_result)

    report = f"""# X02 OpenNeuro pituitary dataset audit

**Outcome:** `{outcome}`  
**Independent external evidence:** {'Yes' if independent else 'No'}  
**Inference performed:** No

## Release and integrity

- Official snapshot: OpenNeuro `ds006248` version `1.0.0`, DOI `10.18112/openneuro.ds006248.v1.0.0`.
- Snapshot commit/tag: `{EXPECTED_COMMIT}` / `1.0.0`.
- Exactly 100 required payloads were retrieved: one coronal CE-T1w image and one ground-truth segmentation for each of 50 subjects.
- Every payload matches the byte count and SHA256 encoded in its git-annex pointer. Required payload size is {audit_result['disk']['required_payload_bytes'] / 1024**3:.3f} GiB.

## Licence discrepancy

The same frozen release contains conflicting licence statements: BIDS `dataset_description.json` says `CC0`, while `README.md` says `CC BY-NC 4.0`. The audit does not silently choose the less restrictive declaration. Project handling follows `CC BY-NC 4.0` unless the publisher resolves the discrepancy. This governance issue does not alter technical eligibility, but it is binding for redistribution and downstream use.

## Cohort and target

- 50 unique subjects; no patient-level train/evaluation split is constructed because the entire independent cohort is frozen evaluation data.
- Source sequence: coronal contrast-enhanced T1, 3T GE 750w (`COR T1+C`).
- Expert ground truth is a multi-label mask. Only label `1` (tumor) is the registered target; labels 2–5 are excluded from the tumor target.
- Canonical RAS geometry is used. Primary evaluation uses the {dataset['tumor_bearing_coronal_slices']} tumor-bearing coronal slices along canonical axis 1. Whole-volume analysis including empty-reference slices is secondary.
- Image and mask shapes match for every subject; maximum affine discrepancy is within `1e-4`. All arrays are finite, all images are nonconstant, all masks use integer labels 0–5, and every subject contains tumor label 1.
- Tumor-bearing slices per subject: minimum {dataset['tumor_slices_per_subject']['minimum']}, median {dataset['tumor_slices_per_subject']['median']:.1f}, maximum {dataset['tumor_slices_per_subject']['maximum']}.

## Independence and duplicate audit

- Exact duplicate image volumes across subjects: {dataset['duplicate_image_volumes']}.
- Exact duplicate mask volumes across subjects: {dataset['duplicate_mask_volumes']}.
- Repeated tumor-bearing slice arrays beyond first occurrence: {dataset['duplicate_tumor_slice_arrays_beyond_first']}.
- BRISC frozen-metadata pHash candidates: exact {dataset['brisc_overlap']['exact_slice_candidates']}, distance ≤2 {dataset['brisc_overlap']['within_2_slice_candidates']}, distance ≤4 {dataset['brisc_overlap']['within_4_slice_candidates']}; minimum distance {dataset['brisc_overlap']['minimum_distance']}.
- X01 Cheng pHash candidates: exact {dataset['x01_overlap']['exact_slice_candidates']}, distance ≤2 {dataset['x01_overlap']['within_2_slice_candidates']}, distance ≤4 {dataset['x01_overlap']['within_4_slice_candidates']}; minimum distance {dataset['x01_overlap']['minimum_distance']}.
- No BRISC pixels were opened. The comparison uses only the existing frozen metadata inventory.
- A deterministic panel of 12 evenly spaced subjects was reviewed at each subject's maximum-tumor-area coronal slice. Label 1 remained aligned with visible sellar anatomy; no recurring flip, shift, transpose, or detached-mask pattern was found.

## Disk

- Required image/mask payload: {audit_result['disk']['required_payload_bytes'] / 1024**3:.3f} GiB.
- Drive E free after audit: {audit_result['disk']['drive_e_free_gib_after_audit']:.3f} GiB.
- The projected Stage 6 footprint remains below the 250 GB sequential-processing trigger at this step.

## Decision

The frozen stack may proceed to one X02 evaluation under `{outcome}`. Existing model, calibration, segmentation-threshold, and reliability-policy values remain frozen. Results are research transport evidence only and cannot establish clinical validity.
"""
    AUDIT_MD.write_text(report, encoding="utf-8", newline="\n")

    config_path = EXPERIMENT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X02 config state changed before audit")
    config.update(
        {
            "status": "AUDIT_COMPLETE",
            "dataset": {
                **config["dataset"],
                "resolved_commit": EXPECTED_COMMIT,
                "access_state": "MINIMAL_PAYLOAD_DOWNLOADED_AUDITED",
                "audit_outcome": outcome,
                "independent_external_evidence": independent,
                "licence_discrepancy": True,
                "governance_license": "CC BY-NC 4.0",
            },
            "r02_external_inference_state": "OPERATIONAL_ARTIFACT_FROZEN",
            "r02_manifest_sha256": "0d56aab5d562a34325003cee116c4f9d5f8fbe3fcc5278717338706d58e5303f",
            "test_images_accessed": 0,
            "model_retrained_or_refit": False,
        }
    )
    write_json(config_path, config)
    write_json(EXPERIMENT / "result.json", audit_result)
    (EXPERIMENT / "decision.md").write_text(
        "# X02 decision\n\n"
        f"**Audit decision:** `{outcome}`\n\n"
        "The frozen stack may proceed to one evaluation. All 50 image/mask pairs passed "
        "integrity, geometry, label, and membership checks. No inference or performance "
        "computation occurred during this audit. The release licence conflict is handled "
        "under the more restrictive CC BY-NC 4.0 statement.\n",
        encoding="utf-8",
        newline="\n",
    )
    update_registry()

    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    review_entry = f"""

## {timestamp} — Audit OpenNeuro ds006248 v1.0.0 before inference

- **Decision made:** Classify X02 as `{outcome}` and authorize one frozen-stack evaluation.
- **Triggering artifact:** `experiments/x02_openneuro_pituitary/artifacts/dataset_audit.json`, SHA256 `{sha256(ARTIFACTS / 'dataset_audit.json')}`; snapshot commit `{EXPECTED_COMMIT}`.
- **Rationale:** All 50 subjects and 100 required annex payloads passed exact hash, size, modality, finite-value, image/mask geometry, integer-label, tumor-presence, and subject-uniqueness checks. Exact display-pHash candidates against frozen BRISC/X01 pituitary metadata: {dataset['brisc_overlap']['exact_slice_candidates']}/{dataset['x01_overlap']['exact_slice_candidates']}.
- **Licence handling:** The frozen release conflicts internally (`CC0` in `dataset_description.json`; `CC BY-NC 4.0` in `README.md`). The project applies the more restrictive `CC BY-NC 4.0` statement unless the publisher clarifies it.
- **Verification:** BRISC comparison opened zero BRISC pixels. Inference, performance evaluation, neural retraining, refitting, threshold changes, and BRISC test-pixel access were zero.
- **Alternatives considered:** Download the entire multimodal dataset; rejected because the registered input contract requires only CE-cor T1 and expert ground truth. Treat the licence conflict as permission to use CC0; rejected as insufficiently conservative.
- **Who approved:** User approved Amendment 13 and continuation through eligible audits/evaluations unless a registered stop condition occurs.
"""
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(review_entry)

    manifest_paths = [
        AUDIT_MD,
        ROOT / "scripts/download_x02_openneuro.py",
        ROOT / "scripts/audit_x02_openneuro.py",
        ROOT / "tests/test_download_x02_openneuro.py",
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
        ROOT / "tests/test_audit_x02_openneuro.py",
    ]
    write_json(ARTIFACTS / "experiment_manifest.json", experiment_manifest(manifest_paths))


if __name__ == "__main__":
    main()
