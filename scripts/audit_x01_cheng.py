"""Audit the Cheng/Figshare v8 cohort before any frozen-model inference."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import imagehash
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/cheng_figshare_v8/raw"
EXTRACTED = ROOT / "data/external/cheng_figshare_v8/extracted"
EXPERIMENT = ROOT / "experiments/x01_cheng_overlap_control"
ARTIFACTS = EXPERIMENT / "artifacts"
AUDIT_MD = ROOT / "audit/x01_cheng_dataset_audit.md"
BRISC_INVENTORY = ROOT / "audit/runs/20260921-182736-884368/inventory.csv"
EXPECTED_BRISC_INVENTORY_SHA256 = (
    "d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28"
)

LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
EXPECTED_LABEL_COUNTS = {"meningioma": 708, "glioma": 1426, "pituitary": 930}
EXPECTED_SOURCE_FILES = {
    "brainTumorDataPublic_1-766.zip": {
        "bytes": 214401279,
        "md5": "74b949ad33f042e6e103523091cd1428",
        "sha256": "cdac0a7f4152cb34dd79b13543da96c37a6928ad8cb823e2662150b1a73ade54",
    },
    "brainTumorDataPublic_1533-2298.zip": {
        "bytes": 215563856,
        "md5": "8227bf6080cb71f15a88be8d25c79ae7",
        "sha256": "612d9f506c55c54db5f2f38b6d50741ef97a6d7c32b3e1be76e789674a97aefb",
    },
    "brainTumorDataPublic_767-1532.zip": {
        "bytes": 217848429,
        "md5": "7e8a875500d2c8a346f270538e29890e",
        "sha256": "d18e896d14f7af791cdd3d9b9342f9f974742a06132effde5a5219209862b4ce",
    },
    "brainTumorDataPublic_2299-3064.zip": {
        "bytes": 231679762,
        "md5": "b378a80d6174e5317d59eb28430c6652",
        "sha256": "9a673c55d139133cfbc57097bf155b5ac9f33684ee97f40bf2afba332d416a3a",
    },
    "cvind.mat": {
        "bytes": 5736,
        "md5": "5ea82eb3212c9fcb84290ccf0beb486d",
        "sha256": "b4310122ee77e08cab74332c11d77ee4c34754fbcd9d5376cbbb88f7debbd5e8",
    },
    "README 2024.txt": {
        "bytes": 3303,
        "md5": "4188f7c30957e9673c3611613071bd6e",
        "sha256": "1fcb55d9dbc5c9b4d650e53fa8e606af356f0eefd3df32c04a2aaff1d237aa95",
    },
}
EXPECTED_METADATA_SHA256 = (
    "b02780e0c61b9150c8eac7ee090411f33f1b57ecf3088c0bb5e33cb68752d252"
)


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matlab_pid(value: np.ndarray) -> str:
    pid = "".join(chr(int(item)) for item in np.asarray(value).reshape(-1))
    if not pid or any(ord(char) < 32 for char in pid):
        raise ValueError("Invalid MATLAB patient identifier")
    return pid


def minmax_uint8(value: np.ndarray) -> np.ndarray:
    numeric = np.asarray(value, dtype=np.float64)
    if numeric.ndim != 2 or not np.isfinite(numeric).all():
        raise ValueError("Image must be a finite two-dimensional array")
    lower = float(numeric.min())
    upper = float(numeric.max())
    if upper <= lower:
        raise ValueError("Blank/constant image")
    return np.rint(np.clip((numeric - lower) / (upper - lower), 0, 1) * 255).astype(
        np.uint8
    )


def phash64(value: np.ndarray) -> str:
    return str(imagehash.phash(Image.fromarray(minmax_uint8(value), mode="L")))


def phash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def boundary_alignment_fraction(mask: np.ndarray, border: np.ndarray) -> float:
    """Return fraction of MATLAB (x,y) points within two pixels of mask boundary."""
    binary = np.asarray(mask, dtype=bool)
    outline = binary_dilation(binary) ^ binary_erosion(binary)
    pairs = np.asarray(border, dtype=np.float64).reshape(-1, 2)
    matches = 0
    for x_value, y_value in pairs:
        x_index = round(float(x_value)) - 1
        y_index = round(float(y_value)) - 1
        y0, y1 = max(0, y_index - 2), min(binary.shape[0], y_index + 3)
        x0, x1 = max(0, x_index - 2), min(binary.shape[1], x_index + 3)
        matches += int(outline[y0:y1, x0:x1].any())
    return matches / len(pairs)


def source_file_audit(metadata: dict) -> list[dict]:
    metadata_files = {item["name"]: item for item in metadata["files"]}
    rows = []
    for filename, expected in EXPECTED_SOURCE_FILES.items():
        path = RAW / filename
        if not path.is_file():
            raise ValueError(f"Missing source file: {filename}")
        observed = {
            "bytes": path.stat().st_size,
            "md5": digest(path, "md5"),
            "sha256": digest(path),
        }
        if observed != expected:
            raise ValueError(f"Source integrity mismatch: {filename}")
        meta = metadata_files.get(filename)
        if meta is None or int(meta["size"]) != expected["bytes"]:
            raise ValueError(f"Figshare metadata mismatch: {filename}")
        if meta["computed_md5"] != expected["md5"]:
            raise ValueError(f"Figshare MD5 mismatch: {filename}")
        rows.append(
            {
                "filename": filename,
                "bytes": expected["bytes"],
                "md5": expected["md5"],
                "sha256": expected["sha256"],
                "figshare_file_id": meta["id"],
                "download_url": meta["download_url"],
            }
        )
    return rows


def zip_member_audit() -> dict:
    names: list[str] = []
    for filename in sorted(EXPECTED_SOURCE_FILES):
        if not filename.endswith(".zip"):
            continue
        with zipfile.ZipFile(RAW / filename) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt ZIP member: {filename}:{bad}")
            names.extend(Path(item.filename).name for item in archive.infolist() if not item.is_dir())
    expected = {f"{index}.mat" for index in range(1, 3065)}
    observed = set(names)
    if len(names) != 3064 or observed != expected:
        raise ValueError("ZIP membership is incomplete, duplicated, or unexpected")
    extracted = {path.name for path in EXTRACTED.glob("*.mat")}
    if extracted != expected:
        raise ValueError("Extracted MATLAB membership differs from archives")
    return {"members": len(names), "unique_members": len(observed), "corrupt_members": 0}


def brisc_rows() -> list[dict]:
    if digest(BRISC_INVENTORY) != EXPECTED_BRISC_INVENTORY_SHA256:
        raise ValueError("Frozen BRISC inventory hash mismatch")
    with BRISC_INVENTORY.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6000 or any(not row["phash"] for row in rows):
        raise ValueError("Frozen BRISC inventory is incomplete")
    return rows


def nearest_brisc(source_phash: str, rows: list[dict], label: str) -> dict:
    best = None
    within_2 = 0
    within_4 = 0
    exact = 0
    for row in rows:
        if row["label"] != label:
            continue
        distance = phash_distance(source_phash, row["phash"])
        exact += int(distance == 0)
        within_2 += int(distance <= 2)
        within_4 += int(distance <= 4)
        key = (distance, row["path"])
        if best is None or key < best[0]:
            best = (key, row)
    if best is None:
        raise ValueError(f"No BRISC rows for label {label}")
    return {
        "nearest_phash_distance": best[0][0],
        "nearest_brisc_path": best[1]["path"],
        "nearest_brisc_original_split": best[1]["original_split"],
        "nearest_brisc_draft_split": best[1]["draft_split"],
        "exact_phash_matches": exact,
        "matches_within_distance_2": within_2,
        "matches_within_distance_4": within_4,
    }


def dataset_audit(cv_folds: np.ndarray, brisc: list[dict]):
    rows: list[dict] = []
    overlap: list[dict] = []
    patient_folds: dict[str, set[int]] = defaultdict(set)
    patient_labels: dict[str, set[str]] = defaultdict(set)
    image_hash_counts: Counter[str] = Counter()
    mask_hash_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    alignment_values: list[float] = []

    for index in range(1, 3065):
        path = EXTRACTED / f"{index}.mat"
        with h5py.File(path, "r") as handle:
            if set(handle.keys()) != {"cjdata"}:
                raise ValueError(f"Unexpected MAT structure: {path.name}")
            group = handle["cjdata"]
            required = {"PID", "image", "label", "tumorBorder", "tumorMask"}
            if set(group.keys()) != required:
                raise ValueError(f"Unexpected cjdata fields: {path.name}")
            pid = matlab_pid(np.asarray(group["PID"]))
            raw_label = float(np.asarray(group["label"]).reshape(-1)[0])
            if not raw_label.is_integer() or int(raw_label) not in LABELS:
                raise ValueError(f"Invalid label: {path.name}")
            label = LABELS[int(raw_label)]
            # MATLAB v7.3/HDF5 reverses MATLAB array dimensions. The transpose is
            # verified against every supplied (x,y) tumorBorder point below.
            image = np.asarray(group["image"]).T
            mask = np.asarray(group["tumorMask"]).T
            border = np.asarray(group["tumorBorder"]).reshape(-1)

        if image.ndim != 2 or mask.shape != image.shape:
            raise ValueError(f"Unexpected image/mask geometry: {path.name}")
        if not np.isfinite(image).all() or not np.isfinite(mask).all():
            raise ValueError(f"Non-finite image or mask: {path.name}")
        if float(image.max()) <= float(image.min()):
            raise ValueError(f"Blank/constant image: {path.name}")
        if not np.all(np.isin(np.unique(mask), [0, 1])) or not np.any(mask):
            raise ValueError(f"Mask is nonbinary or empty: {path.name}")
        if border.size < 6 or border.size % 2 or not np.isfinite(border).all():
            raise ValueError(f"Invalid tumor border: {path.name}")
        border_pairs = border.reshape(-1, 2)
        if (
            float(border_pairs[:, 0].min()) < 1
            or float(border_pairs[:, 0].max()) > image.shape[1]
            or float(border_pairs[:, 1].min()) < 1
            or float(border_pairs[:, 1].max()) > image.shape[0]
        ):
            raise ValueError(f"Tumor border exceeds source dimensions: {path.name}")
        alignment = boundary_alignment_fraction(mask, border)
        if alignment < 0.75:
            raise ValueError(f"Tumor border/mask alignment failed: {path.name} ({alignment})")

        fold = int(cv_folds[index - 1])
        patient_folds[pid].add(fold)
        patient_labels[pid].add(label)
        image_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        mask_hash = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        source_phash = phash64(image)
        nearest = nearest_brisc(source_phash, brisc, label)
        patient_folds[pid].add(fold)
        image_hash_counts[image_hash] += 1
        mask_hash_counts[mask_hash] += 1
        dimension_counts[f"{image.shape[1]}x{image.shape[0]}"] += 1
        alignment_values.append(alignment)
        rows.append(
            {
                "source_index": index,
                "source_path": f"data/external/cheng_figshare_v8/extracted/{index}.mat",
                "patient_id": pid,
                "label": label,
                "official_cv_fold": fold,
                "width": image.shape[1],
                "height": image.shape[0],
                "orientation": "NOT_AVAILABLE",
                "image_dtype": str(image.dtype),
                "image_min": float(image.min()),
                "image_max": float(image.max()),
                "foreground_pixels": int(np.count_nonzero(mask)),
                "foreground_fraction": float(np.mean(mask > 0)),
                "tumor_border_points": border.size // 2,
                "tumor_border_alignment_fraction": alignment,
                "image_array_sha256": image_hash,
                "mask_array_sha256": mask_hash,
                "display_phash": source_phash,
            }
        )
        overlap.append({"source_index": index, "patient_id": pid, "label": label, **nearest})

    if any(len(value) != 1 for value in patient_labels.values()):
        raise ValueError("A patient identifier maps to multiple tumor labels")
    multi_fold_patients = {key: value for key, value in patient_folds.items() if len(value) != 1}
    if multi_fold_patients:
        raise ValueError(f"Official cvind splits patients across folds: {len(multi_fold_patients)}")
    label_counts = Counter(row["label"] for row in rows)
    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Category counts changed: {dict(label_counts)}")
    if len(patient_folds) != 233:
        raise ValueError(f"Patient count changed: {len(patient_folds)}")

    summary = {
        "slices": len(rows),
        "patients": len(patient_folds),
        "label_counts": dict(label_counts),
        "patient_counts_by_label": dict(
            Counter(next(iter(value)) for value in patient_labels.values())
        ),
        "slices_by_fold": dict(Counter(int(value) for value in cv_folds)),
        "patients_by_fold": dict(
            Counter(next(iter(value)) for value in patient_folds.values())
        ),
        "patients_in_multiple_folds": 0,
        "patients_with_multiple_labels": 0,
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "duplicate_image_arrays": sum(value - 1 for value in image_hash_counts.values() if value > 1),
        "duplicate_mask_arrays": sum(value - 1 for value in mask_hash_counts.values() if value > 1),
        "minimum_border_alignment_fraction": min(alignment_values),
        "median_border_alignment_fraction": float(np.median(alignment_values)),
        "brisc_phash": {
            "exact_match_source_slices": sum(row["exact_phash_matches"] > 0 for row in overlap),
            "within_distance_2_source_slices": sum(
                row["matches_within_distance_2"] > 0 for row in overlap
            ),
            "within_distance_4_source_slices": sum(
                row["matches_within_distance_4"] > 0 for row in overlap
            ),
            "nearest_distance_distribution": dict(
                sorted(Counter(row["nearest_phash_distance"] for row in overlap).items())
            ),
        },
    }
    return rows, overlap, summary


def update_registry() -> None:
    path = ROOT / "experiments/registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    match = [item for item in registry["experiments"] if item["id"] == "x01_cheng_overlap_control"]
    if len(match) != 1 or match[0]["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X01 registry state is not the registered pre-audit state")
    match[0]["status"] = "AUDIT_COMPLETE"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8", newline="\n")


def experiment_manifest(paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x01_cheng_overlap_control",
        "status": "AUDIT_COMPLETE",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in sorted(paths)
        ],
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metadata_path = RAW / "figshare_article_v8.json"
    if digest(metadata_path) != EXPECTED_METADATA_SHA256:
        raise ValueError("Figshare article metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["id"] != 1512427
        or metadata["version"] != 8
        or metadata["doi"] != "10.6084/m9.figshare.1512427.v8"
        or metadata["license"]["name"] != "CC BY 4.0"
    ):
        raise ValueError("Figshare release identity changed")
    source_rows = source_file_audit(metadata)
    zip_summary = zip_member_audit()
    with h5py.File(RAW / "cvind.mat", "r") as handle:
        cv_folds = np.asarray(handle["cvind"]).reshape(-1)
    if cv_folds.shape != (3064,) or not np.all(np.isin(cv_folds, [1, 2, 3, 4, 5])):
        raise ValueError("cvind is incomplete or invalid")

    inventory_rows, overlap_rows, dataset_summary = dataset_audit(cv_folds, brisc_rows())
    write_csv(ARTIFACTS / "dataset_manifest.csv", inventory_rows)
    write_csv(ARTIFACTS / "brisc_overlap_candidates.csv", overlap_rows)
    source_manifest = {
        "schema_version": 1,
        "dataset": "Cheng/Figshare brain_tumor_dataset",
        "article_id": 1512427,
        "version": 8,
        "doi": "10.6084/m9.figshare.1512427.v8",
        "license": "CC BY 4.0",
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "files": source_rows,
        "zip_audit": zip_summary,
    }
    write_json(ARTIFACTS / "source_file_manifest.json", source_manifest)
    disk_free = int(__import__("shutil").disk_usage(ROOT.drive + "\\").free)
    audit_result = {
        "schema_version": 1,
        "experiment_id": "x01_cheng_overlap_control",
        "status": "AUDIT_COMPLETE",
        "audit_outcome": "ELIGIBLE_OVERLAP_CONTROL",
        "independent_external_evidence": False,
        "reason": (
            "The official BRISC publication documents Cheng as an upstream source of the "
            "Nickparvar aggregate used to construct BRISC."
        ),
        "dataset": dataset_summary,
        "geometry": {
            "stored_dimensions": dataset_summary["dimension_counts"],
            "matlab_hdf5_transpose_required": True,
            "transpose_verified_against_all_tumor_border_points": True,
            "orientation_per_slice": "NOT_AVAILABLE",
            "documented_in_plane_spacing_mm": [0.49, 0.49],
            "documented_slice_thickness_mm": 6.0,
            "documented_slice_gap_mm": 1.0,
            "volume_and_depth_measurement": "NOT_PERMITTED_WITHOUT_ORDER_AND_ORIENTATION",
        },
        "overlap_detection": {
            "official_provenance_overlap": True,
            "brisc_inventory_path": BRISC_INVENTORY.relative_to(ROOT).as_posix(),
            "brisc_inventory_sha256": EXPECTED_BRISC_INVENTORY_SHA256,
            "brisc_pixels_opened": 0,
            "method": "Existing frozen BRISC pHash metadata compared with transposed Cheng min-max displays",
            **dataset_summary["brisc_phash"],
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
            "raw_bytes": sum((RAW / item).stat().st_size for item in EXPECTED_SOURCE_FILES)
            + metadata_path.stat().st_size,
            "extracted_bytes": sum(path.stat().st_size for path in EXTRACTED.glob("*.mat")),
            "drive_e_free_bytes_after_audit": disk_free,
            "drive_e_free_gib_after_audit": disk_free / 1024**3,
        },
    }
    write_json(ARTIFACTS / "dataset_audit.json", audit_result)

    report = f"""# X01 Cheng/Figshare v8 dataset audit

**Outcome:** `ELIGIBLE_OVERLAP_CONTROL`  
**Independent external evidence:** No  
**Inference performed:** No

## Release and integrity

- Official article: Figshare `1512427`, version `8`, DOI `10.6084/m9.figshare.1512427.v8`.
- Licence: CC BY 4.0.
- All six official source files match their registered byte counts and MD5/SHA256 values.
- All four ZIP archives pass CRC testing and contain exactly `1.mat` through `3064.mat` once.
- All 3,064 extracted MATLAB v7.3/HDF5 files are present and readable.

## Cohort

- 3,064 contrast-enhanced T1 slices from 233 patient identifiers.
- Category slices: glioma 1,426; meningioma 708; pituitary 930.
- Category patients: {json.dumps(dataset_summary['patient_counts_by_label'], sort_keys=True)}.
- The official `cvind` file has five folds. Each patient identifier belongs to exactly one fold; no patient crosses folds and no patient has multiple labels.

## Image, mask, and geometry checks

- Every image and mask pair has matching dimensions, is finite, and is nonblank; every mask is binary and nonempty. Dimensions are {json.dumps(dataset_summary['dimension_counts'], sort_keys=True)}.
- MATLAB v7.3/HDF5 reverses the stored matrix dimensions. Arrays are transposed before use. This was checked against the supplied tumor-border coordinates for every case.
- Minimum tumor-border alignment within a two-pixel tolerance: {dataset_summary['minimum_border_alignment_fraction']:.4f}; median: {dataset_summary['median_border_alignment_fraction']:.4f}.
- Per-slice orientation is not encoded. The README describes axial, coronal, and sagittal images globally but does not identify each slice's orientation.
- The README documents 0.49×0.49 mm in-plane spacing, 6 mm thickness, and 1 mm gap. These values are cohort documentation, not per-file geometry. Volume and skull-depth calculations are prohibited because slice order and orientation are unavailable.

## BRISC overlap decision

The official BRISC paper states that BRISC is derived from the Nickparvar aggregate, which includes the Cheng dataset. That provenance is sufficient to classify this cohort as an overlap control. It cannot be reported as independent external validation.

The audit additionally compared Cheng display pHashes with the already frozen BRISC inventory. It did not open any BRISC pixels, including sealed test pixels. Source slices with a same-label BRISC candidate were: exact pHash {dataset_summary['brisc_phash']['exact_match_source_slices']}, distance ≤2 {dataset_summary['brisc_phash']['within_distance_2_source_slices']}, and distance ≤4 {dataset_summary['brisc_phash']['within_distance_4_source_slices']}. Re-encoding and source conversion can change hashes; these candidates support review but do not replace the documented provenance conclusion.

## Duplicate and storage observations

- Repeated image arrays beyond first occurrence: {dataset_summary['duplicate_image_arrays']}.
- Repeated mask arrays beyond first occurrence: {dataset_summary['duplicate_mask_arrays']}.
- Raw source footprint: {audit_result['disk']['raw_bytes'] / 1024**3:.3f} GiB.
- Extracted footprint: {audit_result['disk']['extracted_bytes'] / 1024**3:.3f} GiB.
- Drive E free after audit: {audit_result['disk']['drive_e_free_gib_after_audit']:.3f} GiB.

## Decision

The frozen stack may be evaluated once on this cohort as X01 overlap-control reproduction. Results must be labeled overlap control and cannot support cross-dataset independence or clinical-generalization claims. Patient identifiers must be the bootstrap unit. All existing thresholds and models remain frozen.
"""
    AUDIT_MD.write_text(report, encoding="utf-8", newline="\n")

    config_path = EXPERIMENT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X01 config state changed before audit")
    config.update(
        {
            "status": "AUDIT_COMPLETE",
            "dataset": {
                **config["dataset"],
                "resolved_version": 8,
                "resolved_doi": "10.6084/m9.figshare.1512427.v8",
                "access_state": "DOWNLOADED_AUDITED",
                "audit_outcome": "ELIGIBLE_OVERLAP_CONTROL",
                "independent_external_evidence": False,
            },
            "r02_external_inference_state": "OPERATIONAL_ARTIFACT_FROZEN",
            "r02_manifest_sha256": (
                "0d56aab5d562a34325003cee116c4f9d5f8fbe3fcc5278717338706d58e5303f"
            ),
            "test_images_accessed": 0,
            "model_retrained_or_refit": False,
        }
    )
    write_json(config_path, config)
    result_path = EXPERIMENT / "result.json"
    write_json(result_path, audit_result)
    decision_path = EXPERIMENT / "decision.md"
    decision_path.write_text(
        "# X01 decision\n\n"
        "**Audit decision:** `ELIGIBLE_OVERLAP_CONTROL`\n\n"
        "The frozen stack may proceed to one evaluation. Cheng is a documented upstream "
        "source of BRISC and therefore cannot provide independent external evidence. No "
        "inference or performance computation occurred during this audit.\n",
        encoding="utf-8",
        newline="\n",
    )
    update_registry()

    review_path = ROOT / "audit/review_log.md"
    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    review_entry = f"""

## {timestamp} — Audit Cheng/Figshare v8 before inference

- **Decision made:** Classify X01 as `ELIGIBLE_OVERLAP_CONTROL` and authorize one frozen-stack evaluation under that label. Do not treat it as independent external validation.
- **Triggering artifact:** `experiments/x01_cheng_overlap_control/artifacts/dataset_audit.json`, SHA256 `{digest(ARTIFACTS / 'dataset_audit.json')}`; official Figshare metadata SHA256 `{EXPECTED_METADATA_SHA256}`.
- **Rationale:** All 3,064 images and masks, 233 patient identifiers, five patient-separated official folds, release hashes, archive membership, mask encoding, and tumor-border alignment passed. The official BRISC publication documents Cheng as an upstream source of BRISC's Nickparvar aggregate, establishing source overlap regardless of re-encoding differences.
- **Verification:** All source files matched byte, MD5, and SHA256 checks. BRISC overlap comparison used only the existing frozen inventory and opened zero BRISC pixels. Inference, performance evaluation, neural retraining, threshold changes, and BRISC test-pixel access were zero.
- **Alternatives considered:** Declare Cheng independent because filenames and encodings differ; rejected because provenance overlap is documented. Stop X01 as ineligible; rejected because the image/mask contract is technically valid for an explicitly labeled overlap control.
- **Who approved:** User approved Amendment 13 and authorized progression through eligible audits and evaluations unless a registered stop condition occurs.
"""
    with review_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(review_entry)

    manifest_paths = [
        ROOT / "audit/x01_cheng_dataset_audit.md",
        ROOT / "scripts/audit_x01_cheng.py",
        ROOT / "scripts/validate_experiment_tracking.py",
        ROOT / "tests/test_foundation_controls.py",
        EXPERIMENT / "hypothesis.md",
        config_path,
        result_path,
        decision_path,
        ARTIFACTS / "dataset_manifest.csv",
        ARTIFACTS / "brisc_overlap_candidates.csv",
        ARTIFACTS / "source_file_manifest.json",
        ARTIFACTS / "dataset_audit.json",
    ]
    write_json(ARTIFACTS / "experiment_manifest.json", experiment_manifest(manifest_paths))


if __name__ == "__main__":
    main()
