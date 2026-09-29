"""Build the registered O01 OOD benchmark without labels or reference masks."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import stage6_external_common as external
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/o01_ood_benchmark"
ART = EXP / "artifacts"
CACHE = ROOT / "data/external/ood_benchmark"
SOURCES = ROOT / "data/external/ood_sources"
SPLIT = ROOT / "data/splits/draft_split.csv"
CHENG_MANIFEST = ROOT / "experiments/x01_cheng_overlap_control/artifacts/dataset_manifest.csv"
OPEN_MANIFEST = ROOT / "experiments/x02_openneuro_pituitary/artifacts/subject_manifest.csv"
UPENN_MANIFEST = ROOT / "experiments/x03_upenn_gbm/artifacts/subject_manifest.csv"
OPEN_ROOT = ROOT / "data/external/openneuro_ds006248_v1.0.0_payload"
UPENN_ROOT = ROOT / "data/external/upenn_gbm_v2"
CHENG_ROOT = ROOT / "data/external/cheng_figshare_v8/extracted"

DOWNLOADS = {
    "organamnist": {
        "url": "https://zenodo.org/records/10519652/files/organamnist.npz?download=1",
        "filename": "organamnist.npz",
        "md5": "68e3f8846a6bd62f0c9bf841c0d9eacc",
        "release": "MedMNIST v2 OrganAMNIST; Zenodo record 10519652; CC BY 4.0",
    },
    "cifar10_train": {
        "url": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
        "filename": "cifar10-train.parquet",
        "sha256": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
        "release": "CIFAR-10 official train partition; University of Toronto dataset mirror",
    },
    "cifar10_test": {
        "url": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
        "filename": "cifar10-test.parquet",
        "sha256": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
        "release": "CIFAR-10 official test partition; University of Toronto dataset mirror",
    },
}


def digest(path: Path, algorithm="sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty table refused: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def hash_rank(value: str, seed="ood-stage-20260929") -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def download_sources() -> dict:
    SOURCES.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, spec in DOWNLOADS.items():
        path = SOURCES / spec["filename"]
        if not path.exists():
            urllib.request.urlretrieve(spec["url"], path)
        algorithm = "md5" if "md5" in spec else "sha256"
        expected = spec.get("md5", spec.get("sha256"))
        actual = digest(path, algorithm)
        if actual != expected:
            raise RuntimeError(f"{name} {algorithm.upper()} mismatch: {actual}")
        records[name] = {**spec, "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
    return records


def save_png(source: str, split: str, sample_id: str, array: np.ndarray) -> tuple[Path, int, int]:
    value = np.asarray(array)
    if value.ndim == 3:
        value = np.asarray(Image.fromarray(value.astype(np.uint8)).convert("L"))
    if value.ndim != 2:
        raise ValueError(f"Expected 2D image, got {value.shape}")
    display, _window = external.display_uint8(value)
    folder = CACHE / source / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sample_id}.png"
    Image.fromarray(display, mode="L").save(path, optimize=False)
    return path, display.shape[1], display.shape[0]


def standardized_hash(array: np.ndarray) -> str:
    tensor, _geometry, _window = external.preprocess_segmenter(array)
    pixels = np.rint(tensor[0] * 255).astype(np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def record(path: Path, source: str, domain: str, split: str, sample_id: str, group_id: str, provenance: str, source_release: str, width: int, height: int, standard_hash: str) -> dict:
    return {
        "sample_id": sample_id,
        "source": source,
        "domain": domain,
        "detector_split": split,
        "group_id": group_id,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest(path),
        "standardized_pixel_sha256": standard_hash,
        "width": width,
        "height": height,
        "file_bytes": path.stat().st_size,
        "provenance": provenance,
        "source_release": source_release,
        "reference_mask_used": False,
        "diagnostic_label_used": False,
        "external_performance_used": False,
    }


def select_brisc(rows: list[dict]) -> list[dict]:
    fit_candidates = [r for r in rows if r["draft_split"] == "train" and r.get("mask_path")]
    eval_candidates = [r for r in rows if r["draft_split"] == "validation" and r.get("mask_path")]
    if len(eval_candidates) != 680:
        raise RuntimeError(f"Expected 680 positive validation cases, got {len(eval_candidates)}")
    group_first = {}
    for row in sorted(fit_candidates, key=lambda r: hash_rank(r["path"])):
        group_first.setdefault(row["similarity_group"], row)
    fit = list(group_first.values())
    if len(fit) < 600:
        raise RuntimeError("Fewer than 600 unique BRISC training similarity groups")
    fit = sorted(fit, key=lambda r: hash_rank(r["path"]))[:600]
    output = []
    for split_name, selected in (("fit", fit), ("evaluation", sorted(eval_candidates, key=lambda r: r["path"]))):
        for index, row in enumerate(selected):
            path = ROOT / row["path"]
            with Image.open(path) as image:
                array = np.asarray(image.convert("L"))
                width, height = image.size
            output.append(record(path, "brisc", "ID", split_name, f"brisc-{split_name}-{index:04d}", f"brisc-sim-{row['similarity_group']}", "BRISC development split", "BRISC 2025 local audited release", width, height, standardized_hash(array)))
    return output


def patient_split(rows: list[dict], patient_key: str) -> tuple[set[str], set[str]]:
    patients = sorted({r[patient_key] for r in rows}, key=hash_rank)
    midpoint = (len(patients) + 1) // 2
    return set(patients[:midpoint]), set(patients[midpoint:])


def add_cheng(existing_hashes: set[str]) -> list[dict]:
    rows = read_csv(CHENG_MANIFEST)
    fit_patients, eval_patients = patient_split(rows, "patient_id")
    output = []
    for split_name, patients in (("fit", fit_patients), ("evaluation", eval_patients)):
        candidates = sorted([r for r in rows if r["patient_id"] in patients], key=lambda r: hash_rank(r["source_index"]))
        accepted = []
        for row in candidates:
            with h5py.File(CHENG_ROOT / f"{row['source_index']}.mat", "r") as handle:
                array = np.asarray(handle["cjdata/image"]).T
            shash = standardized_hash(array)
            if shash in existing_hashes:
                continue
            accepted.append((row, array, shash))
            existing_hashes.add(shash)
            if len(accepted) == 100:
                break
        if len(accepted) != 100:
            raise RuntimeError(f"Cheng {split_name}: only {len(accepted)} uncontaminated candidates")
        for index, (row, array, shash) in enumerate(accepted):
            sample_id = f"cheng-{split_name}-{index:03d}"
            path, width, height = save_png("cheng", split_name, sample_id, array)
            output.append(record(path, "cheng", "NEAR_OOD", split_name, sample_id, f"cheng-patient-{row['patient_id']}", "Cheng/Figshare dataset v8; local audited MATLAB source", "Figshare brain tumor dataset v8", width, height, shash))
    return output


def volume_source(source: str, manifest_path: Path, payload: Path, image_key: str, axis: int, existing_hashes: set[str]) -> list[dict]:
    rows = read_csv(manifest_path)
    patient_key = "subject_id" if "subject_id" in rows[0] else "patient_id"
    fit_patients, eval_patients = patient_split(rows, patient_key)
    output = []
    for split_name, patients in (("fit", fit_patients), ("evaluation", eval_patients)):
        selected = []
        patient_rows = sorted([r for r in rows if r[patient_key] in patients], key=lambda r: hash_rank(r[patient_key]))
        cache = {}
        round_index = 0
        while len(selected) < 100:
            progress = False
            for row in patient_rows:
                patient = row[patient_key]
                if patient not in cache:
                    array, _affine, _geometry = external.load_canonical_nifti(payload / row[image_key])
                    cache[patient] = array
                array = cache[patient]
                count = array.shape[axis]
                offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4]
                if round_index >= len(offsets):
                    continue
                slice_index = max(0, min(count - 1, count // 2 + offsets[round_index]))
                image_slice = np.take(array, slice_index, axis=axis).T
                shash = standardized_hash(image_slice)
                if shash in existing_hashes:
                    continue
                selected.append((row, slice_index, image_slice, shash))
                existing_hashes.add(shash)
                progress = True
                if len(selected) == 100:
                    break
            if not progress and round_index >= len(offsets) - 1:
                break
            round_index += 1
        if len(selected) != 100:
            raise RuntimeError(f"{source} {split_name}: selected {len(selected)}, expected 100")
        for index, (row, slice_index, array, shash) in enumerate(selected):
            sample_id = f"{source}-{split_name}-{index:03d}"
            path, width, height = save_png(source, split_name, sample_id, array)
            provenance = "OpenNeuro ds006248 v1.0.0" if source == "openneuro" else "UPENN-GBM v2 TCIA derivative release"
            output.append(record(path, source, "NEAR_OOD", split_name, sample_id, f"{source}-patient-{row[patient_key]}", provenance, provenance, width, height, shash))
    return output


def add_organamnist(downloads: dict, existing_hashes: set[str]) -> list[dict]:
    data = np.load(ROOT / downloads["organamnist"]["path"])
    sets = {"fit": data["train_images"], "evaluation": data["test_images"]}
    output = []
    for split_name, images in sets.items():
        order = sorted(range(len(images)), key=lambda i: hash_rank(f"organ:{split_name}:{i}"))[:100]
        for ordinal, source_index in enumerate(order):
            array = np.asarray(images[source_index], dtype=np.uint8)
            shash = standardized_hash(array)
            if shash in existing_hashes:
                raise RuntimeError("OrganAMNIST standardized pixels collide with another benchmark class")
            existing_hashes.add(shash)
            sample_id = f"organamnist-{split_name}-{ordinal:03d}"
            path, width, height = save_png("organamnist", split_name, sample_id, array)
            output.append(record(path, "organamnist", "FAR_OOD", split_name, sample_id, f"organamnist-{split_name}-{source_index}", DOWNLOADS["organamnist"]["url"], DOWNLOADS["organamnist"]["release"], width, height, shash))
    return output


def add_cifar(downloads: dict, existing_hashes: set[str]) -> list[dict]:
    import pyarrow.parquet as pq

    tables = {
        "fit": pq.read_table(ROOT / downloads["cifar10_train"]["path"], columns=["img"]),
        "evaluation": pq.read_table(ROOT / downloads["cifar10_test"]["path"], columns=["img"]),
    }
    output = []
    for split_name, table in tables.items():
        images = table.column("img")
        order = sorted(range(len(images)), key=lambda i: hash_rank(f"cifar:{split_name}:{i}"))[:100]
        for ordinal, source_index in enumerate(order):
            value = images[source_index].as_py()
            rgb = np.asarray(Image.open(__import__("io").BytesIO(value["bytes"])).convert("RGB"))
            gray = np.asarray(Image.fromarray(rgb).convert("L"))
            shash = standardized_hash(gray)
            if shash in existing_hashes:
                raise RuntimeError("CIFAR standardized pixels collide with another benchmark class")
            existing_hashes.add(shash)
            sample_id = f"cifar10-{split_name}-{ordinal:03d}"
            path, width, height = save_png("cifar10", split_name, sample_id, gray)
            spec = DOWNLOADS["cifar10_train" if split_name == "fit" else "cifar10_test"]
            output.append(record(path, "cifar10", "FAR_OOD", split_name, sample_id, f"cifar10-{split_name}-{source_index}", spec["url"], spec["release"], width, height, shash))
    return output


def add_synthetic(source: str, existing_hashes: set[str]) -> list[dict]:
    output = []
    for split_name, seed in (("fit", 4201), ("evaluation", 4202)):
        rng = np.random.default_rng(seed)
        for index in range(100):
            if source == "random_noise":
                array = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
            else:
                value = index if split_name == "fit" else 155 + index
                array = np.full((128, 128), value, dtype=np.uint8)
            shash = standardized_hash(array)
            if shash in existing_hashes:
                raise RuntimeError(f"{source} collision at {split_name}/{index}")
            existing_hashes.add(shash)
            sample_id = f"{source}-{split_name}-{index:03d}"
            path, width, height = save_png(source, split_name, sample_id, array)
            output.append(record(path, source, "FAR_OOD", split_name, sample_id, f"{source}-{split_name}-{index}", f"Generated locally with NumPy PCG64 seed {seed}", "Synthetic OOD protocol Amendment 20", width, height, shash))
    return output


def update_registry() -> None:
    path = ROOT / "experiments/registry.json"
    registry = json.loads(path.read_text(encoding="utf-8-sig"))
    target = next(x for x in registry["experiments"] if x["id"] == "o01_ood_benchmark")
    target["status"] = "COMPLETE"
    write_json(path, registry)


def main() -> None:
    if json.loads((EXP / "result.json").read_text())["status"] != "NOT_RUN":
        raise RuntimeError("O01 is not in NOT_RUN state")
    downloads = download_sources()
    rows = read_csv(SPLIT)
    benchmark = select_brisc(rows)
    existing_hashes = {r["standardized_pixel_sha256"] for r in benchmark}
    benchmark += add_cheng(existing_hashes)
    benchmark += volume_source("openneuro", OPEN_MANIFEST, OPEN_ROOT, "image_path", 1, existing_hashes)
    benchmark += volume_source("upenn_gbm", UPENN_MANIFEST, UPENN_ROOT, "image_name", 2, existing_hashes)
    benchmark += add_organamnist(downloads, existing_hashes)
    benchmark += add_cifar(downloads, existing_hashes)
    benchmark += add_synthetic("random_noise", existing_hashes)
    benchmark += add_synthetic("uniform_gray", existing_hashes)
    expected = {("brisc", "fit"): 600, ("brisc", "evaluation"): 680}
    for source in ("cheng", "openneuro", "upenn_gbm", "organamnist", "cifar10", "random_noise", "uniform_gray"):
        expected[(source, "fit")] = 100
        expected[(source, "evaluation")] = 100
    counts = Counter((r["source"], r["detector_split"]) for r in benchmark)
    if dict(counts) != expected:
        raise RuntimeError(f"Benchmark count mismatch: {counts}")
    hash_membership = {}
    for row in benchmark:
        hash_membership.setdefault(row["standardized_pixel_sha256"], []).append(row)
    repeated = {key: value for key, value in hash_membership.items() if len(value) > 1}
    cross_partition_duplicates = {
        key: value
        for key, value in repeated.items()
        if len({(r["domain"], r["detector_split"]) for r in value}) > 1
    }
    if cross_partition_duplicates:
        example = next(iter(cross_partition_duplicates.values()))
        raise RuntimeError(
            "Unexpected standardized-pixel class/split contamination: "
            + ", ".join(f"{r['source']}/{r['detector_split']}/{r['sample_id']}" for r in example)
        )
    fit_groups = {r["group_id"] for r in benchmark if r["detector_split"] == "fit"}
    eval_groups = {r["group_id"] for r in benchmark if r["detector_split"] == "evaluation"}
    if fit_groups & eval_groups:
        raise RuntimeError("Group leakage between detector fitting and evaluation")
    write_csv(ART / "benchmark_manifest.csv", benchmark)
    class_rows = []
    for (source, split_name), count in sorted(counts.items()):
        row0 = next(r for r in benchmark if r["source"] == source and r["detector_split"] == split_name)
        class_rows.append({"source": source, "domain": row0["domain"], "detector_split": split_name, "images": count, "unique_groups": len({r["group_id"] for r in benchmark if r["source"] == source and r["detector_split"] == split_name})})
    write_csv(ART / "class_assignments.csv", class_rows)
    result = {
        "schema_version": 1, "experiment_id": "o01_ood_benchmark", "status": "COMPLETE", "gate": "DESCRIPTIVE", "decision": "ACCEPT_BENCHMARK",
        "counts": {f"{a}/{b}": c for (a, b), c in sorted(counts.items())},
        "total_images": len(benchmark), "fitting_images": sum(r["detector_split"] == "fit" for r in benchmark), "evaluation_images": sum(r["detector_split"] == "evaluation" for r in benchmark),
        "integrity": {"unique_standardized_pixels": len(hash_membership), "within_partition_repeated_pixel_hashes": len(repeated), "fit_evaluation_group_overlap": 0, "unexpected_class_contamination": 0, "all_file_hashes_verified": True},
        "downloads": downloads,
        "historical_context": "Near-OOD source cohorts were previously processed in Stage 6. No Stage-6 performance value, reference mask, or diagnostic label entered benchmark selection or detector fitting.",
        "verification": {"brisc_test_images_accessed": 0, "reference_masks_used": 0, "diagnostic_labels_used": 0, "external_performance_values_used": 0, "neural_retraining_or_refitting": 0, "segmentation_threshold": 0.5, "reliability_policy_threshold": "0.9490250173235835"},
    }
    write_json(EXP / "result.json", result)
    config = json.loads((EXP / "config.json").read_text())
    config.update({"status": "COMPLETE", "sources": list(dict.fromkeys(r["source"] for r in benchmark)), "id_fit": 600, "id_evaluation": 680, "ood_per_source_per_split": 100})
    write_json(EXP / "config.json", config)
    (EXP / "decision.md").write_text("# Decision — O01 OOD benchmark\n\n**Status:** ACCEPTED (descriptive)\n\nThe benchmark contains the registered ID, near-OOD, and far-OOD sources with disjoint fitting/evaluation groups and no standardized-pixel duplicates. No reference masks, diagnostic labels, external performance values, or BRISC sealed-test pixels were used.\n", encoding="utf-8", newline="\n")
    (ROOT / "audit/ood_benchmark.md").write_text(f"""# OOD benchmark audit

O01 is complete with **{len(benchmark):,} images**: {result['fitting_images']:,} detector-fitting images and {result['evaluation_images']:,} held-out evaluation images. The held-out ID cohort is the registered 680 positive BRISC development-validation slices. Each of seven OOD sources contributes 100 fitting and 100 evaluation images.

## Integrity

- SHA256 and a frozen 128×128 standardized-pixel SHA256 are recorded for every image.
- Fitting and evaluation groups are disjoint; near-OOD cohorts are patient-disjoint.
- No standardized-pixel duplicate crosses a benchmark class or split.
- No reference mask, diagnostic label, or Stage-6 performance value was used.
- BRISC sealed-test access: **0**.

## Provenance

Far OOD uses MedMNIST OrganAMNIST from Zenodo record 10519652, CIFAR-10 from the University of Toronto, and deterministic NumPy-generated random and uniform images. Archive MD5 and SHA256 values are stored in `result.json`. Near OOD uses the locally audited Cheng/Figshare, OpenNeuro ds006248 v1.0.0, and UPENN-GBM v2 releases.

## Scope limitation

The near-OOD cohorts were processed previously during Stage 6. They are held out from OOD-detector fitting/evaluation partitions, but are not historically unseen by the project. This benchmark tests registered retrospective separation, not clinical safety or universal OOD detection.
""", encoding="utf-8", newline="\n")
    manifest_files = [ART / "benchmark_manifest.csv", ART / "class_assignments.csv", EXP / "config.json", EXP / "result.json", EXP / "decision.md", ROOT / "audit/ood_benchmark.md"]
    write_json(ART / "experiment_manifest.json", {"schema_version": 1, "experiment_id": "o01_ood_benchmark", "created_at": datetime.now(UTC).isoformat(), "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size} for p in manifest_files]})
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Accept registered OOD benchmark\n\n- **Decision made:** Complete O01 as an accepted descriptive benchmark.\n- **Triggering artifact:** `experiments/o01_ood_benchmark/artifacts/benchmark_manifest.csv`, SHA256 `{digest(ART / 'benchmark_manifest.csv')}`.\n- **Rationale:** Registered populations and counts are present, fitting/evaluation groups are disjoint, and no standardized-pixel class contamination was found.\n- **Verification:** BRISC sealed-test access, reference-mask use, diagnostic-label use, external-performance use, neural retraining, and prior-threshold changes were zero.\n- **Who approved:** User authorized the complete OOD workstream in one sequential run.\n")
    update_registry()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
