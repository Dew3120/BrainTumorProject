"""Fetch the minimal, version-locked OpenNeuro X02 payload with annex verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/external/openneuro_ds006248_v1.0.0"
PAYLOAD = ROOT / "data/external/openneuro_ds006248_v1.0.0_payload"
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds006248"
POINTER_RE = re.compile(
    r"SHA256E-s(?P<size>\d+)--(?P<sha256>[0-9a-f]{64})\.nii\.gz"
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_pointer(path: Path) -> tuple[int, str]:
    match = POINTER_RE.search(path.read_text(encoding="utf-8").strip())
    if match is None:
        raise ValueError(f"Invalid git-annex pointer: {path}")
    return int(match.group("size")), match.group("sha256")


def required_files(snapshot: Path = SNAPSHOT) -> list[dict]:
    rows: list[dict] = []
    subjects = sorted(path for path in snapshot.glob("sub-*") if path.is_dir())
    if len(subjects) != 50:
        raise ValueError(f"Expected 50 subjects, observed {len(subjects)}")
    for subject in subjects:
        image_files = list((subject / "anat").glob("*_acq-CECor_T1w.nii.gz"))
        mask_files = list(
            (
                snapshot
                / "derivatives/segmentations"
                / subject.name
                / "anat"
            ).glob("*_label-groundTruth.nii.gz")
        )
        for kind, files in (("image", image_files), ("mask", mask_files)):
            if len(files) != 1:
                raise ValueError(
                    f"Expected one {kind} pointer for {subject.name}, observed {len(files)}"
                )
            pointer = files[0]
            size, expected_hash = parse_pointer(pointer)
            relative_path = pointer.relative_to(snapshot).as_posix()
            rows.append(
                {
                    "subject_id": subject.name,
                    "kind": kind,
                    "relative_path": relative_path,
                    "bytes": size,
                    "sha256": expected_hash,
                    "url": f"{BASE_URL}/{relative_path}",
                }
            )
    return rows


def fetch(row: dict, payload: Path = PAYLOAD) -> str:
    destination = payload / row["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == row["bytes"] and sha256(destination) == row["sha256"]:
            return "verified_existing"
        raise ValueError(f"Existing payload failed integrity verification: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(row["url"], headers={"User-Agent": "BrainTumorProject-X02-audit/1.0"})
    value = hashlib.sha256()
    observed_bytes = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                value.update(chunk)
                observed_bytes += len(chunk)
        if observed_bytes != row["bytes"]:
            raise ValueError(
                f"Size mismatch for {row['relative_path']}: {observed_bytes} != {row['bytes']}"
            )
        if value.hexdigest() != row["sha256"]:
            raise ValueError(f"SHA256 mismatch for {row['relative_path']}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    rows = required_files()
    summary = {
        "snapshot": "OpenNeuro ds006248 1.0.0",
        "files": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "bytes": sum(row["bytes"] for row in rows),
        "payload_root": str(PAYLOAD),
    }
    print(json.dumps(summary, indent=2))
    if args.manifest_only:
        return
    counts = {"downloaded": 0, "verified_existing": 0}
    for index, row in enumerate(rows, start=1):
        state = fetch(row)
        counts[state] += 1
        print(f"[{index:03d}/{len(rows):03d}] {state}: {row['relative_path']}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
