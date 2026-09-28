"""Download the minimal expert-labelled UPENN-GBM CE-T1 evaluation payload."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "data/external/upenn_gbm_v2"
API_URL = "https://api.github.com/repos/data-nih/tcia/releases/tags/upenn-gbm"
EXPECTED_RELEASE_ID = 222317008
EXPECTED_TAG = "upenn-gbm"
EXPECTED_PUBLISHED = "2025-06-02T02:42:11Z"
EXPECTED_SEGMENTATIONS = 147


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_release() -> dict:
    request = urllib.request.Request(API_URL, headers={"User-Agent": "BrainTumorProject-Audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    if (
        release.get("id") != EXPECTED_RELEASE_ID
        or release.get("tag_name") != EXPECTED_TAG
        or release.get("published_at") != EXPECTED_PUBLISHED
    ):
        raise ValueError("UPENN-GBM release identity changed")
    return release


def select_assets(release: dict) -> list[dict]:
    by_name = {asset["name"]: asset for asset in release["assets"]}
    segmentations = sorted(
        name for name in by_name if name.startswith("sub-") and name.endswith("_seg.nii.gz")
    )
    if len(segmentations) != EXPECTED_SEGMENTATIONS:
        raise ValueError("UPENN-GBM expert segmentation count changed")
    selected = []
    for segmentation_name in segmentations:
        subject_id = segmentation_name.removesuffix("_seg.nii.gz")
        image_name = f"{subject_id}_ce-gd_T1w.nii.gz"
        if image_name not in by_name:
            raise ValueError(f"Missing CE-T1 pair for {subject_id}")
        for kind, name in (("image", image_name), ("mask", segmentation_name)):
            asset = by_name[name]
            digest = asset.get("digest", "")
            if not digest.startswith("sha256:"):
                raise ValueError(f"Missing SHA256 digest for {name}")
            selected.append(
                {
                    "subject_id": subject_id,
                    "kind": kind,
                    "name": name,
                    "bytes": int(asset["size"]),
                    "sha256": digest.removeprefix("sha256:"),
                    "url": asset["browser_download_url"],
                    "asset_id": int(asset["id"]),
                }
            )
    if len(selected) != EXPECTED_SEGMENTATIONS * 2:
        raise ValueError("UPENN-GBM minimal payload membership changed")
    return selected


def download_one(row: dict) -> dict:
    destination = PAYLOAD / row["name"]
    if destination.is_file():
        if destination.stat().st_size == row["bytes"] and sha256(destination) == row["sha256"]:
            return {**row, "download_state": "verified_existing"}
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(row["url"], headers={"User-Agent": "BrainTumorProject-Audit"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if partial.stat().st_size != row["bytes"] or sha256(partial) != row["sha256"]:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Downloaded asset failed integrity verification: {row['name']}")
    partial.replace(destination)
    return {**row, "download_state": "downloaded_verified"}


def main() -> None:
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    release = fetch_release()
    selected = select_assets(release)
    required = sum(row["bytes"] for row in selected)
    free_before = shutil.disk_usage(PAYLOAD).free
    if free_before < required * 3:
        raise ValueError("Insufficient free space for verified X03 minimal payload")
    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(download_one, selected))
    manifest = {
        "schema_version": 1,
        "dataset": "UPENN-GBM",
        "dataset_version": 2,
        "doi": "10.7937/TCIA.709X-DN49",
        "source_repository": "data-nih/tcia",
        "release_id": EXPECTED_RELEASE_ID,
        "tag_name": EXPECTED_TAG,
        "published_at": EXPECTED_PUBLISHED,
        "release_api_url": API_URL,
        "release_html_url": release["html_url"],
        "license": "CC BY 4.0",
        "selection": "147 expert segmentation assets and their paired CE-Gd T1w volumes",
        "files": verified,
        "total_bytes": required,
        "drive_e_free_bytes_before": free_before,
        "drive_e_free_bytes_after": shutil.disk_usage(PAYLOAD).free,
    }
    path = PAYLOAD / "source_file_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "files": len(verified), "bytes": required}, indent=2))


if __name__ == "__main__":
    main()
