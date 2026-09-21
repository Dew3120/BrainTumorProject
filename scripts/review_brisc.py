"""Reproducible Phase 1 review. Never edits raw data or chooses medical labels."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import csv
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, help="Existing audit run directory")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    audit = Path(args.audit).resolve()
    rows = read_csv(audit / "inventory.csv")
    pairs = read_csv(audit / "similarity_candidates.csv")
    raw = project / "data/raw/brisc/brisc2025"
    output = project / "audit/reviews" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    print("Verifying every file listed in the publisher manifest...", flush=True)
    manifest = read_csv(raw / "manifest.csv")
    integrity_errors = []
    manifest_paths = set()
    for entry in manifest:
        path = (raw / entry["relative_path"].replace("\\", "/")).resolve()
        if not path.is_relative_to(raw.resolve()):
            raise ValueError("Manifest path escapes dataset directory")
        manifest_paths.add(path)
        if not path.is_file():
            integrity_errors.append({"path": entry["relative_path"], "issue": "missing"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"].lower():
            integrity_errors.append({"path": entry["relative_path"], "issue": "sha256 mismatch"})
        if path.stat().st_size != int(entry["file_size_bytes"]):
            integrity_errors.append({"path": entry["relative_path"], "issue": "size mismatch"})
    actual = {
        path.resolve() for task in ("classification_task", "segmentation_task")
        for path in (raw / task).rglob("*") if path.is_file()
    }
    unlisted = actual - manifest_paths
    write_csv(output / "integrity_issues.csv", integrity_errors, ["path", "issue"])

    groups = defaultdict(list)
    exact = defaultdict(list)
    by_path = {row["path"]: row for row in rows}
    for row in rows:
        groups[row["similarity_group"]].append(row)
        exact[row["pixel_sha256"]].append(row)
    mixed = {g: rs for g, rs in groups.items() if len({r["label"] for r in rs}) > 1}
    exact_conflicts = {g: rs for g, rs in exact.items() if len({r["label"] for r in rs}) > 1}
    exact_cross = {
        g: rs for g, rs in exact.items()
        if len({r["original_split"] for r in rs}) > 1
    }
    review_rows = []
    print("Creating training-only review sheets for all mixed-label groups...", flush=True)
    for group, members in mixed.items():
        train = [r for r in members if r["original_split"] == "train"]
        for row in members:
            review_rows.append({
                "group": group, "path": row["path"], "label": row["label"],
                "original_split": row["original_split"],
                "decision": "retain source label; quarantine development members",
                "reason": "similarity candidate only; no medical relabeling",
            })
        if not train:
            continue
        width, height = 320, 360
        sheet = Image.new("RGB", (width * 3, height * ((len(train) + 2) // 3)), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(train):
            with Image.open(project / row["path"]) as source:
                im = ImageOps.contain(source.convert("RGB"), (width, 300))
            x, y = (index % 3) * width, (index // 3) * height
            sheet.paste(im, (x, y))
            draw.text((x + 3, y + 303), f"Group {group} | {row['label']}", fill="white")
            draw.text((x + 3, y + 323), row["filename"], fill="white")
        sheet.save(output / f"group_{group}_train.png")
    write_csv(output / "group_review.csv", review_rows,
              ["group", "path", "label", "original_split", "decision", "reason"])

    print("Measuring threshold sensitivity on TRAINING masks only...", flush=True)
    mask_rows = []
    for row in rows:
        if row["original_split"] != "train" or not row["mask_path"]:
            continue
        with Image.open(project / row["mask_path"]) as source:
            a = np.asarray(source.convert("L"))
        areas = {t: int(np.count_nonzero(a >= t)) for t in (64, 128, 192)}
        base = areas[128]
        mask_rows.append({
            "path": row["mask_path"], "label": row["label"],
            "area_64": areas[64], "area_128": base, "area_192": areas[192],
            "intermediate_pixel_fraction": float(np.mean((a > 0) & (a < 255))),
            "relative_area_span": (areas[64] - areas[192]) / max(base, 1),
            "dice_64_vs_128": 2 * base / max(areas[64] + base, 1),
            "dice_192_vs_128": 2 * areas[192] / max(areas[192] + base, 1),
        })
    write_csv(output / "mask_threshold_review.csv", mask_rows, list(mask_rows[0]))
    spans = np.array([r["relative_area_span"] for r in mask_rows])
    draft = read_csv(project / "data/splits/draft_split.csv")
    split_groups = {
        split: {r["similarity_group"] for r in draft if r["draft_split"] == split}
        for split in ("train", "validation", "test")
    }
    separation = not any(
        split_groups[a] & split_groups[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    exclusions = []
    for row in draft:
        if row["draft_split"] != "review_excluded":
            continue
        members = groups[row["similarity_group"]]
        same_exact = exact[row["pixel_sha256"]]
        exclusions.append({
            "path": row["path"], "label": row["label"],
            "similarity_group": row["similarity_group"],
            "group_contains_original_test": any(r["original_split"] == "test" for r in members),
            "group_has_multiple_labels": row["similarity_group"] in mixed,
            "identical_pixels_to_original_test": any(r["original_split"] == "test" for r in same_exact),
            "action": "quarantine; no deletion or relabeling",
        })
    write_csv(output / "exclusion_reasons.csv", exclusions, list(exclusions[0]))
    report = {
        "source_audit": str(audit),
        "manifest_entries": len(manifest), "integrity_errors": len(integrity_errors),
        "unlisted_task_files": len(unlisted),
        "mixed_label_similarity_groups": len(mixed),
        "exact_pixel_groups_with_conflicting_labels": len(exact_conflicts),
        "exact_pixel_groups_crossing_original_splits": len(exact_cross),
        "exact_cross_split_training_images": sum(
            r["original_split"] == "train" for rs in exact_cross.values() for r in rs
        ),
        "cross_split_candidate_pairs": sum(p["cross_original_split"] == "True" for p in pairs),
        "quarantined_training_images": len(exclusions),
        "draft_similarity_groups_separated": separation,
        "all_original_test_rows_preserved": all(r["draft_split"] == "test" for r in draft if r["original_split"] == "test"),
        "training_masks_reviewed": len(mask_rows),
        "empty_masks_at_128": sum(r["area_128"] == 0 for r in mask_rows),
        "threshold_area_span_percentiles": dict(zip(
            ["p50", "p95", "p99", "max"], map(float, np.percentile(spans, [50, 95, 99, 100]))
        )),
        "masks_area_span_above_5_percent": int(np.count_nonzero(spans > .05)),
        "status": "TECHNICAL REVIEW; NO CLINICAL LABEL VERIFICATION",
    }
    (output / "review_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Review output:", output)


if __name__ == "__main__":
    main()
