import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw" / "brisc"
OUT = PROJECT / "audit" / "runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
SPLITS = PROJECT / "data" / "splits"

OUT.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

CLASSES = {"glioma", "meningioma", "pituitary", "no_tumor"}
EXTENSIONS = {".jpg", ".jpeg", ".png"}
NEAR_DISTANCE = 4  # Review heuristic, NOT a patient identifier.
MASK_THRESHOLD = 128  # Provisional 8-bit grayscale -> binary conversion.


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_digest(image):
    value = f"{image.mode}:{image.size}".encode() + image.tobytes()
    return hashlib.sha256(value).hexdigest()


def read_binary_mask(path):
    """Preserve source files; document thresholding instead of hiding it."""
    with Image.open(path) as source:
        source.load()
        if source.mode not in {"1", "L"}:
            raise ValueError(f"Unsupported mask mode: {source.mode}")
        gray = source.convert("L")
    histogram = gray.histogram()
    values = {value for value, count in enumerate(histogram) if count}
    intermediate = sum(histogram[2:255])
    if values.issubset({0, 1}):
        binary = gray.point(lambda value: 255 if value else 0)
        policy = "binary_0_1"
    elif values.issubset({0, 255}):
        binary = gray.copy()
        policy = "binary_0_255"
    else:
        binary = gray.point(lambda value: 255 if value >= MASK_THRESHOLD else 0)
        policy = "grayscale_threshold_128_REVIEW_REQUIRED"
    return binary, policy, intermediate / (gray.width * gray.height)


roots = [
    path for path in RAW.rglob("classification_task")
    if path.is_dir()
]

if len(roots) != 1:
    raise SystemExit(
        "Expected exactly one classification_task directory. "
        "Check the download/extraction; do not merge multiple copies."
    )

classification = roots[0]
dataset = classification.parent
segmentation = dataset / "segmentation_task"

if not segmentation.is_dir():
    raise SystemExit("Missing segmentation_task directory.")

rows = []
issues = []
used_masks = set()
seen_names = set()

print("Checking images and masks...", flush=True)

for path in sorted(classification.rglob("*")):
    if path.suffix.lower() not in EXTENSIONS:
        continue

    relative = path.relative_to(classification)
    if len(relative.parts) != 3:
        issues.append({
            "file": str(relative),
            "issue": "Unexpected classification folder structure",
        })
        continue

    split, label, filename = relative.parts
    if split not in {"train", "test"} or label not in CLASSES:
        issues.append({
            "file": str(relative),
            "issue": "Unexpected split or category",
        })
        continue

    try:
        with Image.open(path) as source:
            source.load()
            image = source.convert("L")

        if filename in seen_names:
            raise ValueError("Repeated classification filename")
        seen_names.add(filename)

        row = {
            "path": path.relative_to(PROJECT).as_posix(),
            "filename": filename,
            "original_split": split,
            "label": label,
            "width": image.width,
            "height": image.height,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pixel_sha256": image_digest(image),
            "phash": str(imagehash.phash(image)),
            "mask_path": "",
            "mask_policy": "none",
            "mask_intermediate_fraction": 0.0,
        }

        if image.getextrema()[0] == image.getextrema()[1]:
            issues.append({
                "file": str(relative),
                "issue": "Constant-intensity image",
            })

        mask_path = segmentation / split / "masks" / f"{path.stem}.png"
        segment_image = segmentation / split / "images" / filename

        if label != "no_tumor" or mask_path.exists():
            if not mask_path.exists():
                raise ValueError("Missing tumor mask")
            if not segment_image.exists():
                raise ValueError("Missing segmentation image")

            mask, policy, fraction = read_binary_mask(mask_path)
            row["mask_policy"] = policy
            row["mask_intermediate_fraction"] = fraction

            with Image.open(segment_image) as source:
                source.load()
                other_image = source.convert("L")

            if mask.size != image.size:
                raise ValueError("Image and mask dimensions differ")
            if image_digest(other_image) != image_digest(image):
                raise ValueError(
                    "Classification and segmentation images differ"
                )

            if label != "no_tumor" and not mask.getbbox():
                raise ValueError("Tumor example has an empty mask")
            if label == "no_tumor" and mask.getbbox():
                raise ValueError("No-tumor example has a nonempty mask")

            row["mask_path"] = mask_path.relative_to(PROJECT).as_posix()
            used_masks.add(mask_path.resolve())

        rows.append(row)

    except (KeyError, OSError, ValueError) as error:
        issues.append({"file": str(relative), "issue": str(error)})

    if len(rows) and len(rows) % 500 == 0:
        print(f"Checked {len(rows)} usable images", flush=True)

for path in segmentation.rglob("*.png"):
    if path.parent.name == "masks" and path.resolve() not in used_masks:
        issues.append({
            "file": path.relative_to(dataset).as_posix(),
            "issue": "Mask was not successfully matched",
        })

if not rows:
    write_csv(OUT / "issues.csv", issues, ["file", "issue"])
    raise SystemExit("No usable images found. Inspect audit/issues.csv.")

# Conservatively group candidate similar images.
# These groups do not identify patients.
parent = list(range(len(rows)))


def find(index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def join(left, right):
    left, right = find(left), find(right)
    if left != right:
        parent[right] = left


hashes = [int(row["phash"], 16) for row in rows]
pair_count = 0
cross_split_pairs = 0

print("Screening image similarity; this may take several minutes...", flush=True)

with (OUT / "similarity_candidates.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "file_a", "file_b", "phash_distance",
        "same_decoded_pixels", "cross_original_split",
    ])

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            exact = (
                rows[left]["pixel_sha256"]
                == rows[right]["pixel_sha256"]
            )
            distance = (hashes[left] ^ hashes[right]).bit_count()

            if exact or distance <= NEAR_DISTANCE:
                join(left, right)
                cross = (
                    rows[left]["original_split"]
                    != rows[right]["original_split"]
                )
                pair_count += 1
                cross_split_pairs += int(cross)
                writer.writerow([
                    rows[left]["path"],
                    rows[right]["path"],
                    distance,
                    exact,
                    cross,
                ])

groups = defaultdict(list)
for index in range(len(rows)):
    group = find(index)
    rows[index]["similarity_group"] = group
    groups[group].append(index)

test_groups = {
    row["similarity_group"]
    for row in rows
    if row["original_split"] == "test"
}

conflicting_groups = {
    group for group, members in groups.items()
    if len({rows[index]["label"] for index in members}) > 1
}

# Keep the original test membership.
# Exclude related training candidates from the draft development split.
eligible = []
for index, row in enumerate(rows):
    group = row["similarity_group"]
    if row["original_split"] == "test":
        row["draft_split"] = "test"
    elif group in test_groups or group in conflicting_groups:
        row["draft_split"] = "review_excluded"
    else:
        row["draft_split"] = "train"
        eligible.append(index)

split_created = False

try:
    group_counts = {
        label: len({
            rows[index]["similarity_group"]
            for index in eligible
            if rows[index]["label"] == label
        })
        for label in CLASSES
    }

    if min(group_counts.values()) < 5:
        raise ValueError("Too few eligible groups in at least one class")

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    _, validation = next(splitter.split(
        X=[[0] for _ in eligible],
        y=[rows[index]["label"] for index in eligible],
        groups=[rows[index]["similarity_group"] for index in eligible],
    ))

    for position in validation:
        rows[eligible[position]]["draft_split"] = "validation"

    split_created = True

except (StopIteration, ValueError) as error:
    issues.append({"file": "split", "issue": str(error)})

counts = Counter(
    f"{row['original_split']}/{row['label']}" for row in rows
)
draft_counts = Counter(row["draft_split"] for row in rows)

summary = {
    "status": "DRAFT AUDIT - HUMAN REVIEW REQUIRED",
    "readable_classification_images": len(rows),
    "matched_masks": sum(bool(row["mask_path"]) for row in rows),
    "counts": dict(sorted(counts.items())),
    "technical_issues": len(issues),
    "similarity_candidate_pairs": pair_count,
    "cross_original_split_candidate_pairs": cross_split_pairs,
    "similarity_groups_with_multiple_labels": len(conflicting_groups),
    "draft_split_created": split_created,
    "draft_split_counts": dict(draft_counts),
    "patient_independence": "NOT VERIFIED",
    "physical_measurements": "PIXELS ONLY until spacing is verified",
    "mask_threshold": MASK_THRESHOLD,
    "thresholded_masks_requiring_review": sum(
        row["mask_policy"] == "grayscale_threshold_128_REVIEW_REQUIRED"
        for row in rows
    ),
    "mask_policy_note": "Original masks unchanged. Threshold 128 is provisional; review before training and use the same approved conversion in training and evaluation.",
}

write_csv(OUT / "inventory.csv", rows, list(rows[0].keys()))
write_csv(OUT / "issues.csv", issues, ["file", "issue"])

if split_created:
    write_csv(SPLITS / "draft_split.csv", rows, list(rows[0].keys()))

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

# Training examples only: original image beside mask overlay.
for label in sorted(CLASSES):
    candidates = [
        row for row in rows
        if row["original_split"] == "train" and row["label"] == label
    ]
    if not candidates:
        continue

    count = min(8, len(candidates))
    selected = [
        candidates[position * len(candidates) // count]
        for position in range(count)
    ]

    sheet = Image.new("RGB", (512, 280 * count), "white")
    draw = ImageDraw.Draw(sheet)

    for position, row in enumerate(selected):
        with Image.open(PROJECT / row["path"]) as source:
            image = source.convert("RGB")

        overlay = image.copy()

        if row["mask_path"]:
            binary, _, _ = read_binary_mask(PROJECT / row["mask_path"])
            mask = binary.point(lambda value: 100 if value else 0)
            red = Image.new("RGB", image.size, "red")
            overlay = Image.composite(red, image, mask)

        y = position * 280
        sheet.paste(ImageOps.contain(image, (256, 256)), (0, y))
        sheet.paste(ImageOps.contain(overlay, (256, 256)), (256, y))
        draw.text((4, y + 258), row["filename"], fill="black")

    sheet.save(OUT / f"preview_{label}.png")

print(json.dumps(summary, indent=2))
print(f"\nResults folder: {OUT}")
print("\nAudit finished. This is NOT automatic approval to train.")
