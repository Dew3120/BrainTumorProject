"""Fixed diagnostic: can original image dimensions predict the binary label?"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def development_parts(rows):
    required = {"path", "draft_split", "label", "width", "height", "similarity_group"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Expected nonempty BRISC split with columns: {sorted(required)}"
        )
    parts = {
        name: [r for r in rows if r["draft_split"] == name]
        for name in ("train", "validation")
    }
    for name, subset in parts.items():
        if not subset or {r["label"] == "no_tumor" for r in subset} != {True, False}:
            raise ValueError(f"{name} requires both tumor and no_tumor examples")
        for row in subset:
            if row["label"] not in {"glioma", "meningioma", "pituitary", "no_tumor"}:
                raise ValueError("Unexpected BRISC class")
            dimensions = [float(row["width"]), float(row["height"])]
            if not all(np.isfinite(v) and v > 0 for v in dimensions):
                raise ValueError("Image dimensions must be positive and finite")
            if not row["similarity_group"]:
                raise ValueError("Missing similarity group")
    for key in ("path", "similarity_group"):
        if {r[key] for r in parts["train"]} & {r[key] for r in parts["validation"]}:
            raise ValueError(f"Training/validation overlap in {key}")
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    with (run / "split_snapshot.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    parts = development_parts(rows)
    values = {}
    for part, subset in parts.items():
        x = np.array(
            [
                [
                    float(r["width"]),
                    float(r["height"]),
                    float(r["width"]) / float(r["height"]),
                ]
                for r in subset
            ]
        )
        y = np.array([int(r["label"] != "no_tumor") for r in subset])
        values[part] = (x, y)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(*values["train"])
    x_val, y_val = values["validation"]
    score = model.predict_proba(x_val)[:, 1]
    result = {
        "experiment": "Geometry-only tumor/no-tumor classifier",
        "split_sha256": hashlib.sha256(
            (run / "split_snapshot.csv").read_bytes()
        ).hexdigest(),
        "model": "StandardScaler fit on training only; LogisticRegression class_weight=balanced, max_iter=2000, random_state=42",
        "features": ["original_width", "original_height", "width_divided_by_height"],
        "image_pixels_used": False,
        "test_evaluated": False,
        "train_images": len(parts["train"]),
        "validation_images": len(parts["validation"]),
        "validation_roc_auc": float(roc_auc_score(y_val, score)),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(y_val, score >= 0.5)
        ),
        "confusion_matrix_rows_true_columns_predicted_no_tumor_then_tumor": confusion_matrix(
            y_val, score >= 0.5, labels=[0, 1]
        ).tolist(),
        "interpretation": "Above-chance results show geometry-label association, not proof that ResNet uses it. Near-chance results do not exclude other shortcuts. Original dimensions are not explicit ResNet inputs; aspect ratio can survive padding.",
    }
    destination = run / "geometry_shortcut_check.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Saved:", destination)


if __name__ == "__main__":
    main()
