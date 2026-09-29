"""Shared registered OOD evaluation utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/o01_ood_benchmark/artifacts/benchmark_manifest.csv"
BOOTSTRAPS = 2000
SEED = 42


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty CSV refused: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_manifest() -> list[dict]:
    rows = read_csv(MANIFEST)
    if len(rows) != 2680:
        raise RuntimeError("O01 manifest population changed")
    for row in rows:
        path = ROOT / row["path"]
        if not path.exists() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"O01 image hash mismatch: {row['sample_id']}")
    return rows


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([0 if row["domain"] == "ID" else 1 for row in rows], dtype=np.int64)


def equal_frequency_calibration(probability, target, bins=15):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    order = np.argsort(probability, kind="mergesort")
    parts = np.array_split(order, bins)
    records = []
    ece = 0.0
    mce = 0.0
    for index, part in enumerate(parts, start=1):
        if not len(part):
            continue
        confidence = float(probability[part].mean())
        observed = float(target[part].mean())
        gap = abs(confidence - observed)
        ece += len(part) / len(target) * gap
        mce = max(mce, gap)
        records.append({"bin": index, "n": len(part), "mean_probability": confidence, "observed_ood_fraction": observed, "absolute_gap": gap})
    return {"ece": float(ece), "mce": float(mce), "bins": records}


def stratified_group_bootstrap_auc(rows, target, score, resamples=BOOTSTRAPS, seed=SEED):
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    groups = np.asarray([row["group_id"] for row in rows], dtype=object)
    rng = np.random.default_rng(seed)
    by_label = {value: np.unique(groups[target == value]) for value in (0, 1)}
    values = []
    for _ in range(resamples):
        indices = []
        for value in (0, 1):
            pool = by_label[value]
            chosen = rng.choice(pool, size=len(pool), replace=True)
            for group in chosen:
                indices.extend(np.where((groups == group) & (target == value))[0].tolist())
        values.append(roc_auc_score(target[indices], score[indices]))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])], np.asarray(values)


def paired_group_bootstrap_auc_difference(rows, target, first, second, resamples=BOOTSTRAPS, seed=SEED):
    target = np.asarray(target, dtype=np.int64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    groups = np.asarray([row["group_id"] for row in rows], dtype=object)
    rng = np.random.default_rng(seed)
    by_label = {value: np.unique(groups[target == value]) for value in (0, 1)}
    values = []
    for _ in range(resamples):
        indices = []
        for value in (0, 1):
            pool = by_label[value]
            chosen = rng.choice(pool, size=len(pool), replace=True)
            for group in chosen:
                indices.extend(np.where((groups == group) & (target == value))[0].tolist())
        values.append(roc_auc_score(target[indices], first[indices]) - roc_auc_score(target[indices], second[indices]))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def metrics(rows, probability, score=None):
    target = labels(rows)
    probability = np.asarray(probability, dtype=np.float64)
    ranking = probability if score is None else np.asarray(score, dtype=np.float64)
    auc = float(roc_auc_score(target, ranking))
    ci, _ = stratified_group_bootstrap_auc(rows, target, ranking)
    calibration = equal_frequency_calibration(probability, target)
    return {"auroc": auc, "auroc_ci95": ci, "ece": calibration["ece"], "mce": calibration["mce"], "brier": float(brier_score_loss(target, probability)), "calibration_bins": calibration["bins"]}


def roc_points(rows, score):
    target = labels(rows)
    fpr, tpr, thresholds = roc_curve(target, score)
    return fpr, tpr, thresholds


def mark_complete(experiment_id: str) -> None:
    path = ROOT / "experiments/registry.json"
    registry = read_json(path)
    item = next(x for x in registry["experiments"] if x["id"] == experiment_id)
    item["status"] = "COMPLETE"
    write_json(path, registry)


def experiment_manifest(experiment: Path, paths: list[Path]) -> None:
    from datetime import UTC, datetime

    write_json(experiment / "artifacts/experiment_manifest.json", {"schema_version": 1, "experiment_id": experiment.name, "created_at": datetime.now(UTC).isoformat(), "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]})

