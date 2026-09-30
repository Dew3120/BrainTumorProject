"""Fail-closed verification for Stage 9 documentation artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FROZEN_TREE = "439b4fca4b47f12b173c24eaa039b1258f9ddf03"
FROZEN_MANIFEST_SHA256 = "7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a"
THRESHOLDS = {
    "segmentation": "0.50",
    "reliability": "0.9490250173235835",
    "ood_review": "0.36854275476100085",
    "ood_refuse": None,
}


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def assert_close(left: float, right: float, label: str) -> None:
    if left != right:
        raise AssertionError(f"{label}: {left!r} != {right!r}")


def verify_frozen_stack() -> None:
    manifest_path = ROOT / "runs/final_stack/manifest.json"
    manifest = read_json("runs/final_stack/manifest.json")
    assert sha256(manifest_path) == FROZEN_MANIFEST_SHA256
    assert git("rev-parse", "HEAD:runs/final_stack") == FROZEN_TREE
    assert manifest["thresholds"] == THRESHOLDS
    assert manifest["selection"] == {
        "classifier": "v1 raw probabilities",
        "segmenter": "E08 arithmetic mean: E05 + E06 + classifier-routed E07/v1",
        "calibrator": "C03 final-all-validation isotonic mapping",
        "reliability": "R02 10-feature L2 logistic regression",
        "ood": "O05 secondary logistic ensemble",
        "policy": "O06 ACCEPT/REVIEW; REFUSE disabled",
    }
    assert not git("status", "--porcelain", "--", "runs/final_stack")
    assert not git("diff", "--name-only", "--", "runs/final_stack")
    assert not git("diff", "--cached", "--name-only", "--", "runs/final_stack")


def verify_ledger() -> None:
    ledger = read_json("docs/stage9_source_ledger.json")
    metrics = ledger["metrics"]
    classifier = read_json("audit/classifier_training_findings.json")["validation_metrics"]
    segmenter = read_json("audit/segmentation_training_findings.json")["best_validation"]
    e08 = read_json("experiments/e08_ensemble/result.json")["mean_dice"]["ensemble"]
    c03 = read_json("experiments/c03_isotonic/result.json")["metrics"]["aggregate"]
    r02 = read_json("experiments/r02_reduced_features/result.json")["reduced_10_feature_model"]
    o05 = read_json("experiments/o05_ensemble_ood/result.json")["selected_metrics"]
    o06 = read_json("experiments/o06_ood_policy/result.json")["coverage"]
    x02 = read_json("experiments/x02_openneuro_pituitary/result.json")
    x03 = read_json("experiments/x03_upenn_gbm/result.json")
    p02 = read_json("experiments/p02_threshold_selection/result.json")["binary_selection"]
    p05 = read_json("experiments/p05_threshold_sensitivity/result.json")["sensitivity"]

    assert_close(metrics["classifier_internal_validation"]["macro_f1"], classifier["macro_f1"], "classifier macro F1")
    assert_close(metrics["classifier_internal_validation"]["accuracy"], classifier["accuracy"], "classifier accuracy")
    assert_close(metrics["segmenter_v1_internal_validation"]["image_mean_dice"], segmenter["image_mean_dice"], "v1 Dice")
    for category in ("glioma", "meningioma", "pituitary"):
        assert_close(metrics["segmenter_e08_internal_validation"]["per_category_dice"][category], e08[category], f"E08 {category}")
    assert_close(metrics["c03_pixel_calibration"]["lesion_centered_raw_ece"], c03["lesion_centered"]["raw"]["pixel_ece"], "C03 raw ECE")
    assert_close(metrics["c03_pixel_calibration"]["lesion_centered_oof_ece"], c03["lesion_centered"]["oof_calibrated"]["pixel_ece"], "C03 OOF ECE")
    assert_close(metrics["r02_case_reliability_internal_validation"]["auroc"], r02["failure_detection_auroc"], "R02 AUROC")
    assert metrics["r02_case_reliability_internal_validation"]["auroc_ci95"] == [r02["bootstrap"]["ci95_lower"], r02["bootstrap"]["ci95_upper"]]
    assert_close(metrics["o05_ood_evaluation"]["auroc"], o05["auroc"], "O05 AUROC")
    assert metrics["o05_ood_evaluation"]["auroc_ci95"] == o05["auroc_ci95"]
    assert_close(metrics["o06_ood_policy"]["ood_detection_rate"], o06["ood_detection_rate"], "O06 detection")
    assert_close(metrics["external_openneuro"]["subject_macro_dice"], x02["segmentation"]["subject_macro"]["dice"]["subject_macro_mean"], "X02 Dice")
    assert_close(metrics["external_openneuro"]["reliability_auroc"], x02["reliability_transfer"]["failure_detection_auroc"], "X02 reliability")
    assert_close(metrics["external_upenn_gbm"]["subject_macro_dice"], x03["segmentation"]["subject_macro"]["dice"]["subject_macro_mean"], "X03 Dice")
    assert_close(metrics["external_upenn_gbm"]["reliability_auroc"], x03["reliability_transfer"]["failure_detection_auroc"], "X03 reliability")
    assert_close(metrics["policy_internal_validation"]["threshold"], p02["threshold"], "policy threshold")
    assert metrics["policy_internal_validation"]["selected_cost"] == p02["total_cost"]
    assert metrics["policy_sensitivity"]["classification"] == p05["classification"]
    assert ledger["frozen_stack"]["git_tree"] == FROZEN_TREE
    assert ledger["frozen_stack"]["thresholds"] == THRESHOLDS
    for section in metrics.values():
        source = section["source"]
        assert (ROOT / source).is_file(), source


def verify_report() -> dict[str, int]:
    report = (ROOT / "docs/report.md").read_text(encoding="utf-8")
    required = [
        "## What this project is not",
        "## How to read this report",
        "## 7. What we learned that we did not expect",
        "### 9.13 What would change my mind",
        "not a clinical tool",
        "not a deployable medical device",
        "not been validated for patient care",
        "not established generalizability beyond BRISC",
        "not FDA approved, CE marked",
        "0.9490250173235835",
        "0.36854275476100085",
        "c2566214bb921c7185fc8949b3e0376492e7113e",
        "Research prototype — not for clinical use",
        "5 minutes",
        "30 minutes",
        "2 hours",
        "Verification path",
    ]
    for phrase in required:
        assert phrase in report, phrase

    assert report.startswith("---\nproject:")
    words = len(report.split())
    page_equivalent = words / 300
    assert 30 <= page_equivalent <= 50, (words, page_equivalent)
    assert report.index("## What this project is not") < report.index("## Abstract")
    assert report.count("![") == 14

    metric_sources = [
        "audit/classifier_training_findings.json",
        "audit/segmentation_training_findings.json",
        "experiments/e08_ensemble/result.json",
        "experiments/c03_isotonic/result.json",
        "experiments/r02_reduced_features/result.json",
        "experiments/o05_ensemble_ood/result.json",
        "experiments/x02_openneuro_pituitary/result.json",
        "experiments/x03_upenn_gbm/result.json",
        "experiments/p02_threshold_selection/result.json",
        "experiments/p05_threshold_sensitivity/result.json",
    ]
    for source in metric_sources:
        assert source in report or source in json.dumps(read_json("docs/stage9_source_ledger.json"))

    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for target in link_pattern.findall(report):
        clean = target.split("#", maxsplit=1)[0]
        if not clean or clean.startswith(("http://", "https://", "mailto:")):
            continue
        assert (ROOT / "docs" / clean).exists(), target
    return {"words": words, "page_equivalent_at_300_words": round(page_equivalent)}


def verify_figures() -> None:
    manifest = read_json("docs/figures/figure_manifest.json")
    assert manifest["figure_count"] == 14
    assert len(manifest["figures"]) == 14
    for item in manifest["figures"]:
        path = ROOT / item["figure"]
        assert path.is_file(), path
        assert sha256(path) == item["sha256"]
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert all(abs(axis - 300) < 0.01 for axis in dpi), (path, dpi)
            assert image.width >= 1600 and image.height >= 900, (path, image.size)
        for source in item["sources"]:
            source_path = ROOT / source["path"]
            assert source_path.is_file(), source_path
            assert sha256(source_path) == source["sha256"]


def verify_supporting_docs() -> None:
    required = [
        "README.md",
        "docs/methods.md",
        "docs/limitations.md",
        "docs/appendix.md",
        "docs/stage9_source_ledger.json",
        "docs/build_stage9_figures.py",
        "docs/build_stage9_appendix.py",
        "audit/stage10_self_review.md",
        "audit/stage10_review_package.md",
        "audit/stage10_completion_report.md",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative
    appendix = (ROOT / "docs/appendix.md").read_text(encoding="utf-8")
    amendment_ids = set(re.findall(r"preregistration_amendment_(\d{2})", appendix))
    assert amendment_ids == {f"{index:02d}" for index in range(1, 22)}
    registry = read_json("experiments/registry.json")
    for experiment in registry["experiments"]:
        assert f"`{experiment['id']}`" in appendix
    review_log = (ROOT / "audit/review_log.md").read_text(encoding="utf-8")
    assert "Complete Stage 10 hostile self-review and external-review preparation" in review_log
    review_package = (ROOT / "audit/stage10_review_package.md").read_text(encoding="utf-8")
    assert "It does not claim that any external review has already occurred." in review_package


def main() -> None:
    verify_frozen_stack()
    verify_ledger()
    report_stats = verify_report()
    verify_figures()
    verify_supporting_docs()
    print(
        json.dumps(
            {
                "status": "PASS",
                "frozen_tree": FROZEN_TREE,
                "manifest_sha256": FROZEN_MANIFEST_SHA256,
                "figures": 14,
                **report_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
