"""Run locked pixel-calibration and case-reliability protocols for E06."""

import importlib
import json
from pathlib import Path

import calibrate_segmenter_pixels as pixel

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e06_boundary_loss"
ARTIFACTS = EXPERIMENT / "artifacts"


def write_json_lf(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def configure_pixel(run, checkpoint, selected_checkpoint):
    pixel.RUN_NAME = run.name
    pixel.RUN = run
    pixel.CHECKPOINT = checkpoint
    pixel.FINDINGS = selected_checkpoint
    pixel.OUTPUT_JSON = ARTIFACTS / "segmenter_pixel_calibration.json"
    pixel.OUTPUT_MD = ARTIFACTS / "segmenter_pixel_calibration.md"
    pixel.CALIBRATION_FINDINGS = ARTIFACTS / "confidence_calibration_findings.json"
    pixel.FOREGROUND_DIAGRAM = run / "segmenter_foreground_reliability_diagram.png"
    pixel.WHOLE_DIAGRAM = run / "segmenter_whole_image_reliability_diagram.png"
    pixel.FOLD_RESULTS = run / "segmenter_pixel_calibration_fold_results.csv"
    pixel.OOF_RESULTS = run / "segmenter_pixel_calibration_oof_metrics.csv"


def configure_reliability(module, run, checkpoint):
    module.RUN_NAME = run.name
    module.RUN = run
    module.CHECKPOINT = checkpoint
    module.PIXEL_CALIBRATION = pixel.OUTPUT_JSON
    module.CALIBRATION_FINDINGS = pixel.CALIBRATION_FINDINGS
    module.OUTPUT_JSON = ARTIFACTS / "segmenter_case_reliability.json"
    module.OUTPUT_MD = ARTIFACTS / "segmenter_case_reliability.md"
    module.OOF_PATH = run / "segmenter_case_reliability_oof.csv"
    module.COEFFICIENT_PATH = run / "segmenter_case_reliability_coefficients.csv"
    module.TOP_RISK_PATH = run / "segmenter_case_reliability_top20_risk.csv"
    module.DIAGRAM_PATH = run / "segmenter_case_reliability_diagram.png"


def main():
    run = Path(
        (ROOT / "runs/segmentation_v2_boundary_loss/latest_boundary.txt")
        .read_text(encoding="utf-8")
        .strip()
    ).resolve()
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    best = result["top3"][0]
    checkpoint = (run / "top3_checkpoints" / best["checkpoint"]).resolve()
    checkpoint_sha256 = pixel.sha256(checkpoint)
    selected = ARTIFACTS / "selected_checkpoint.json"
    write_json_lf(
        selected,
        {
            "experiment_id": "e06_boundary_loss",
            "run": str(run),
            "top3_checkpoints": [
                {
                    "rank": 1,
                    "epoch": result["best_epoch"],
                    "file": checkpoint.name,
                    "sha256": checkpoint_sha256.upper(),
                }
            ],
        },
    )
    configure_pixel(run, checkpoint, selected)
    pixel.main()
    reliability = importlib.import_module("fit_segmenter_case_reliability")
    configure_reliability(reliability, run, checkpoint)
    reliability.main()


if __name__ == "__main__":
    main()
