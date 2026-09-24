"""Close the preregistered E08 ensemble after visual review."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e08_ensemble"
ARTIFACTS = EXPERIMENT / "artifacts"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def write_text(path, value):
    Path(path).write_bytes(value.replace("\r\n", "\n").encode("utf-8"))


def main():
    config_path = EXPERIMENT / "config.json"
    result_path = EXPERIMENT / "result.json"
    config = read_json(config_path)
    result = read_json(result_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E08 is not in its registered pre-close state")
    if result["status"] != "EVALUATION_COMPLETE_VISUAL_REVIEW_PENDING":
        raise ValueError("E08 evaluation is incomplete")
    if result["gate"]["status"] != "PASS" or not all(
        result["gate"]["checks"].values()
    ):
        raise ValueError("E08 expected a passed local gate")
    with (ARTIFACTS / "per_case_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E08 comparison population mismatch")
    visual = {
        "status": "PASS_NO_SYSTEMATIC_TECHNICAL_DEFECT",
        "selected_tiles": 42,
        "finding": (
            "Best cases show close reference/prediction alignment. Worst cases show "
            "mislocalization, fragmentation, and coverage errors without a repeated "
            "flip, stretch, offset, or image/mask pairing defect."
        ),
        "reviewer": "Codex visual inspection under the user-authorized audit protocol",
        "test_images_accessed": 0,
    }
    write_json(ARTIFACTS / "visual_review.json", visual)
    result["status"] = "PASS"
    result["gate_passed"] = True
    result["visual_review"] = visual
    result["parent_g1"] = {
        "status": "FAIL",
        "required_glioma_dice_delta_over_v1": 0.05,
        "observed_glioma_dice_delta_over_v1": result["mean_dice"]
        ["ensemble_delta_vs_v1"]["glioma"],
        "consequence": "E08 passes its local gate but cannot replace frozen v1 under the parent model-replacement rule.",
    }
    write_json(result_path, result)
    config["status"] = "COMPLETE"
    write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    item = next(
        entry for entry in registry["experiments"] if entry["id"] == "e08_ensemble"
    )
    item["status"] = "COMPLETE"
    write_json(registry_path, registry)
    mean = result["mean_dice"]
    write_text(
        EXPERIMENT / "decision.md",
        f"""# Fixed three-system probability ensemble decision

**Decision:** ACCEPTED LOCALLY

The fixed equal-weight ensemble changed glioma Dice by `{mean['ensemble_delta_vs_best_eligible_single']['glioma']:+.4f}` over E07, exceeding the local `+0.0200` requirement. Meningioma and pituitary changed by `{mean['ensemble_delta_vs_v1']['meningioma']:+.4f}` and `{mean['ensemble_delta_vs_v1']['pituitary']:+.4f}` against the v1 harm floor of `-0.0100`. All three local checks passed, and the gate was not altered.

The ensemble's glioma change over frozen v1 is only `{mean['ensemble_delta_vs_v1']['glioma']:+.4f}`, below parent G1's `+0.0500` replacement requirement. E08 therefore remains an accepted local experiment and does not replace v1.

All 680 component predictions reproduced exactly, visual review found no systematic technical defect, and no sealed-test image was accessed. This is an internal BRISC development result, not clinical validation.
""",
    )
    write_text(
        ARTIFACTS / "summary.md",
        f"""# E08 fixed-ensemble summary

**Local decision:** PASS

- Ensemble Dice: glioma `{mean['ensemble']['glioma']:.6f}`, meningioma `{mean['ensemble']['meningioma']:.6f}`, pituitary `{mean['ensemble']['pituitary']:.6f}`
- Glioma change over best eligible v2 single: `{mean['ensemble_delta_vs_best_eligible_single']['glioma']:+.6f}`
- Glioma change over frozen v1: `{mean['ensemble_delta_vs_v1']['glioma']:+.6f}`
- Parent G1 replacement gate: `FAIL` (`+0.05` required)
- Sealed-test images accessed: `0`
""",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
