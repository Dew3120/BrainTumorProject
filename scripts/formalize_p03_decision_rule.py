"""Formalize and verify the frozen P03 two-state reliability policy."""

import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/p03_decision_rule"
ARTIFACTS = EXPERIMENT / "artifacts"
OOF = ROOT / "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv"
DOCUMENT = ROOT / "docs/decision_rule.md"

POLICY_VERSION = "p03_decision_rule"
THRESHOLD = 0.9490250173235835
EXPECTED_HASHES = {
    "experiments/p01_cost_model/result.json": "d66042c2ce9ea848c6e98e8a06a26c9aace02a4f54b253de893b1ccafb531109",
    "experiments/p01_cost_model/artifacts/experiment_manifest.json": "ef52545bc645be0a453978046cdf094546408091d68c6ad8f15b5f99230bab50",
    "experiments/p02_threshold_selection/result.json": "ab86f12bb3116429bbfeb633cdef0ea0f636623757202d821e3f5b52a2fac05c",
    "experiments/p02_threshold_selection/artifacts/experiment_manifest.json": "80e1b9f3f655d43c451a194b7a9b5ddd6b6cc3de930cbd961855f880fe8196ef",
    "experiments/r02_reduced_features/result.json": "fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2",
    "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv": "9b986435b0da0fdb340a16730209ebb895115ece694386c298830e3956b2fc6a",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def write_text(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def apply_policy(value):
    """Return the normalized, fail-closed policy output for one score."""
    base = {
        "reliability_threshold": THRESHOLD,
        "policy_version": POLICY_VERSION,
    }
    if value is None:
        return {
            "reliability_score": None,
            **base,
            "reliability_status": "REVIEW_REQUIRED",
            "requires_manual_review": True,
            "status_reason": "RELIABILITY_UNAVAILABLE",
        }
    if isinstance(value, bool) or not isinstance(value, Real):
        return {
            "reliability_score": None,
            **base,
            "reliability_status": "REVIEW_REQUIRED",
            "requires_manual_review": True,
            "status_reason": "RELIABILITY_INVALID",
        }
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return {
            "reliability_score": None,
            **base,
            "reliability_status": "REVIEW_REQUIRED",
            "requires_manual_review": True,
            "status_reason": "RELIABILITY_INVALID",
        }
    if score < THRESHOLD:
        return {
            "reliability_score": score,
            **base,
            "reliability_status": "REVIEW_REQUIRED",
            "requires_manual_review": True,
            "status_reason": "BELOW_THRESHOLD",
        }
    return {
        "reliability_score": score,
        **base,
        "reliability_status": "STANDARD",
        "requires_manual_review": False,
        "status_reason": "AT_OR_ABOVE_THRESHOLD",
    }


def verify_and_load():
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"INVALID/STOP hash mismatch: {relative}: {actual}")

    config = read_json(EXPERIMENT / "config.json")
    p01 = read_json(ROOT / "experiments/p01_cost_model/result.json")
    p02 = read_json(ROOT / "experiments/p02_threshold_selection/result.json")
    if config["reliability_threshold"] != THRESHOLD:
        raise ValueError("INVALID/STOP P03 threshold changed")
    if p02["binary_selection"]["threshold"] != THRESHOLD:
        raise ValueError("INVALID/STOP P02 threshold mismatch")
    if p02["three_state_selection"]["theta_low"] != THRESHOLD:
        raise ValueError("INVALID/STOP theta_low mismatch")
    if p02["three_state_selection"]["theta_high"] != THRESHOLD:
        raise ValueError("INVALID/STOP theta_high mismatch")
    if config["binary_costs"] != {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 10,
    }:
        raise ValueError("INVALID/STOP P03 costs changed")
    if p01["cost_matrix"]["false_negative_accepted_bad"] != 10:
        raise ValueError("INVALID/STOP P01 costs changed")
    if config["segmentation_threshold"] != 0.5:
        raise ValueError("INVALID/STOP segmentation threshold changed")
    if config["model_retrained_or_refit"]:
        raise ValueError("INVALID/STOP model refitting permitted")

    with OOF.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("INVALID/STOP OOF population mismatch")
    if any("/test/" in row["path"].replace("\\", "/").lower() for row in rows):
        raise ValueError("INVALID/STOP sealed-test path detected")
    folds = sorted({int(row["fold"]) for row in rows})
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError("INVALID/STOP OOF fold mismatch")

    return rows, {
        "status": "PASS",
        "hashes_verified": len(EXPECTED_HASHES),
        "cases": 680,
        "unique_cases": 680,
        "folds": folds,
        "threshold_unchanged_from_p02": True,
        "cost_vector_unchanged": True,
        "segmentation_threshold": 0.5,
        "segmentation_threshold_changed": False,
        "test_images_accessed": 0,
        "model_retrained_or_refit": False,
    }


def policy_schema():
    common_properties = {
        "reliability_score": {"type": ["number", "null"]},
        "reliability_threshold": {"const": THRESHOLD},
        "reliability_status": {"enum": ["STANDARD", "REVIEW_REQUIRED"]},
        "requires_manual_review": {"type": "boolean"},
        "status_reason": {
            "enum": [
                "AT_OR_ABOVE_THRESHOLD",
                "BELOW_THRESHOLD",
                "RELIABILITY_UNAVAILABLE",
                "RELIABILITY_INVALID",
            ]
        },
        "policy_version": {"const": POLICY_VERSION},
    }
    required = list(common_properties)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/brain-tumor/reliability-policy-output.schema.json",
        "title": "Reliability policy output",
        "type": "object",
        "additionalProperties": False,
        "properties": common_properties,
        "required": required,
        "oneOf": [
            {
                "properties": {
                    "reliability_score": {"type": "number", "minimum": THRESHOLD, "maximum": 1},
                    "reliability_status": {"const": "STANDARD"},
                    "requires_manual_review": {"const": False},
                    "status_reason": {"const": "AT_OR_ABOVE_THRESHOLD"},
                }
            },
            {
                "properties": {
                    "reliability_score": {"type": "number", "minimum": 0, "exclusiveMaximum": THRESHOLD},
                    "reliability_status": {"const": "REVIEW_REQUIRED"},
                    "requires_manual_review": {"const": True},
                    "status_reason": {"const": "BELOW_THRESHOLD"},
                }
            },
            {
                "properties": {
                    "reliability_score": {"type": "null"},
                    "reliability_status": {"const": "REVIEW_REQUIRED"},
                    "requires_manual_review": {"const": True},
                    "status_reason": {"const": "RELIABILITY_UNAVAILABLE"},
                }
            },
            {
                "properties": {
                    "reliability_score": {"type": "null"},
                    "reliability_status": {"const": "REVIEW_REQUIRED"},
                    "requires_manual_review": {"const": True},
                    "status_reason": {"const": "RELIABILITY_INVALID"},
                }
            },
        ],
    }


def policy_examples():
    below = float(np.nextafter(THRESHOLD, -np.inf))
    above = float(np.nextafter(THRESHOLD, np.inf))
    inputs = [
        ("immediately_below", "number", below, repr(below)),
        ("exact_threshold", "number", THRESHOLD, repr(THRESHOLD)),
        ("immediately_above", "number", above, repr(above)),
        ("minimum_valid", "number", 0.0, "0.0"),
        ("maximum_valid", "number", 1.0, "1.0"),
        ("unavailable", "null", None, "null"),
        ("negative", "number", -0.1, "-0.1"),
        ("above_one", "number", 1.1, "1.1"),
        ("nan", "non_finite_number", float("nan"), "NaN"),
        ("positive_infinity", "non_finite_number", float("inf"), "+Infinity"),
        ("negative_infinity", "non_finite_number", float("-inf"), "-Infinity"),
        ("string", "string", "0.99", "\"0.99\""),
        ("boolean", "boolean", True, "true"),
    ]
    return [
        {
            "case": name,
            "input_kind": kind,
            "input_display": display,
            "output": apply_policy(value),
        }
        for name, kind, value, display in inputs
    ]


def document(result):
    behavior = result["oof_behavior"]
    return f"""# Reliability decision rule

## Policy

The internal-development reliability policy for the C03-calibrated E08 segmenter and R02 reliability model is:

```text
RELIABILITY_THRESHOLD = {THRESHOLD:.17g}

if reliability_score < RELIABILITY_THRESHOLD:
    status = REVIEW_REQUIRED
else:
    status = STANDARD
```

Equality belongs to `STANDARD`. The threshold came from P02's exhaustive internal-validation cost sweep under P01 costs `TP=1`, `FP=1`, `TN=0`, `FN=10`.

## Why there is no Warning state

P02 evaluated 230,860 valid three-state threshold pairs and selected `theta_low = theta_high = {THRESHOLD:.17g}`. Therefore the interval `theta_low <= r < theta_high` is empty. A nonempty Warning band increased expected cost under the registered weights, so P03 does not invent one.

## Expected OOF behavior

| Measure | Count |
|---|---:|
| Review required | {behavior['review_required']} |
| Standard | {behavior['standard']} |
| Failures caught | {behavior['failures_caught']} |
| Failures missed | {behavior['failures_missed']} |
| Acceptable masks reviewed | {behavior['acceptable_reviewed']} |
| Total relative cost | {behavior['total_cost']} |

These are internal-development estimates on 680 OOF BRISC cases, not prospective or clinical performance.

## Glioma policy

The descriptive glioma optimum was `0.950106439123928`, while the global threshold is `{THRESHOLD:.17g}`. They change zero assignments and have identical subgroup cost. Glioma does not receive a separate policy.

Glioma remains the weakest true-category reliability subgroup, with R02 AUROC `0.876843` (approximately `0.877`). The global rule applies uniformly; weaker subgroup ranking is retained as a limitation rather than used to create a post-hoc threshold.

## Phase 6 integration contract

Every inference result must provide:

- `reliability_score`: a finite number in `[0,1]`, or `null` after invalid/unavailable normalization
- `reliability_threshold`: exactly `{THRESHOLD:.17g}`
- `reliability_status`: `STANDARD` or `REVIEW_REQUIRED`
- `requires_manual_review`: Boolean
- `status_reason`: a registered reason code
- `policy_version`: `p03_decision_rule`

### STANDARD

Display the prediction normally together with its reliability score and status. Final research-only wording is defined separately in Stage 5.4.

### REVIEW_REQUIRED

Keep the prediction visible for inspection and show a prominent manual-review warning. Do not present it with normal/standard styling.

### Missing or invalid reliability

Missing, nonnumeric, non-finite, negative, or greater-than-one reliability fails closed to `REVIEW_REQUIRED`. The normalized output score is `null`, and `status_reason` distinguishes `RELIABILITY_UNAVAILABLE` from `RELIABILITY_INVALID`.

## Boundaries and prohibited inputs

- Immediately below the threshold: `REVIEW_REQUIRED`
- Exactly equal to the threshold: `STANDARD`
- Immediately above the threshold: `STANDARD`
- `None`: `REVIEW_REQUIRED / RELIABILITY_UNAVAILABLE`
- NaN, infinities, Boolean, string, negative, or greater than one: `REVIEW_REQUIRED / RELIABILITY_INVALID`

True tumor type, reference-mask information, patient identity, and test membership do not enter this policy.

## Limits

This policy was optimized and verified on internal BRISC validation. It does not establish clinical validity, human-review effectiveness, external generalization, or real workflow utility. The segmentation mask threshold remains `0.50`; the reliability-policy threshold does not alter the predicted mask.
"""


def main():
    source_rows, verification = verify_and_load()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    keep = ARTIFACTS / ".gitkeep"
    if keep.exists():
        keep.unlink()

    assignments = []
    counts = {"review_required": 0, "standard": 0, "failures_caught": 0, "failures_missed": 0, "acceptable_reviewed": 0, "total_cost": 0}
    for row in source_rows:
        score = float(row["reduced_oof_reliability"])
        success = bool(int(row["success_dice_at_least_0_50"]))
        dice = float(row["calibrated_dice"])
        if success != (dice >= 0.5):
            raise ValueError("INVALID/STOP Dice target mismatch")
        policy = apply_policy(score)
        review = policy["reliability_status"] == "REVIEW_REQUIRED"
        if review:
            counts["review_required"] += 1
        else:
            counts["standard"] += 1
        if review and not success:
            outcome, cost = "TP", 1
            counts["failures_caught"] += 1
        elif review and success:
            outcome, cost = "FP", 1
            counts["acceptable_reviewed"] += 1
        elif not review and success:
            outcome, cost = "TN", 0
        else:
            outcome, cost = "FN", 10
            counts["failures_missed"] += 1
        counts["total_cost"] += cost
        assignments.append(
            {
                "path": row["path"],
                "true_label_evaluation_only": row["true_label"],
                "fold": int(row["fold"]),
                "calibrated_dice_evaluation_only": dice,
                "success_dice_at_least_0_50": int(success),
                **policy,
                "evaluation_outcome": outcome,
                "relative_cost": cost,
            }
        )

    expected = {
        "review_required": 195,
        "standard": 485,
        "failures_caught": 81,
        "failures_missed": 1,
        "acceptable_reviewed": 114,
        "total_cost": 205,
    }
    if counts != expected:
        raise ValueError(f"INVALID/STOP OOF behavior mismatch: {counts}")
    if {row["reliability_status"] for row in assignments} != {"STANDARD", "REVIEW_REQUIRED"}:
        raise ValueError("INVALID/STOP state contract mismatch")

    with (ARTIFACTS / "oof_policy_assignments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    write_json(ARTIFACTS / "policy_contract.schema.json", policy_schema())
    examples = policy_examples()
    write_json(ARTIFACTS / "policy_examples.json", {"policy_version": POLICY_VERSION, "examples": examples})

    r02 = read_json(ROOT / "experiments/r02_reduced_features/result.json")
    glioma_auroc = r02["reduced_10_feature_model"]["true_category_aurocs_descriptive"]["glioma"]["failure_detection_auroc"]
    result = {
        "schema_version": 1,
        "experiment_id": "p03_decision_rule",
        "status": "PASS",
        "decision": "ADOPT_TWO_STATE_POLICY_CONTRACT",
        "policy": {
            "version": POLICY_VERSION,
            "threshold": THRESHOLD,
            "states": ["STANDARD", "REVIEW_REQUIRED"],
            "review_rule": "r < threshold",
            "standard_rule": "r >= threshold",
            "warning_state": None,
            "warning_absence_reason": "P02 selected theta_low equal to theta_high after exhaustive registered cost minimization.",
            "invalid_or_unavailable_fallback": "REVIEW_REQUIRED",
        },
        "oof_behavior": counts,
        "glioma": {
            "global_threshold": THRESHOLD,
            "descriptive_optimum": 0.950106439123928,
            "assignment_differences": 0,
            "cost_difference": 0,
            "separate_policy_required": False,
            "true_category_reliability_auroc": glioma_auroc,
        },
        "ui_contract": {
            "schema": "artifacts/policy_contract.schema.json",
            "standard_behavior": "Display prediction normally with score and policy status.",
            "review_required_behavior": "Keep prediction visible and display a prominent manual-review warning.",
            "missing_or_invalid_behavior": "Fail closed to REVIEW_REQUIRED with a reason code and normalized null score.",
        },
        "verification": {
            **verification,
            "source_hashes": EXPECTED_HASHES,
            "boundary_tests_required": True,
            "oof_behavior_reproduced": True,
            "only_registered_states_emitted": True,
            "true_category_used_by_policy": False,
            "reference_mask_used_by_policy": False,
            "script_sha256": sha256(__file__),
        },
        "limitations": [
            "The policy is optimized on internal BRISC validation and awaits sealed evaluation.",
            "The costs are engineering preferences rather than clinically elicited utilities.",
            "Glioma reliability ranking remains weaker even though its threshold assignments match the global policy.",
            "The policy does not establish the effectiveness or consistency of human review.",
        ],
    }

    config = read_json(EXPERIMENT / "config.json")
    config["status"] = "COMPLETE"
    write_json(EXPERIMENT / "config.json", config)
    write_json(EXPERIMENT / "result.json", result)
    write_text(
        EXPERIMENT / "decision.md",
        "# P03 decision\n\n"
        "**Decision: PASS — adopt the two-state reliability policy contract.**\n\n"
        f"Scores below `{THRESHOLD:.17g}` are `REVIEW_REQUIRED`; scores at or above it are `STANDARD`. "
        "Missing or invalid scores fail closed to `REVIEW_REQUIRED`. There is no Warning state because P02's "
        "cost optimum set `theta_low` equal to `theta_high`.\n\n"
        "P03 reproduced 195 Review Required, 485 Standard, 81 failures caught, one missed failure, and total cost 205. "
        "No test artifact was accessed, the segmentation threshold stayed `0.50`, and no model was trained or refit. "
        "Stage 5.4 remains unstarted pending user approval.\n",
    )
    write_text(DOCUMENT, document(result))

    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    matches = [item for item in registry["experiments"] if item["id"] == "p03_decision_rule"]
    if len(matches) != 1:
        raise ValueError("P03 registry identity mismatch")
    matches[0]["status"] = "COMPLETE"
    write_json(registry_path, registry)

    manifest_paths = [
        "docs/decision_rule.md",
        "experiments/p03_decision_rule/hypothesis.md",
        "experiments/p03_decision_rule/config.json",
        "experiments/p03_decision_rule/result.json",
        "experiments/p03_decision_rule/decision.md",
        "experiments/p03_decision_rule/artifacts/policy_contract.schema.json",
        "experiments/p03_decision_rule/artifacts/oof_policy_assignments.csv",
        "experiments/p03_decision_rule/artifacts/policy_examples.json",
        "scripts/formalize_p03_decision_rule.py",
        "tests/test_formalize_p03_decision_rule.py",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "p03_decision_rule",
        "status": "PASS",
        "decision": result["decision"],
        "threshold": THRESHOLD,
        "states": ["STANDARD", "REVIEW_REQUIRED"],
        "test_images_accessed": 0,
        "segmentation_threshold_changed": False,
        "model_retrained_or_refit": False,
        "files": [
            {
                "path": relative,
                "sha256": sha256(ROOT / relative),
                "bytes": (ROOT / relative).stat().st_size,
            }
            for relative in manifest_paths
        ],
    }
    write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "threshold": THRESHOLD,
                "oof_behavior": counts,
                "glioma_auroc": glioma_auroc,
                "examples": len(examples),
                "manifest_sha256": sha256(ARTIFACTS / "experiment_manifest.json"),
                "result_sha256": sha256(EXPERIMENT / "result.json"),
                "document_sha256": sha256(DOCUMENT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
