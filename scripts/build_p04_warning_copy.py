"""Build and verify the fixed P04 user-facing message catalogue."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/p04_warning_copy"
ARTIFACTS = EXPERIMENT / "artifacts"
DOCUMENT = ROOT / "docs/warning_copy.md"

THRESHOLD = 0.9490250173235835
THRESHOLD_TEXT = "0.9490250173235835"
DISCLAIMER = "Research prototype \u2014 not for clinical use"
EXPECTED_HASHES = {
    "experiments/p03_decision_rule/result.json": "5b7573987e92b1a82461e92e37efd68ba46bdfc0bc0cc426c1128218e1944011",
    "experiments/p03_decision_rule/artifacts/experiment_manifest.json": "2a2bbc1198ae08e4b206ee91c0bdf9c0ac7d45c4d80ffff7eaf982a769038377",
    "docs/decision_rule.md": "43290ba09ec991819dbbc21346b447cd70c0cfcda1e75b0d21e99b502fd744c5",
}
PROHIBITED_TERMS = {
    "accurate",
    "accuracy",
    "clinically validated",
    "confirmed",
    "correct diagnosis",
    "diagnose",
    "diagnosed",
    "diagnosis",
    "diagnostic",
    "guaranteed",
    "safe",
    "trusted",
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


def make_message(message_id, status, reason, category_condition, severity, title, body):
    return {
        "message_id": message_id,
        "status": status,
        "reason": reason,
        "predicted_category_condition": category_condition,
        "threshold": THRESHOLD,
        "severity": severity,
        "title": title,
        "body": body,
        "disclaimer": DISCLAIMER,
        "full_text": f"{title}\n{body}\n{DISCLAIMER}",
    }


def build_messages():
    return [
        make_message(
            "standard_default",
            "STANDARD",
            "AT_OR_ABOVE_THRESHOLD",
            "ANY",
            "INFO",
            "Result ready for review",
            f"This output met the research prototype's internal reliability threshold of {THRESHOLD_TEXT} and can be shown in the standard view. Review the uploaded image, category label, and highlighted region before using the output.",
        ),
        make_message(
            "standard_predicted_glioma",
            "STANDARD",
            "AT_OR_ABOVE_THRESHOLD",
            "PREDICTED_GLIOMA",
            "INFO",
            "Result ready for review",
            f"This output met the research prototype's internal reliability threshold of {THRESHOLD_TEXT} and can be shown in the standard view. The model-predicted category is glioma. In internal development testing, reliability ranking was weaker for glioma images. This does not change the threshold or status. Review the image, category label, and highlighted region carefully.",
        ),
        make_message(
            "review_below_threshold_default",
            "REVIEW_REQUIRED",
            "BELOW_THRESHOLD",
            "ANY",
            "HIGH",
            "Manual review required",
            f"This output did not meet the research prototype's internal reliability threshold of {THRESHOLD_TEXT}. Inspect the uploaded image and predicted region carefully before relying on the output.",
        ),
        make_message(
            "review_below_threshold_predicted_glioma",
            "REVIEW_REQUIRED",
            "BELOW_THRESHOLD",
            "PREDICTED_GLIOMA",
            "HIGH",
            "Manual review required",
            f"This output did not meet the research prototype's internal reliability threshold of {THRESHOLD_TEXT}. The model-predicted category is glioma, where reliability ranking was weaker in internal development testing. Inspect the uploaded image, category label, and predicted region carefully before relying on the output.",
        ),
        make_message(
            "review_unavailable_default",
            "REVIEW_REQUIRED",
            "RELIABILITY_UNAVAILABLE",
            "ANY",
            "HIGH",
            "Manual review required",
            f"A reliability score could not be computed. The output has been placed in Review Required. The policy threshold is {THRESHOLD_TEXT}, but no valid score was available for comparison. Inspect the uploaded image and predicted region carefully.",
        ),
        make_message(
            "review_unavailable_predicted_glioma",
            "REVIEW_REQUIRED",
            "RELIABILITY_UNAVAILABLE",
            "PREDICTED_GLIOMA",
            "HIGH",
            "Manual review required",
            f"A reliability score could not be computed. The output has been placed in Review Required. The policy threshold is {THRESHOLD_TEXT}, but no valid score was available for comparison. The model-predicted category is glioma. In internal development testing, reliability ranking was weaker for glioma images. Inspect the uploaded image, category label, and predicted region carefully.",
        ),
        make_message(
            "review_invalid_default",
            "REVIEW_REQUIRED",
            "RELIABILITY_INVALID",
            "ANY",
            "HIGH",
            "Manual review required",
            f"The reliability output was invalid. The output has been placed in Review Required. A finite score between 0 and 1 was expected, so the policy threshold of {THRESHOLD_TEXT} could not be applied. Inspect the uploaded image and predicted region carefully.",
        ),
        make_message(
            "review_invalid_predicted_glioma",
            "REVIEW_REQUIRED",
            "RELIABILITY_INVALID",
            "PREDICTED_GLIOMA",
            "HIGH",
            "Manual review required",
            f"The reliability output was invalid. The output has been placed in Review Required. A finite score between 0 and 1 was expected, so the policy threshold of {THRESHOLD_TEXT} could not be applied. The model-predicted category is glioma. In internal development testing, reliability ranking was weaker for glioma images. Inspect the uploaded image, category label, and predicted region carefully.",
        ),
    ]


def expected_combinations():
    return {
        ("STANDARD", "AT_OR_ABOVE_THRESHOLD", "ANY"),
        ("STANDARD", "AT_OR_ABOVE_THRESHOLD", "PREDICTED_GLIOMA"),
        ("REVIEW_REQUIRED", "BELOW_THRESHOLD", "ANY"),
        ("REVIEW_REQUIRED", "BELOW_THRESHOLD", "PREDICTED_GLIOMA"),
        ("REVIEW_REQUIRED", "RELIABILITY_UNAVAILABLE", "ANY"),
        ("REVIEW_REQUIRED", "RELIABILITY_UNAVAILABLE", "PREDICTED_GLIOMA"),
        ("REVIEW_REQUIRED", "RELIABILITY_INVALID", "ANY"),
        ("REVIEW_REQUIRED", "RELIABILITY_INVALID", "PREDICTED_GLIOMA"),
    }


def validate_messages(messages):
    if len(messages) != 8:
        raise ValueError(f"Expected eight messages, found {len(messages)}")
    if len({message["message_id"] for message in messages}) != 8:
        raise ValueError("Message IDs are not unique")
    combinations = {
        (message["status"], message["reason"], message["predicted_category_condition"])
        for message in messages
    }
    if combinations != expected_combinations():
        raise ValueError("Message combinations do not match the contract")

    for message in messages:
        full = message["full_text"]
        lower = full.lower()
        if message["threshold"] != THRESHOLD or THRESHOLD_TEXT not in full:
            raise ValueError(f"Threshold missing or changed: {message['message_id']}")
        if message["disclaimer"] != DISCLAIMER or DISCLAIMER not in full:
            raise ValueError(f"Disclaimer missing or changed: {message['message_id']}")
        if message["status"] not in {"STANDARD", "REVIEW_REQUIRED"}:
            raise ValueError(f"Unsupported state: {message['message_id']}")
        if "WARNING" in message["status"]:
            raise ValueError(f"Warning policy state found: {message['message_id']}")
        if message["predicted_category_condition"] == "PREDICTED_GLIOMA":
            if "model-predicted category is glioma" not in lower:
                raise ValueError(f"Glioma context is not prediction-qualified: {message['message_id']}")
            if "weaker" not in lower:
                raise ValueError(f"Glioma limitation missing: {message['message_id']}")
        elif "glioma" in lower:
            raise ValueError(f"General message contains glioma context: {message['message_id']}")
        if (
            message["reason"] in {"RELIABILITY_UNAVAILABLE", "RELIABILITY_INVALID"}
            and (message["status"] != "REVIEW_REQUIRED" or message["severity"] != "HIGH")
        ):
            raise ValueError(f"Fallback does not require review: {message['message_id']}")
        for term in PROHIBITED_TERMS:
            if term in lower:
                raise ValueError(f"Prohibited claim '{term}' in {message['message_id']}")
        if re.search(r"\b\d+(?:\.\d+)?%", full):
            raise ValueError(f"Percentage claim in {message['message_id']}")
        if len(message["title"].split()) > 5:
            raise ValueError(f"Title is not concise: {message['message_id']}")
        if len(message["body"].split()) > 85:
            raise ValueError(f"Body is not plain and concise: {message['message_id']}")
        if full != f"{message['title']}\n{message['body']}\n{message['disclaimer']}":
            raise ValueError(f"Full text was not composed exactly: {message['message_id']}")
    return {
        "message_count": len(messages),
        "unique_ids": True,
        "complete_combination_coverage": True,
        "exact_threshold_in_every_message": True,
        "exact_disclaimer_in_every_message": True,
        "glioma_copy_prediction_qualified": True,
        "fallbacks_require_review": True,
        "warning_policy_state_absent": True,
        "prohibited_claims_absent": True,
        "percentage_claims_absent": True,
        "plain_language_limits_passed": True,
    }


def verify_sources():
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"INVALID/STOP hash mismatch: {relative}: {actual}")
    config = read_json(EXPERIMENT / "config.json")
    p03 = read_json(ROOT / "experiments/p03_decision_rule/result.json")
    if config["reliability_threshold"] != THRESHOLD or p03["policy"]["threshold"] != THRESHOLD:
        raise ValueError("INVALID/STOP threshold mismatch")
    if config["required_disclaimer"] != DISCLAIMER:
        raise ValueError("INVALID/STOP disclaimer mismatch")
    if config["allowed_states"] != ["STANDARD", "REVIEW_REQUIRED"]:
        raise ValueError("INVALID/STOP policy-state mismatch")
    if p03["policy"]["warning_state"] is not None:
        raise ValueError("INVALID/STOP P03 warning state changed")
    if config["segmentation_threshold"] != 0.5:
        raise ValueError("INVALID/STOP segmentation threshold changed")
    if config["test_images_accessed"] != 0:
        raise ValueError("INVALID/STOP test access declared")
    if config["model_retrained_or_refit"] or config["gate_or_policy_relaxed"]:
        raise ValueError("INVALID/STOP refit or relaxation declared")
    return {
        "status": "PASS",
        "hashes_verified": len(EXPECTED_HASHES),
        "threshold_unchanged": True,
        "disclaimer_unchanged": True,
        "two_state_policy_unchanged": True,
        "segmentation_threshold": 0.5,
        "segmentation_threshold_changed": False,
        "test_images_accessed": 0,
        "model_retrained_or_refit": False,
        "gate_or_policy_relaxed": False,
    }


def catalogue_document(messages):
    sections = []
    for message in messages:
        sections.append(
            f"## `{message['message_id']}`\n\n"
            f"- Status: `{message['status']}`\n"
            f"- Reason: `{message['reason']}`\n"
            f"- Category condition: `{message['predicted_category_condition']}`\n"
            f"- Severity: `{message['severity']}`\n\n"
            f"> **{message['title']}**\n>\n"
            f"> {message['body']}\n>\n"
            f"> **{message['disclaimer']}**\n"
        )
    return """# User-facing reliability messages

## Usage contract

These eight complete messages implement the frozen P03 two-state reliability policy. The application selects a message by policy status, reason code, and whether the classifier-predicted category is glioma. True category never selects copy.

The policy threshold is `0.9490250173235835`. There is no Warning policy state. Every message includes the exact disclaimer **Research prototype — not for clinical use**.

Meeting the threshold means only that the output reached the registered internal reliability-policy operating point. It is not a promise of correctness or clinical validity.

""" + "\n".join(sections) + """
## Integration notes

- Display the title, body, and disclaimer together; do not omit or replace the disclaimer.
- Use the predicted-glioma variant only when the classifier output is glioma.
- Missing and invalid reliability always use `REVIEW_REQUIRED` variants.
- Keep the prediction visible in Review Required so a user can inspect it.
- Do not create a Warning state or round the stored policy threshold in machine-readable data.
"""


def main():
    source_verification = verify_sources()
    messages = build_messages()
    copy_verification = validate_messages(messages)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    keep = ARTIFACTS / ".gitkeep"
    if keep.exists():
        keep.unlink()

    catalogue = {
        "schema_version": 1,
        "catalogue_id": "p04_warning_copy",
        "policy_version": "p03_decision_rule",
        "threshold": THRESHOLD,
        "required_disclaimer": DISCLAIMER,
        "allowed_states": ["STANDARD", "REVIEW_REQUIRED"],
        "messages": messages,
    }
    write_json(ARTIFACTS / "message_catalogue.json", catalogue)
    write_text(DOCUMENT, catalogue_document(messages))

    result = {
        "schema_version": 1,
        "experiment_id": "p04_warning_copy",
        "status": "PASS",
        "decision": "ADOPT_FIXED_USER_MESSAGE_CATALOGUE",
        "catalogue": {
            "messages": len(messages),
            "general_variants": 4,
            "predicted_glioma_variants": 4,
            "states": ["STANDARD", "REVIEW_REQUIRED"],
            "warning_state": None,
            "threshold": THRESHOLD,
            "required_disclaimer": DISCLAIMER,
        },
        "verification": {
            **source_verification,
            **copy_verification,
            "source_hashes": EXPECTED_HASHES,
            "script_sha256": sha256(__file__),
        },
        "limitations": [
            "The messages describe an internal research policy and do not establish clinical validity.",
            "The copy has not undergone user testing, accessibility testing, translation review, or clinical stakeholder review.",
            "The glioma note describes weaker internal reliability ranking and must not be interpreted as patient-specific risk.",
        ],
    }

    config = read_json(EXPERIMENT / "config.json")
    config["status"] = "COMPLETE"
    write_json(EXPERIMENT / "config.json", config)
    write_json(EXPERIMENT / "result.json", result)
    write_text(
        EXPERIMENT / "decision.md",
        "# P04 decision\n\n"
        "**Decision: PASS — adopt the fixed eight-message catalogue.**\n\n"
        f"Every message references threshold `{THRESHOLD_TEXT}`, includes the exact required disclaimer, and uses only "
        "`STANDARD` or `REVIEW_REQUIRED`. General and predicted-glioma variants cover every P03 reason code. "
        "Unavailable and invalid reliability fail closed to manual review.\n\n"
        "No clinical or accuracy claim, percentage promise, Warning policy state, test access, segmentation-threshold "
        "change, retraining, refitting, gate relaxation, or policy relaxation occurred. Stage 5.5 remains unstarted "
        "pending user approval.\n",
    )

    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    matches = [item for item in registry["experiments"] if item["id"] == "p04_warning_copy"]
    if len(matches) != 1:
        raise ValueError("P04 registry identity mismatch")
    matches[0]["status"] = "COMPLETE"
    write_json(registry_path, registry)

    manifest_paths = [
        "docs/warning_copy.md",
        "experiments/p04_warning_copy/hypothesis.md",
        "experiments/p04_warning_copy/config.json",
        "experiments/p04_warning_copy/result.json",
        "experiments/p04_warning_copy/decision.md",
        "experiments/p04_warning_copy/artifacts/message_catalogue.json",
        "scripts/build_p04_warning_copy.py",
        "tests/test_build_p04_warning_copy.py",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "p04_warning_copy",
        "status": "PASS",
        "decision": result["decision"],
        "threshold": THRESHOLD,
        "message_count": len(messages),
        "test_images_accessed": 0,
        "segmentation_threshold_changed": False,
        "model_retrained_or_refit": False,
        "gate_or_policy_relaxed": False,
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
                "messages": len(messages),
                "verification": copy_verification,
                "manifest_sha256": sha256(ARTIFACTS / "experiment_manifest.json"),
                "result_sha256": sha256(EXPERIMENT / "result.json"),
                "document_sha256": sha256(DOCUMENT),
                "catalogue_sha256": sha256(ARTIFACTS / "message_catalogue.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
