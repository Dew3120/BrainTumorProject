from build_p04_warning_copy import (
    DISCLAIMER,
    PROHIBITED_TERMS,
    THRESHOLD,
    THRESHOLD_TEXT,
    build_messages,
    expected_combinations,
    validate_messages,
)


def test_catalogue_has_exactly_the_eight_registered_combinations():
    messages = build_messages()
    combinations = {
        (message["status"], message["reason"], message["predicted_category_condition"])
        for message in messages
    }
    assert len(messages) == 8
    assert combinations == expected_combinations()
    assert validate_messages(messages)["complete_combination_coverage"] is True


def test_every_message_contains_exact_threshold_and_disclaimer():
    for message in build_messages():
        assert message["threshold"] == THRESHOLD
        assert THRESHOLD_TEXT in message["body"]
        assert THRESHOLD_TEXT in message["full_text"]
        assert message["disclaimer"] == DISCLAIMER
        assert DISCLAIMER in message["full_text"]


def test_glioma_copy_is_prediction_qualified_and_general_copy_is_not_glioma_specific():
    for message in build_messages():
        lower = message["full_text"].lower()
        if message["predicted_category_condition"] == "PREDICTED_GLIOMA":
            assert "model-predicted category is glioma" in lower
            assert "weaker" in lower
        else:
            assert "glioma" not in lower


def test_fallback_messages_always_require_review():
    fallback_reasons = {"RELIABILITY_UNAVAILABLE", "RELIABILITY_INVALID"}
    for message in build_messages():
        if message["reason"] in fallback_reasons:
            assert message["status"] == "REVIEW_REQUIRED"
            assert message["severity"] == "HIGH"


def test_no_warning_state_or_prohibited_claim_is_present():
    for message in build_messages():
        assert message["status"] in {"STANDARD", "REVIEW_REQUIRED"}
        assert message["status"] != "WARNING"
        lower = message["full_text"].lower()
        assert all(term not in lower for term in PROHIBITED_TERMS)
