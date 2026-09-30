# Pre-registration amendment 22 — application sequencing and no-tumor fail-closed policy

**Status:** REGISTERED BEFORE PHASE 6 APPLICATION IMPLEMENTATION

**Registered:** 2026-09-30

**Authorizer:** User

**Evidence state before registration:** commit `2961d37f5ab9dc7766d0264898f687e529fdfa21`

**Frozen-stack manifest SHA256:** `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`

**Frozen-stack Git tree:** `439b4fca4b47f12b173c24eaa039b1258f9ddf03`

## Reason for amendment

The application design exposed two policy details that were not explicit in the historical inference contracts. First, OOD status must be known before any classification or segmentation result is disclosed or used by the user-facing policy, even though the frozen stack may need silent forward passes to compute OOD features. Second, a classifier prediction of `no_tumor` has no meaningful tumor-mask reliability interpretation. Treating an absent reliability score as a normal standard result would violate the registered fail-closed posture.

This amendment specifies application sequencing and a new fail-closed reason code. It does not alter a model, calibration mapping, score, threshold, experiment result, or the frozen stack.

## Item A — OOD sequencing interpretation

The binding interpretation is **OOD before classification disclosure and policy use**.

1. The application may execute preprocessing and frozen neural forward passes silently when those computations are required to produce O05 component features.
2. Classification probabilities, the predicted category, segmentation probabilities, hard masks, measurements, and reliability outputs remain withheld from the user interface and from downstream user-facing policy decisions until O05 completes and its output passes validation.
3. O05 is evaluated at the unchanged OOD review threshold `0.36854275476100085`. REFUSE remains disabled.
4. An O05 score that is missing, malformed, non-finite, outside its registered domain, or produced under an artifact-verification failure fails closed. No classification or segmentation result may be disclosed as `STANDARD` in that state.
5. After a valid O05 result exists, the application may disclose the frozen model outputs with the O06 and reliability statuses that apply. OOD `REVIEW_REQUIRED` does not authorize threshold changes, hidden re-inference, or a different model route.
6. Logging may record state transitions and integrity outcomes, but it must not expose a classification result before the OOD gate reaches a terminal valid state.

The required application ordering is:

`INPUT_VALIDATED -> SILENT_FROZEN_FORWARD_PASSES -> O05_VALIDATED -> O06_POLICY -> CLASSIFICATION_DISCLOSURE -> CONDITIONAL_SEGMENTATION_RELIABILITY_POLICY -> RESULT`

Any implementation that displays classification or segmentation before `O05_VALIDATED` is non-conforming.

## Item B — no-tumor fail-closed path

When the frozen classifier's model-predicted category is `no_tumor`, segmentation reliability is not applicable because R02 estimates the reliability of a tumor-region segmentation. The application must emit the following P03-compatible policy output:

```json
{
  "reliability_score": null,
  "reliability_threshold": 0.9490250173235835,
  "reliability_status": "REVIEW_REQUIRED",
  "requires_manual_review": true,
  "status_reason": "SEGMENTATION_RELIABILITY_NOT_APPLICABLE",
  "policy_version": "p03_decision_rule+amendment22"
}
```

The reason code `SEGMENTATION_RELIABILITY_NOT_APPLICABLE` is registered as an additive extension to the P03 application catalogue. Historical P03 and P04 artifacts remain unchanged evidence of the original two-state experiment. Phase 6 must implement and test the additive contract rather than rewriting those historical artifacts.

The application catalogue entry is:

```json
{
  "message_id": "review_segmentation_reliability_not_applicable_no_tumor",
  "status": "REVIEW_REQUIRED",
  "reason": "SEGMENTATION_RELIABILITY_NOT_APPLICABLE",
  "predicted_category_condition": "PREDICTED_NO_TUMOR",
  "threshold": 0.9490250173235835,
  "severity": "HIGH",
  "title": "Manual review required",
  "body": "The model-predicted category is no tumor. A segmentation reliability score applies only when a tumor region is evaluated, so no segmentation reliability score was produced. This case has been placed in Review Required.",
  "disclaimer": "Research prototype — not for clinical use"
}
```

Additional binding rules:

1. Message selection uses only the model-predicted category, never the reference or true category.
2. The reliability score is normalized to `null`; it must not be fabricated as `0`, `1`, or the policy threshold.
3. The case remains `REVIEW_REQUIRED` even if a silent segmenter forward pass produced a nonempty region.
4. The segmentation threshold remains exactly `0.50`; the reliability threshold remains exactly `0.9490250173235835`.
5. The user interface must explain that reliability is not applicable; it must not describe the no-tumor prediction as confirmed absence of tumor.

## Phase 6 documentation contract

Phase 6 comprises nine ordered stages: domain layer, worker process, Qt shell, core panels, mask editor, history, PDF export, error handling, and tests. Each stage must:

1. document every created file with path, purpose, rationale, content summary, test command, and known edge cases;
2. append exactly one stage entry to `docs/app_build_log.md`;
3. report the stage summary, file list, test count, known gaps, and commit hash;
4. use a descriptive commit message; and
5. pause for user review before the next stage.

The build log is append-only after creation. Before presenting a stage as complete, the agent must read the new build-log entry, run the stated tests, inspect the commit message, and correct any omission.

## Constraints retained

- BRISC sealed-test access remains prohibited.
- Neural retraining, calibration fitting, reliability fitting, and threshold selection remain prohibited.
- Segmentation threshold remains exactly `0.50`.
- Reliability threshold remains exactly `0.9490250173235835`.
- OOD review threshold remains exactly `0.36854275476100085`; REFUSE remains disabled.
- `runs/final_stack/` remains immutable and read-only to the application.
- No application state may relax a registered gate or silently substitute an artifact.
- Missing or invalid integrity, OOD, classification, segmentation, or reliability output fails closed.

## Completion rule

Amendment 22 is registered when this document, its SHA256 entry in `experiments/registry.json`, and its append-only review-log entry are committed. Registration authorizes Phase 6 planning and implementation under this contract; it does not itself implement Phase 6.
