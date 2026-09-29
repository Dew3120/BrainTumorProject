# Pre-registration amendment 19 — X03 domain-bootstrap membership correction and attempt four

**Status:** REGISTERED AFTER THE THIRD PRE-PERFORMANCE X03 STOP AND BEFORE ANY X03 PERFORMANCE RESULT

**Registered:** 2026-09-29

**Parent registration:** `audit/preregistration_amendment_13.md`, SHA256 `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5`

**Target handling:** `audit/preregistration_amendment_17.md`, SHA256 `3c7941e388f8c095884d7bd62bc1886fb3aa8a1b5003520f76cdcbffd6ee12b7`

**Loader correction:** `audit/preregistration_amendment_18.md`, SHA256 `cce55327a687c6ef4adec4ba85284e887438bfc6352abeb615f0ab3f361d8c63`

This amendment corrects a downstream analysis membership defect discovered after the third X03 attempt completed frozen inference on all 22,785 slices. The error occurred before result writing or metric inspection. It does not modify model outputs, preprocessing, targets, metrics, gates, or thresholds.

## Triggering verification stop

Amendment-17 mask-derived domain features contain 9,382 representable cases. The patient bootstrap incorrectly used indices from the full 9,464-row primary membership, producing an out-of-bounds index during domain-gap analysis. The script stopped before writing probabilities, metrics, panels, decisions, or a performance result.

## Procedural correction

1. Mask-derived features, including foreground area, centroid, component count, and reference-boundary gradient, use `representable_rows` with exactly 9,382 cases.
2. Input intensity, encoder representation, reliability-score distributions, and policy-rate analyses use `primary_rows` with exactly 9,464 cases.
3. Both memberships and their required counts are declared explicitly and asserted at runtime.
4. Every bootstrapped feature array must assert `len(feature_array) == len(bootstrap_membership)` before any indexing.
5. A regression test must construct different full and representable memberships and prove both bootstrap paths complete without an index-boundary error.
6. The correction is procedural bookkeeping. It does not alter any computed feature or resampling rule.

## Audit and attempt-four authorization

X03 must rerun the complete audit against the same 294 registered derivative files, published SHA256 values, labels `{0,1,2,3}`, 147 subjects, identifiers, overlap checks, and decision criteria. The renewed audit decision must be committed before inference.

Exactly one fourth X03 attempt is authorized:

- **Attempt 1:** stopped before inference after detecting 82 target-representability losses; Amendment 17 registered conservative handling.
- **Attempt 2:** stopped after 144 inferred slices when native `uint8` was incorrectly converted to `float32`; Amendment 18 corrected the loader contract.
- **Attempt 3:** inferred all 22,785 slices, then stopped during domain bootstrap because a 9,382-element feature array was indexed using the 9,464-row membership.
- **Attempt 4:** uses the corrected explicit memberships and must record completion or the next fail-closed condition.

If attempt 4 encounters another verification failure, X03 stops again and reports; no automatic restart is authorized.

## Constraints retained

- BRISC sealed-test pixels remain closed.
- Frozen preprocessing and native-datatype routing remain unchanged.
- The 240×240 to 128×128 adapter and interpolation methods remain unchanged.
- Mask preprocessing remains nearest-neighbor; no dilation, closing, or rescue transform is allowed.
- Neural retraining, calibrator refitting, and reliability refitting remain prohibited.
- Segmentation threshold remains exactly `0.50`.
- Reliability-policy threshold remains exactly `0.9490250173235835`.
- Gates, labels, cohort membership, bootstrap count `2,000`, bootstrap seed `42`, and Amendment-17 metric views remain unchanged.

## Decision authority

The user explicitly approved Amendment 19, the membership assertions and regression test, a complete renewed audit, and exactly one fourth X03 attempt.
