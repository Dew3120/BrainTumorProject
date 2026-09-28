# Reliability decision rule

## Policy

The internal-development reliability policy for the C03-calibrated E08 segmenter and R02 reliability model is:

```text
RELIABILITY_THRESHOLD = 0.94902501732358346

if reliability_score < RELIABILITY_THRESHOLD:
    status = REVIEW_REQUIRED
else:
    status = STANDARD
```

Equality belongs to `STANDARD`. The threshold came from P02's exhaustive internal-validation cost sweep under P01 costs `TP=1`, `FP=1`, `TN=0`, `FN=10`.

## Why there is no Warning state

P02 evaluated 230,860 valid three-state threshold pairs and selected `theta_low = theta_high = 0.94902501732358346`. Therefore the interval `theta_low <= r < theta_high` is empty. A nonempty Warning band increased expected cost under the registered weights, so P03 does not invent one.

## Expected OOF behavior

| Measure | Count |
|---|---:|
| Review required | 195 |
| Standard | 485 |
| Failures caught | 81 |
| Failures missed | 1 |
| Acceptable masks reviewed | 114 |
| Total relative cost | 205 |

These are internal-development estimates on 680 OOF BRISC cases, not prospective or clinical performance.

## Glioma policy

The descriptive glioma optimum was `0.950106439123928`, while the global threshold is `0.94902501732358346`. They change zero assignments and have identical subgroup cost. Glioma does not receive a separate policy.

Glioma remains the weakest true-category reliability subgroup, with R02 AUROC `0.876843` (approximately `0.877`). The global rule applies uniformly; weaker subgroup ranking is retained as a limitation rather than used to create a post-hoc threshold.

## Phase 6 integration contract

Every inference result must provide:

- `reliability_score`: a finite number in `[0,1]`, or `null` after invalid/unavailable normalization
- `reliability_threshold`: exactly `0.94902501732358346`
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
