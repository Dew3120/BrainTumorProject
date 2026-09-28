# Reliability-threshold sensitivity

## Decision

**FRAGILE**; locally flat: **NO**.

Small perturbations produce large cost changes. The threshold requires careful calibration.

The selected threshold remains `0.9490250173235835`. This descriptive analysis does not alter the policy.

## Requested perturbations

| Shift | Applied threshold | Flagged | Failures caught | Failures missed | Acceptable flagged | Total cost | Mean cost | Cost delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.10 | 0.849025017323583 | 134 | 70 | 12 | 64 | 254 | 0.373529 | +49 |
| -0.05 | 0.899025017323583 | 156 | 71 | 11 | 85 | 266 | 0.391176 | +61 |
| -0.02 | 0.929025017323583 | 175 | 75 | 7 | 100 | 245 | 0.360294 | +40 |
| selected | 0.949025017323583 | 195 | 81 | 1 | 114 | 205 | 0.301471 | +0 |
| +0.02 | 0.969025017323583 | 245 | 82 | 0 | 163 | 245 | 0.360294 | +40 |
| +0.05 | 0.999025017323584 | 628 | 82 | 0 | 546 | 628 | 0.923529 | +423 |
| +0.10 | 1.000000000000000 | 680 | 82 | 0 | 598 | 680 | 1.000000 | +475 |

At the selected threshold, 195 cases are flagged, 81 failures are caught, 1 failure is missed, and total cost is 205.

The worst endpoint cost at `±0.05` is **628**. The worst endpoint cost at `±0.10` is **680**. The positive `+0.10` request is clipped to threshold `1.0` because the registered threshold domain is `[0,1]`.

Cost change per `0.01` is reported in `result.json` for each nonzero requested perturbation using the registered directional formula.

## Historical comparison

| Operating point | Threshold | Flagged | Failures caught | Failures missed | Total cost | Cost delta vs selected |
|---|---:|---:|---:|---:|---:|---:|
| flag_nothing | 0.000000000000000 | 0 | 0 | 82 | 820 | +615 |
| conventional_0.50 | 0.500000000000000 | 66 | 48 | 34 | 406 | +201 |
| selected | 0.949025017323583 | 195 | 81 | 1 | 205 | +0 |
| flag_everything | 1.000000000000000 | 680 | 82 | 0 | 680 | +475 |

## Integrity

- All 551 CSV rows were independently recomputed.
- The selected row reproduced P02 exactly.
- Population: 680 OOF cases, 598 acceptable and 82 failures.
- Cost vector: TP=1, FP=1, TN=0, FN=10.
- Segmentation threshold remained `0.50`.
- Sealed test-set access: 0.
- Model, calibrator, and reliability-model retraining/refitting: 0.

This is an internal development-policy sensitivity analysis. It does not establish clinical validity or cross-dataset generalization.
