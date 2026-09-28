# Reliability-policy cost model

## Scope

This document fixes the relative engineering costs used to evaluate reliability-policy decisions for the C03-calibrated E08 segmenter with the selected R02 ten-feature logistic reliability model. It applies only to the 680-case BRISC internal development-validation population. It is not a clinical utility model, a monetary valuation, or evidence of external generalization.

The segmentation failure target is `Dice < 0.50`. The reliability model estimates `P(Dice >= 0.50)`. A positive policy decision means **manual review required**; a negative policy decision means **accepted without mandatory review**.

## Cost matrix

| Actual segmentation | Policy decision | Outcome | Relative cost |
|---|---|---:|---:|
| Bad (`Dice < 0.50`) | Review required | True positive | 1 |
| Acceptable (`Dice >= 0.50`) | Review required | False positive | 1 |
| Acceptable (`Dice >= 0.50`) | Accept | True negative | 0 |
| Bad (`Dice < 0.50`) | Accept | False negative | 10 |

For `N` cases, the mean policy cost is:

```text
mean_cost = (1 * TP + 1 * FP + 0 * TN + 10 * FN) / N
```

Lower mean cost is better.

## Meaning of each cost

### Manual review: 1 unit

One relative unit represents the effort and delay required to inspect an image and predicted mask, decide whether the mask is usable, and correct or reject it when needed. A reviewed case costs one unit whether the prediction proves acceptable or bad because the review action itself consumes the resource.

### False alarm: 1 unit

An acceptable segmentation flagged for review consumes the same review effort. There is no additional false-alarm penalty beyond this unit, which prevents double counting review cost.

### Missed failure: 10 units

A bad segmentation accepted without mandatory review costs ten units. The tenfold penalty encodes the preregistered engineering preference to spend review resources rather than silently accept a poor mask. It does not estimate medical injury, treatment cost, legal exposure, or patient harm.

### Correct acceptance: 0 units

An acceptable segmentation accepted without mandatory review incurs no additional reliability-policy cost.

## Assumptions

1. Every case has equal weight.
2. Every mandatory manual review has the same relative cost.
3. A flagged case receives competent review before downstream use.
4. Review prevents the silent-acceptance cost assigned to a missed failure.
5. Costs add linearly across cases.
6. Every failure is assigned the same cost regardless of failure severity.
7. Glioma, meningioma, and pituitary cases initially use the same cost matrix.
8. Reviewer capacity, turnaround time, disagreement, and residual reviewer error are not modeled.
9. The internal BRISC validation prevalence is used only to calculate development expected cost. It is not assumed to represent clinical prevalence.
10. A bad segmentation is defined by the registered development cutoff `Dice < 0.50`.
11. The neural segmentation hard-mask threshold remains fixed at `0.50`.
12. Reliability-policy thresholds are separate operating-policy parameters. Selecting them does not change model predictions or the segmentation threshold.
13. Costs are relative engineering preferences and must be reconsidered with clinical, workflow, and institutional input before any real deployment.

## Governance

The weights were declared in `audit/preregistration_amendment_01.md` before policy thresholds were selected. P01 documents those weights without modification and does not optimize a threshold. Stage 5.2 will sweep the existing R02 out-of-fold reliability scores under this fixed cost matrix only after separate approval.

The P01 integrity check confirmed 680 unique validation cases, 598 acceptable outcomes, 82 failures, five original held-out folds, finite reliability scores within `[0,1]`, exact target agreement with the `0.50` Dice cutoff, zero sealed-test paths, no threshold change, and no model retraining or refitting.
