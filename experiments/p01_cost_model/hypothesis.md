# P01 — Reliability-policy cost model

## Purpose

P01 fixes the relative decision costs used by the Stage 5 reliability policy before any reliability threshold is selected.

## Registered outcomes

The development failure target is `Dice < 0.50`. Reliability is the selected R02 model's out-of-fold estimate of `P(Dice >= 0.50)` for the C03-calibrated E08 segmenter.

For policy evaluation, a positive decision means **manual review required** and a negative decision means **accepted without mandatory review**:

- True positive: bad segmentation reviewed — cost `1`.
- False positive: acceptable segmentation reviewed — cost `1`.
- True negative: acceptable segmentation accepted — cost `0`.
- False negative: bad segmentation accepted — cost `10`.

These are relative engineering costs rather than monetary, clinical-utility, or patient-harm estimates. P01 is descriptive governance work and has no performance gate.

## Decision rule

P01 is complete only if the registered weights are documented without alteration and every fail-closed source, population, score, threshold, leakage, test-access, and retraining check passes. Threshold optimization is outside P01 and begins only after separate approval of P02.
