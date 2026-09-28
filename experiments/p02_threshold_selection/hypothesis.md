# P02 — Reliability threshold selection

## Purpose

P02 selects reliability-policy operating thresholds for the C03-calibrated E08 segmenter with R02's frozen ten-feature out-of-fold reliability scores. It uses the P01 costs fixed before threshold outcomes were inspected.

## Binary policy

For threshold `t`, review is required when reliability is strictly below `t`; otherwise the case is accepted. Sweep `0`, every distinct observed OOF reliability score, and `1`. Minimize `(TP + FP + 10*FN) / 680`. Ties use fewer accepted failures, lower review rate, then higher threshold.

## Three-state policy

For `theta_low <= theta_high`, review is required below `theta_low`, warning applies from `theta_low` up to but excluding `theta_high`, and standard applies at or above `theta_high`. Costs are review-required `1` for either outcome; warning acceptable `0.25`; warning bad `5`; standard acceptable `0`; standard bad `10`.

Minimize mean cost. Ties use fewer bad masks in Standard, fewer bad masks outside Review Required, lower mandatory-review rate, higher `theta_high`, then higher `theta_low`.

## Interpretation

The global thresholds are internal-development operating candidates. True-category optima are descriptive and cannot route deployment decisions. P02 changes no segmentation mask or model and makes no clinical-validity claim.
