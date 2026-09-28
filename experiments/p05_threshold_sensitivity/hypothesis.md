# P05 — Reliability-threshold sensitivity

## Purpose

P05 tests how sensitive the registered P02 policy cost is to perturbing the selected reliability threshold `0.9490250173235835`. It is a read-only analysis of the same 680 out-of-fold R02 reliability scores.

## Falsifiable claims

1. **Locally flat:** the worst total-cost increase at the applied `-0.02` and `+0.02` perturbations is no more than 10% of the selected-threshold cost.
2. **Robust:** the locally-flat claim passes and the worst total cost at the applied `-0.05` and `+0.05` perturbations is no more than 25% above the selected-threshold cost.
3. Otherwise the selected threshold is classified as **fragile**.

The robustness labels are descriptive. They cannot change the selected threshold or relax an earlier gate.

## Evaluation contract

- Apply nominal shifts `-0.10`, `-0.05`, `-0.02`, `0`, `+0.02`, `+0.05`, and `+0.10`, clipping applied thresholds to `[0,1]` and recording both nominal and applied shifts.
- Use the unchanged rule `reliability < threshold -> REVIEW_REQUIRED`; equality is `STANDARD`.
- Use the unchanged binary costs `TP=1`, `FP=1`, `TN=0`, `FN=10`.
- Report flagged and accepted cases, failures caught and missed, acceptable cases flagged and accepted, total cost, mean cost, and cost change from the selected threshold.
- Compute cost change per `0.01` as `(perturbed cost - selected cost) / (abs(applied shift) / 0.01)` for every nonzero applied shift.
- Define the worst case at `±0.05` and `±0.10` as the maximum of their respective negative and positive endpoint costs. The positive endpoints may be clipped to `1.0`.
- Produce an exact local step curve using every distinct OOF score within the clipped `±0.10` interval, together with the seven requested perturbation points.
- Compare the selected threshold against `0.50`, `0.00`, and `1.00`.

## Completion rule

P05 completes only if all upstream hashes and the 680-case population match, every generated CSV row is independently recomputed, the selected row exactly reproduces P02, and no test-set access, segmentation-threshold change, model fitting, retraining, or gate relaxation occurs.
