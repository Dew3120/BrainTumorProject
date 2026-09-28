# X02 decision

**Audit decision:** `ELIGIBLE_INDEPENDENT`

The frozen stack may proceed to one evaluation. All 50 image/mask pairs passed integrity, geometry, label, and membership checks. No inference or performance computation occurred during this audit. The release licence conflict is handled under the more restrictive CC BY-NC 4.0 statement.

## Aborted first evaluation attempt

The first evaluation launch stopped fail-closed at `sub-008:coronal:003` because the native source mask contained 10 tumor pixels but nearest-neighbor resize to the frozen 128×128 model space produced zero reference pixels. Five complete batches (80 of 624 slices) had passed through the frozen models before the data-loader assertion stopped the next batch. No prediction arrays, performance metrics, figures, decisions, or result artifacts were saved or inspected.

Read-only verification found five such source-positive slices among the fixed 425-slice primary cohort. Amendment 15 registers conservative handling before the restart: all five receive Dice 0 in the strict primary gate, are reported explicitly, and remain failures for reliability evaluation. The frozen adapter is unchanged. One clean restart is authorized; another verification failure stops X02 again.

## Final external-evaluation decision

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`

The independent OpenNeuro pituitary cohort failed classifier-calibration, strict pituitary Dice, case-reliability ECE, and top-20% failure-capture gates. Classification recall and reliability ranking AUROC passed, but they do not override the conjunctive failures. The five erased targets remain explicit zero-score cases and do not explain the broad failure. No gate is relaxed and X02 will not be used for tuning.
