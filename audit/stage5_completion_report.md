# Stage 5 reliability-policy completion report

## Completion status

| Item | Status | Final outcome | Primary artifact |
|---|---|---|---|
| 5.1 / P01 — Cost model | COMPLETE | Relative engineering costs locked at TP=1, FP=1, TN=0, FN=10. | `docs/cost_model.md` |
| 5.2 / P02 — Threshold selection | PASS | Global threshold `0.9490250173235835` minimized registered expected cost at 205. | `audit/threshold_selection.md` |
| 5.3 / P03 — Decision rule | PASS | Two-state rule adopted; no Warning state was supported by the exhaustive cost search. | `docs/decision_rule.md` |
| 5.4 / P04 — Warning copy | PASS | Fixed eight-message catalogue adopted with general and predicted-glioma variants. | `docs/warning_copy.md` |
| 5.5 / P05 — Sensitivity | PASS — FRAGILE | Sensitivity analysis completed honestly; the threshold failed the preregistered locally-flat and robustness descriptions. | `audit/threshold_sensitivity.md` |

All five registered Stage 5 items have documented outcomes, experiment packages, decisions, review-log entries, and committed evidence.

## Final reliability policy

For finite reliability score `r` in `[0,1]`:

```text
if r < 0.9490250173235835:
    status = REVIEW_REQUIRED
else:
    status = STANDARD
```

Reliability at exactly the threshold is `STANDARD`. Missing, nonnumeric, non-finite, negative, or greater-than-one reliability fails closed to `REVIEW_REQUIRED`. There is no Warning state because the registered three-state search selected equal lower and upper thresholds.

On the 680 internal OOF development cases, the policy flags 195 cases and accepts 485. It catches 81 of 82 Dice-below-0.50 failures, misses one, flags 114 acceptable cases, and has total registered cost 205.

The segmentation hard-mask threshold remains `0.50`; the reliability-policy threshold does not change model masks.

## Threshold sensitivity

P05 classified the selected reliability threshold as **FRAGILE** and **not locally flat** under the rules registered before execution.

- Selected cost: `205`.
- Both `±0.02` endpoint costs: `245`, an increase of 40 or 19.51%.
- `-0.05` cost: `266`; `+0.05` cost: `628`; worst `±0.05` cost: `628`.
- `-0.10` cost: `254`; requested `+0.10` clips to threshold `1.0` and costs `680`; worst `±0.10` cost: `680`.
- Conventional `0.50` cost: `406`, or 201 above selected.
- Flag nothing at `0.00`: cost `820`, or 615 above selected.
- Flag everything at `1.00`: cost `680`, or 475 above selected.

Small perturbations produce large cost changes. The threshold requires careful calibration.

This finding does not alter the selected threshold. It requires exact implementation, preservation of the calibrated reliability pipeline, and future validation on a sealed population before any broader claim.

## User-facing message catalogue

The final catalogue is `experiments/p04_warning_copy/artifacts/message_catalogue.json`, SHA256 `8e9a8c185221fac7f1bad5a4dedfdf8661e8d59b889ae83b0dd7b731f1004c2f`.

It contains eight complete messages:

- `STANDARD / AT_OR_ABOVE_THRESHOLD`, general and predicted-glioma variants.
- `REVIEW_REQUIRED / BELOW_THRESHOLD`, general and predicted-glioma variants.
- `REVIEW_REQUIRED / RELIABILITY_UNAVAILABLE`, general and predicted-glioma variants.
- `REVIEW_REQUIRED / RELIABILITY_INVALID`, general and predicted-glioma variants.

Every message contains the exact threshold and the disclaimer `Research prototype — not for clinical use`. No Warning message, clinical claim, diagnostic claim, or accuracy promise is present.

## H7 recap

H7 is **PASS** on both registered conditions:

1. **Calibration condition:** C03 isotonic calibration passed its registered gate and remains the selected operational calibration for E08.
2. **Reliability condition:** R02's 10-feature aggregate logistic model achieved aggregate OOF AUROC `0.955502`, above the registered v1 comparator `0.941`. R08's mean prospective held-out-fold AUROC was `0.960424` and also passed.

The selected combination remains **C03-calibrated E08 + R02 10-feature logistic reliability model**. These are internal development results. They do not establish clinical validity, patient-independent generalization, or cross-dataset performance.

## Verification and governance

- Stage 5 evidence commit before this report: `d2ef1eb142100d38ad6ab6c438442d02743d90c5`.
- The commit containing this report is recorded in Git history and in the delivery message because a tracked file cannot contain its own commit hash.
- Complete test suite: **153 passed**.
- Ruff: **passed**.
- Experiment registry validation: **passed** with 37 registered experiment directories.
- P05 curve verification: **551 of 551 rows independently recomputed**.
- P05 selected row reproduced P02 exactly.
- Sealed test-set accesses during Stage 5: **0**.
- Segmentation hard-mask threshold changes during Stage 5: **0**; it remained `0.50`.
- Neural-model retraining during Stage 5: **0**.
- Calibrator or reliability-model refitting during Stage 5: **0**.
- Cost vector changes after P01: **0**.
- Reliability-policy threshold changes after P02: **0**.
- Gate relaxations: **0**.
- Expected final working tree after the report commit: **clean**; confirmed in the delivery message.

**Stage 5 complete, Phase 5 complete, ready for Phase 6**
