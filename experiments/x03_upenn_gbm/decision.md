# X03 decision

**Audit decision:** `ELIGIBLE_INDEPENDENT`

All 147 expert-labelled CE-T1/mask pairs passed release-integrity, target, geometry, and overlap checks. One frozen-stack evaluation may proceed. Glioblastoma is mapped coarsely to glioma and is not directly comparable to heterogeneous BRISC glioma.

## Attempt history and Amendment 19 authorization

- **Attempt 1:** Stopped before inference after 82 of 9,464 source-positive slices became empty under the frozen nearest-neighbor target transform. Amendment 17 retains them as strict zero-overlap failures and reports 9,382 representable cases separately.
- **Attempt 2:** Stopped after 144 inferred slices because unconditional NIfTI `float32` conversion violated the native-`uint8` adapter contract and rejected 18 constant reference-empty slices. Amendment 18 corrected datatype preservation and added a constant-`uint8` regression test.
- **Attempt 3:** Completed inference on all 22,785 slices, then stopped before result writing because 9,382 representable mask-feature rows were indexed with the 9,464-row primary bootstrap membership. Amendment 19 binds each analysis to its correct membership and adds explicit length assertions and a regression test.
- **Attempt 4:** Authorized once after this renewed audit. Any new fail-closed condition stops X03 without another automatic restart.

No prior attempt saved probabilities, metrics, panels, decisions, or a performance result. The complete Amendment-19 audit passed on the same 294 release files, labels, patients, geometry, overlap criteria, and native-`uint8` adapter path.

## Final external-evaluation decision

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`

The independent UPENN-GBM cohort failed classifier recall and calibration, strict glioma Dice, case-reliability ECE, and top-20% failure-capture gates. Reliability ranking AUROC passed, but it does not override the conjunctive failures. The 82 erased targets remain explicit zero-score cases and do not explain the broad failure. No gate is relaxed and X03 will not be used for tuning.

**Attempt 4:** Completed inference on all 22,785 slices and wrote the final registered strict, representable-only, native-resolution, and exploratory whole-volume views. Independent recomputation matched all 45 checked values exactly.
