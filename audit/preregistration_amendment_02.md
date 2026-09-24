# Pre-registration amendment 02 — conditional E08 final-model pathway

**Status:** REGISTERED BEFORE STAGE 4 RESULTS

**Registered:** 2026-09-24

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Prior amendment:** `audit/preregistration_amendment_01.md`, SHA256 `62abb1d13d3ffd16d3fdf41b461ad48aafc746caf4639f18e4545467ebf8c0fb`

## Registration timing and scope

Stage 2 is closed: no intervention met parent G1, frozen v1 remains the accepted model, and E08 is the locked experimental v2 for calibration and reliability studies. C01 had already been inspected and passed its local calibration gate before this amendment. C02–C09 and all E08 Stage 4 reliability results were unobserved at registration. This amendment adds a prospective conditional final-selection hypothesis; it does not revise G1 or reinterpret any completed result.

## H7 — conditional E08 final-model selection

E08 will replace v1 as the model nominated for the one-time sealed evaluation only if both conditions pass:

1. **Calibration condition:** at least one operational E08 calibration method among C01–C07 passes its own unchanged conjunctive gate. Reference-mask-dependent oracle diagnostics cannot satisfy this condition unless the chosen inference-time routing uses prediction-derived inputs only. Selection among passing operational calibrators follows amendment 01: lowest aggregate whole-image ECE, then lowest whole-image NLL, then lowest whole-image Brier, then experiment ID. Reported selection values must be five-fold similarity-aware out-of-fold results. The final all-validation fit is an operational proposal and cannot supply selection metrics.
2. **Reliability condition:** the final operational E08 case-reliability model selected under the fixed Stage 4 rules achieves five-fold similarity-aware out-of-fold failure-detection AUROC `>=0.941`. This is the frozen v1 comparator stated by the user. A candidate must also pass its own registered Stage 4 gate. R01 is descriptive and cannot be selected. Among passing R02–R06 candidates, select highest aggregate AUROC, then lowest ECE, then lowest Brier, then experiment ID. If none passes, the locked E08 base logistic model is used for the comparison but cannot satisfy H7 unless it reaches `0.941`. R07–R10 assess uncertainty and stability and do not change the selected model.

Both conditions are conjunctive. If either fails, v1 remains the nominated final model. Passing H7 nominates E08 for the still-sealed one-time evaluation; it does not establish clinical validity, external generalization, or a sealed-test pass.

## Probability, mask, and threshold policy

The hard threshold remains exactly `0.50`; no threshold may be tuned. If the selected calibrator changes probabilities and therefore changes hard masks at `0.50`, Stage 4 must use those locked calibrated probabilities and resulting masks. Hard-mask changes must be reported. Calibration methods requiring reference masks at inference are ineligible for final operation. No retraining is permitted during C02–C09.

## Integrity rules

- The sealed test remains unopened until the model, calibrator, reliability model, and policy are locked.
- Existing gates are not relaxed.
- Verification mismatches stop an experiment without a decision artifact.
- C02–C09 run in order. Failures remain recorded findings.
- Do not skip, merge, or substitute any of C03–C09. Each is a separate registered experiment with its own hypothesis, config, result, decision, and review log entry.
