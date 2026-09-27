# Stage 3 calibration completion report

**Status: COMPLETE. Stage 4 has not started. The sealed test remains unopened.**

## Decision

C03 isotonic calibration is the selected operational E08 calibrator. It passed its unchanged six-check gate and ranked ahead of C01 by whole-image OOF ECE (`0.000076` versus `0.000331`). C04 and C07 passed locally but are excluded because they require reference-mask routing at inference.

The selected C03 mapping changes 22,891 OOF hard-mask pixels across 679 of 680 images at the unchanged `0.50` threshold. Amendment 02 therefore requires Stage 4 to use the locked calibrated probabilities and resulting masks. The final all-validation mapping is an operational proposal; it was not used for reported metrics.

## Experiment outcomes

| Experiment | Outcome | Finding |
|---|---|---|
| c01_vector_scaling | PASS_OPERATIONAL | Segmenter gate passed; classifier vector scaling rejected descriptively; raw classifier probabilities retained. |
| c02_dirichlet | FAIL | Probability metrics passed, but mandatory hard-mask identity failed on 43,228 pixels across 680 images. |
| c03_isotonic | PASS_OPERATIONAL_SELECTED | All six ECE/Brier checks passed; selected by the fixed operational ranking. |
| c04_split_temperature | PASS_ORACLE_ONLY | Local gate passed with zero mask changes, but reference-region routing is unavailable at inference. |
| c05_reliability_weighted | FAIL | Low-reliability-quintile ECE worsened despite NLL improvement. |
| c06_per_category | FAIL | Predicted-glioma lesion-centred ECE was 0.055963, failing one of nine checks. |
| c07_hierarchical | PASS_ORACLE_ONLY | Nested gate passed, but every outer fold selected R or RC reference-region routing. |
| c08_unified_ece | VERIFIED | C01-C07 ECE values independently verified with 15 equal-frequency bins. |
| c09_category_diagrams | VERIFIED | Six OOF category diagrams produced; glioma calibration weakness remains explicit. |

## Category limitation

| Category | Cases | Raw lesion ECE | C03 lesion ECE | Case-reliability ECE |
|---|---:|---:|---:|---:|
| glioma | 204 | 0.082850 | 0.085523 | 0.097247 |
| meningioma | 227 | 0.036631 | 0.013591 | 0.020790 |
| pituitary | 249 | 0.037867 | 0.009881 | 0.027838 |

C03 improves meningioma and pituitary lesion-centred calibration but worsens glioma (`0.082850` to `0.085523`). The aggregate pass therefore does not mean calibration is uniformly improved across tumor types.

## H7 and governance

H7's calibration condition is satisfied. Its reliability condition remains pending Stage 4, so E08 is not the final nominated model. All C01–C09 experiments are separately registered, closed, committed, and represented in the review log. No neural retraining occurred, the threshold stayed `0.50`, no gate was relaxed, and the sealed test was accessed zero times.

## Limits

Stage 3 does not establish clinical validity, external or patient-independent generalization, or correction of confident spatial errors. It does not validate the all-validation operational fit prospectively. These results remain internal BRISC development evidence.
