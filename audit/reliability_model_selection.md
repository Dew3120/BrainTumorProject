# Reliability model selection

**Selected model:** R02 — 10-feature aggregate logistic

The registered rule considers only R02–R06 models that passed their own gates, then orders them by aggregate OOF AUROC, lower ECE, lower Brier, and experiment ID. R01 and R07–R10 cannot be selected.

| Experiment | Outcome | Eligible | Cases | Aggregate OOF AUROC | ECE | Brier |
|---|---|---|---:|---:|---:|---:|
| R01 | DESCRIPTIVE_COMPLETE | no | 680 | 0.954503 | 0.017072 | 0.053708 |
| R02 | PASS | yes | 680 | 0.955502 | 0.017133 | 0.052757 |
| R03 | FAIL | no | 680 | 0.951423 | 0.017258 | 0.052866 |
| R04 | PASS | yes | 679 | 0.930860 | 0.017659 | 0.057901 |
| R05 | FAIL | no | 679 | 0.954808 | 0.018598 | 0.053588 |
| R06 | FAIL | no | 680 | 0.940268 | 0.025110 | 0.059124 |

R02 outranks the other passing candidate, R04, on aggregate AUROC (`0.955502` versus `0.930860`). On R04's matched 679-case cohort, R02 is still higher at `0.955015`. Tie-breakers are therefore unnecessary.

H7 reliability condition: **PASS**. Selected aggregate AUROC `0.955502` and R08 mean fold AUROC `0.960424` both exceed the registered `0.941` minimum.

Compared with v1, selected E08 reliability AUROC changes by `+0.014512`, ECE by `-0.004038` (lower is better), and Brier skill by `+0.027296`.

This selection nominates the C03-calibrated E08 pathway with R02 reliability for later policy and still-sealed evaluation. It does not establish clinical validity, patient-independent generalization, or external generalization.
