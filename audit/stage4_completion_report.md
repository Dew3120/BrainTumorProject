# Stage 4 reliability completion report

**Stage 4 complete, ready for Stage 5 review.**

| Study | Status | Principal outcome |
|---|---|---|
| R01 | DESCRIPTIVE_COMPLETE | 14-feature ablation baseline; boundary entropy dominated. |
| R02 | PASS — SELECTED | 10-feature logistic AUROC 0.955502; passed noninferiority. |
| R03 | FAIL | Extended features AUROC 0.951423; superiority gate missed. |
| R04 | PASS — NOT SELECTED | All category gates passed, but routed aggregate AUROC 0.930860. |
| R05 | FAIL | Predicted-glioma AUROC 0.871096 missed 0.88. |
| R06 | FAIL | All nonlinear models underperformed R02. |
| R07 | DESCRIPTIVE_COMPLETE | Four coefficient intervals excluded zero; boundary entropy most stable. |
| R08 | PASS | Mean held-out-fold AUROC 0.960424 passed H7 gate. |
| R09 | DESCRIPTIVE_COMPLETE | MODERATE fold sensitivity; AUROC SD 0.013362. |
| R10 | DESCRIPTIVE_COMPLETE | EXACT seed stability across five registered seeds. |

## Selected operational reliability model

The selected model is **R02 — 10-feature aggregate logistic**, operating on C03-calibrated E08 probability maps. Its aggregate OOF AUROC is `0.955502`, ECE `0.017133`, Brier `0.052757`, and Brier skill `0.503348`.

H7: **PASS**. R08's unweighted mean fold AUROC is `0.960424` against the `0.941` minimum.

## E08 versus v1 reliability

- AUROC: E08 `0.955502` vs v1 `0.940990`; difference `+0.014512`.
- ECE: E08 `0.017133` vs v1 `0.021171`; difference `-0.004038`.
- Brier skill: E08 `0.503348` vs v1 `0.476052`; difference `+0.027296`.
- These internal-development intervals overlap; no statistical superiority or clinical-generalization claim is made.

## Integrity

- Evidence checkpoint before report consolidation: `1eb5c4f2ae60f8da45392ffc448a39a4bf7a2265`; it was clean and matched `origin/master`.
- Final report commit: recorded in Git history and the delivery message because a tracked file cannot contain its own content-derived commit SHA.
- Expected final working tree: clean after the report commit; verified in the delivery message.
- Tests: 136 passed.
- Ruff: passed.
- Sealed test-set accesses: 0.
- Segmentation threshold changes: 0; threshold remained 0.50.
- Classifier or neural-segmenter retraining events during Stage 4: 0.
- Registered reliability-estimator fitting occurred as required and is not neural-model retraining.

Stage 4 complete, ready for Stage 5 review.
