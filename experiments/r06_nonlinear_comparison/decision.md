# Nonlinear calibrated-E08 reliability decision

**Decision:** FAIL

The fixed gate was AUROC at least `0.9750`. R02 reproduced at `0.955502`.

- **xgboost:** AUROC `0.940268` (95% CI `0.918127–0.959850`), change versus R02 `-0.015234`, gate `FAIL`.
- **mlp:** AUROC `0.906181` (95% CI `0.869755–0.939232`), change versus R02 `-0.049321`, gate `FAIL`.
- **random_forest:** AUROC `0.935252` (95% CI `0.909057–0.958358`), change versus R02 `-0.020250`, gate `FAIL`.

The retained operational reliability model is **r02_logistic**. No nonlinear model passed; retain R02's coefficient-level interpretability.

All ten inputs derive only from the C03 OOF calibrated probability map and threshold-0.50 prediction. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
