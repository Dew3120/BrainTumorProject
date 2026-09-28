# Reliability estimator seed stability

**Status:** DESCRIPTIVE_COMPLETE

R10 varied only the logistic estimator seed while preserving every case, fold, feature, target, scaler, and other estimator parameter.

| Seed | AUROC | Difference from R02 | ECE | Brier skill | Maximum OOF probability difference |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.955502 | +0.000000 | 0.017133 | 0.503348 | 0 |
| 21 | 0.955502 | +0.000000 | 0.017133 | 0.503348 | 0 |
| 42 | 0.955502 | +0.000000 | 0.017133 | 0.503348 | 0 |
| 84 | 0.955502 | +0.000000 | 0.017133 | 0.503348 | 0 |
| 168 | 0.955502 | +0.000000 | 0.017133 | 0.503348 | 0 |

Mean AUROC was 0.955502 with sample SD 0.000000.

The preregistered descriptive stability label is **EXACT**. R10 has no gate and cannot change model selection.

All ten features are prediction-derived. No test image was accessed, threshold 0.50 was unchanged, and no classifier or neural segmenter was retrained.
