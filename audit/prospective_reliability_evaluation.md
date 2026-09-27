# Prospective reliability evaluation

**Decision:** PASS

Each case was scored only by a standardized logistic model fitted on the other four similarity-aware folds. The controlling statistic is the unweighted mean of the five held-out-fold AUROCs.

| Fold | Cases | Failures | AUROC | ECE | Brier skill |
|---:|---:|---:|---:|---:|---:|
| 1 | 136 | 18 | 0.9831 | 0.0313 | 0.6753 |
| 2 | 136 | 13 | 0.9256 | 0.0938 | 0.1647 |
| 3 | 137 | 18 | 0.9771 | 0.0550 | 0.5448 |
| 4 | 135 | 19 | 0.9574 | 0.0459 | 0.5596 |
| 5 | 136 | 14 | 0.9590 | 0.0344 | 0.4847 |

Mean fold AUROC was **0.9604** (SD 0.0225). The registered minimum was `0.941`, so R08 **passed**.

Aggregate OOF AUROC was 0.9555 (95% group-bootstrap CI 0.9387–0.9704).

## True-category AUROC

| Category | Cases | Failures | AUROC | 95% CI |
|---|---:|---:|---:|---:|
| glioma | 204 | 63 | 0.8768 | [0.8255, 0.9241] |
| meningioma | 227 | 7 | 0.9942 | [0.9815, 1.0000] |
| pituitary | 249 | 12 | 0.9571 | [0.9183, 0.9842] |

The top 136 risks captured 18/19 zero-Dice cases.

All features are prediction-derived. No test image was accessed, the threshold remained 0.50, and no classifier or neural segmenter was retrained.
