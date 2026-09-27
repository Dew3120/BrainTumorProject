# Reliability fold sensitivity

**Status:** DESCRIPTIVE_COMPLETE

R09 regenerated three complete five-fold partitions while preserving similarity-group isolation. It is descriptive and cannot change the selected model.

| Seed | AUROC | Difference from original | ECE | Brier skill |
|---:|---:|---:|---:|---:|
| 17 | 0.9325 | -0.0230 | 0.0188 | 0.4913 |
| 42 | 0.9553 | -0.0002 | 0.0172 | 0.5176 |
| 93 | 0.9560 | +0.0005 | 0.0192 | 0.5156 |

Alternative-split AUROC mean was 0.9479 with sample SD 0.0134. The maximum absolute difference from original R02 was 0.0230.

The preregistered descriptive label is **MODERATE**. This label is not a performance gate and does not alter R02 selection.

## True-category AUROCs

| Seed | Glioma | Meningioma | Pituitary |
|---:|---:|---:|---:|
| 17 | 0.8620 | 0.8565 | 0.9620 |
| 42 | 0.8755 | 0.9961 | 0.9610 |
| 93 | 0.8777 | 0.9968 | 0.9624 |

No test image was accessed, the segmentation threshold remained 0.50, and no classifier or neural segmenter was retrained.
