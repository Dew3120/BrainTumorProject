# Largest-component filtering result

| Category | Baseline Dice | Filtered Dice | Delta | 95% group-bootstrap CI | Gate |
|---|---:|---:|---:|---:|---:|
| glioma | 0.6090 | 0.6050 | -0.0040 | (-0.0149, +0.0074) | fail |
| meningioma | 0.8702 | 0.8714 | +0.0011 | (-0.0059, +0.0078) | pass |
| pituitary | 0.8088 | 0.8108 | +0.0020 | (-0.0015, +0.0057) | pass |

**Conjunctive decision: FAIL.**

Filtering removes disconnected components from a hard mask. It cannot recover a missed lesion and can remove real multifocal disease. Results are internal development evidence only.