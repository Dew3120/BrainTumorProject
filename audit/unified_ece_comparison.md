# Unified 15-bin ECE comparison

All values use 15 equal-frequency bins. C01–C07 source hashes, bin counts, pixel coverage, and stored ECE values were independently verified. This table does not change any registered gate or decision.

## Segmenter pixel calibration

| Experiment | Decision | Region | Raw ECE | OOF calibrated ECE | Change |
|---|---:|---|---:|---:|---:|
| c01_vector_scaling | PASS | lesion_centered | 0.050543 | 0.038164 | -0.012379 |
| c01_vector_scaling | PASS | background | 0.001761 | 0.002138 | +0.000378 |
| c01_vector_scaling | PASS | whole_image | 0.002503 | 0.000331 | -0.002172 |
| c02_dirichlet | FAIL | lesion_centered | 0.050543 | 0.036148 | -0.014394 |
| c02_dirichlet | FAIL | background | 0.001761 | 0.002129 | +0.000368 |
| c02_dirichlet | FAIL | whole_image | 0.002503 | 0.000210 | -0.002293 |
| c03_isotonic | PASS | lesion_centered | 0.050543 | 0.033509 | -0.017033 |
| c03_isotonic | PASS | background | 0.001761 | 0.002005 | +0.000245 |
| c03_isotonic | PASS | whole_image | 0.002503 | 0.000076 | -0.002428 |
| c04_split_temperature | PASS | lesion_centered | 0.050543 | 0.024521 | -0.026021 |
| c04_split_temperature | PASS | background | 0.001761 | 0.001512 | -0.000248 |
| c04_split_temperature | PASS | whole_image | 0.002503 | 0.002388 | -0.000115 |
| c05_reliability_weighted | FAIL | lesion_centered | 0.050543 | 0.037836 | -0.012707 |
| c05_reliability_weighted | FAIL | background | 0.001761 | 0.003125 | +0.001365 |
| c05_reliability_weighted | FAIL | whole_image | 0.002503 | 0.003427 | +0.000924 |
| c06_per_category | FAIL | lesion_centered | 0.050543 | 0.037725 | -0.012818 |
| c06_per_category | FAIL | background | 0.001761 | 0.003145 | +0.001384 |
| c06_per_category | FAIL | whole_image | 0.002503 | 0.003463 | +0.000960 |
| c07_hierarchical | PASS | lesion_centered | 0.050543 | 0.024577 | -0.025966 |
| c07_hierarchical | PASS | background | 0.001761 | 0.001516 | -0.000245 |
| c07_hierarchical | PASS | whole_image | 0.002503 | 0.002442 | -0.000061 |

## Frozen classifier outputs

| Output | State | ECE | Cases |
|---|---|---:|---:|
| four_class | raw | 0.008471 | 893 |
| four_class | oof | 0.008835 | 893 |
| binary | raw | 0.002404 | 893 |
| binary | oof | 0.003185 | 893 |

## Case-level reliability

The accepted v1 reliability model has 15-bin ECE `0.031665` and MCE `0.162491` across 680 out-of-fold cases.

## Interpretation

Low whole-image ECE can coexist with weaker lesion-centred calibration because background pixels dominate the image. Reference-region methods C04 and C07 remain oracle diagnostics and cannot be deployed. C08 is a reporting standardization only.
