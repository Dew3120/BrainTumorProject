# Controlled glioma intervention model selection

**Scope:** BRISC internal validation only. Patient independence, external generalization, and clinical validity remain unverified.

| ID | Intervention | Glioma Dice | Meningioma Dice | Pituitary Dice | Glioma Δ vs v1 | Harm-safe | Local gate | Pixel foreground ECE | Reliability AUROC |
|---|---|---:|---:|---:|---:|:---:|:---:|---:|---:|
| e01_lcc_filter | Largest component | 0.6050 | 0.8714 | 0.8108 | -0.0040 | yes | fail | — | — |
| e02_morphology | Morphology 3x3 | 0.6120 | 0.8738 | 0.8090 | +0.0031 | yes | fail | — | — |
| e03_tta | Eight-view TTA | 0.4487 | 0.8778 | 0.5508 | -0.1603 | no | fail | — | — |
| e04_multiscale | 128/256 multiscale | 0.5687 | 0.8785 | 0.7669 | -0.0403 | no | fail | — | — |
| e05_glioma_weighted | Glioma-weighted retraining | 0.5972 | 0.8391 | 0.7801 | -0.0117 | no | fail | 0.0580 | 0.9423 |
| e06_boundary_loss | Boundary-loss retraining | 0.5794 | 0.8600 | 0.7978 | -0.0296 | no | fail | 0.0584 | 0.9153 |
| e07_glioma_specialized | Classifier-routed specialist | 0.5802 | 0.8669 | 0.8088 | -0.0287 | yes | fail | 0.0604 | 0.9410 |
| e08_ensemble | Fixed E05-E07 ensemble | 0.6324 | 0.8822 | 0.8183 | +0.0234 | yes | pass | — | — |

## Decision

E08 is the best category-harm-safe experimental candidate. It has the highest glioma Dice (`0.6324`) and improves meningioma and pituitary relative to v1. No calibration or reliability tie-break was needed because the glioma-Dice ranking was not tied.

E08 does **not** replace frozen v1. Its glioma improvement over v1 is only `+0.0234`, below parent G1's required `+0.0500`. Frozen v1 remains the accepted final model. E08 is the locked experimental v2 target for the requested calibration and reliability comparisons; those analyses cannot retroactively change this model-selection decision.

The ensemble's pass is compatible with all three component models failing locally: averaging can reduce uncorrelated errors. It does not show that any component is individually superior, and every result was obtained on the same internal validation partition used during development.

No sealed-test image or prediction was accessed.
