# Calibrated-E08 reliability feature ablation

**Status:** REGISTERED — NOT STARTED

**Experiment:** Reliability study R01

The locked fourteen prediction-only features will not contribute equally to calibrated-E08 segmentation reliability. R01 will quantify each feature's association with out-of-fold failure discrimination using two complementary descriptive analyses: leave-one-feature-out refitting and held-out permutation importance.

The population is the same 680 segmentation-validation cases. Every probability and hard mask comes from the case's five-fold out-of-fold C03 isotonic mapping. The target is `Dice >=0.50` computed from the C03-calibrated hard mask at the unchanged `0.50` threshold. Reference masks create that target and evaluation metrics only. No reference-derived quantity enters a feature.

R01 is descriptive and has no pass/fail performance gate. It cannot be selected as the final Stage 4 model. Its ablation and coefficient-sign results become the preregistered inputs to R02's feature-selection rule. No neural model is retrained and no sealed-test image is accessed.
