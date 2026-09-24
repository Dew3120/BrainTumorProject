# Reliability-weighted pixel calibration

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** Calibration experiment C05

A single positive scalar temperature fitted with each image weighted by its frozen, out-of-fold v1 case-reliability score will achieve whole-image ECE at most `0.04`, while neither whole-image ECE nor image-balanced whole-image NLL worsens within the fixed lowest-reliability quintile. These three checks are conjunctive.

The bottom quintile is fixed before fitting as the 136 lowest v1 OOF reliability scores, with normalized path as the deterministic tie-break. During each calibration-fold fit, a case's total pixel weight is proportional to its frozen reliability score and normalized within that fit fold; every pixel within a case shares that case weight equally. The fitted transform receives only E08 raw logits at inference. The reliability score influences development fitting only and is not an inference-time routing input.

The v1 scores are reused because they are the only accepted prediction-derived, out-of-fold reliability scores available before Stage 4. This domain-transfer limitation is reported. Dice and reference-mask geometry do not enter the calibration weights. Reference masks provide pixel targets and evaluation regions only.

The threshold remains `0.50`; hard-mask changes are descriptive. No neural model is retrained and no sealed-test image is accessed.
