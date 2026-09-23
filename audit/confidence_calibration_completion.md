# Confidence calibration completion report

All three frozen-model confidence studies are closed. The classifier keeps its raw probabilities, segmenter pixel temperature scaling is rejected under the unchanged gate, and the case-level segmenter reliability model passes for later application design. Reported calibrated and reliability metrics are out-of-fold; no test data was accessed.

| Component | Result | Decisive evidence | Accepted output |
|---|---|---|---|
| Classifier confidence | Closed | Raw four-class ECE `0.0084` and binary ECE `0.0021`; both temperature gates failed to show an accepted stable improvement | Raw classifier probabilities |
| Segmenter pixel confidence | Closed | Foreground ECE improved `0.0755` → `0.0514`, but remained `0.0014` above the `0.05` gate; background NLL worsened by `0.0003` | Raw segmenter probabilities |
| Segmenter case reliability | **Pass** | AUROC `0.9410` (95% CI `0.9184`–`0.9612`); top-risk 20% captured `14/17` zero-Dice cases; ECE `0.0212`; Brier skill `0.4761` | Reliability score eligible for later review-policy design |

## Findings that matter

Boundary entropy is the strongest observed failure signal: mean standardized coefficient `-1.0378` with a negative sign in `5/5` folds. High-confidence foreground fraction is also negatively associated with reliability (mean `-0.0863`, negative in `5/5` folds). This supports the earlier finding that spatial failures can be confident. These coefficients are descriptive associations, not causal effects; features are correlated and the study contains 680 cases.

**Glioma remains the weakest subgroup.** The aggregate AUROC of `0.9410` is not uniform. Glioma AUROC is `0.8475`, reliability ECE is `0.0883` versus `0.0212` overall, and glioma contributes `60/83` Dice-below-0.50 failures plus `12` zero-Dice cases. The aggregate pass remains valid under its declared gate, but it does not establish subgroup equivalence. Per-category metrics are descriptive and have no confidence intervals. Later use must keep this qualification visible.

## What this work establishes

- The frozen classifier's raw probabilities are acceptably calibrated on the BRISC internal-validation partition under the declared gates.
- Pixel temperature scaling improved several segmenter calibration metrics but did not satisfy the full fixed gate, so it was rejected.
- Prediction-derived case features can rank Dice-below-0.50 failures in similarity-aware out-of-fold evaluation on these 680 validation cases.

## What this work does not establish

- Clinical validity, safety, diagnosis, or autonomous use.
- Patient-independent, cross-dataset, scanner-independent, or prospective performance.
- Uniform reliability across tumor types; glioma remains materially weaker.
- Guaranteed catastrophic-failure capture; 3/17 zero-Dice and 15/83 Dice-below-0.50 cases remain outside the top-risk quintile.
- Acceptance of calibrated segmenter pixel probabilities; raw pixels remain in force.
- Prospective performance of the final all-validation case model; reported metrics are out-of-fold.
- Causal meaning for any reliability-model coefficient.

## Handoff

The only new confidence output eligible for the next application stage is the case-level score `P(Dice ≥ 0.50)`. Its features come only from the raw prediction map and unchanged `0.50` hard mask; reference masks created and evaluated the development target but supplied no features. The final model fitted on all validation cases is a proposed operational model only and did not produce the reported metrics. The next stage may design and validate how this score maps to display, warning, or required-review states. It must retain the glioma warning and must not describe the score as clinical certainty.

No base model was retrained, no test image or prediction was accessed, the segmentation threshold remained `0.50`, no gate was relaxed, and no application integration was performed.
