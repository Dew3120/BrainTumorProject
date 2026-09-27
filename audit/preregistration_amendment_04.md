# Pre-registration amendment 04 — R02 reduced reliability model

**Status:** REGISTERED BEFORE R02 FITTING

**Registered:** 2026-09-27

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**R01 source result:** `experiments/r01_ablation/result.json`, SHA256 `d53b160a7aec1ab2c8b0d612576530cdcc530e1b7b30bb789d3e49bee3d9499d`

**R01 experiment manifest:** `experiments/r01_ablation/artifacts/experiment_manifest.json`, SHA256 `7e7162d4d1b33349c7944a8d8045898dbe37966d2c407b5feec556dbcfa51a7a`

## Prospective R02 hypothesis

The ten-feature reliability model selected by R01's registered candidate screen will preserve failure discrimination within `0.005` AUROC of the same-fold fourteen-feature R01 baseline on the 680 C03-calibrated internal-validation cases.

This claim is falsified if the reduced model's five-fold out-of-fold failure-detection AUROC is below `0.9495028142589118`. The fixed fourteen-feature comparator is `0.9545028142589118`.

## Locked candidate list

R01's screen retained a feature when its leave-one-feature-out AUROC decrease was at least `0.005` or its standardized coefficient had the same non-zero sign in all five folds. The resulting list, frozen before R02 fitting, is:

- `empty_prediction`
- `predicted_area_fraction`
- `whole_probability_mean`
- `whole_probability_sd`
- `predicted_region_entropy_mean`
- `boundary_entropy_mean`
- `high_confidence_foreground_fraction`
- `uncertain_pixel_fraction`
- `bounding_box_fill_ratio`
- `normalized_perimeter`

The four excluded features are:

- `predicted_region_probability_mean`
- `log1p_component_count`
- `largest_component_fraction`
- `border_touch_fraction`

No feature may be added, removed, substituted, or redefined after R02 results are observed.

## Locked protocol

- Population: the same 680 BRISC internal segmentation-validation cases used by R01.
- Inputs: each case's locked C03 five-fold out-of-fold isotonic-calibrated probability map and its threshold-`0.50` hard mask.
- Target: reliability is `P(Dice >= 0.50)`; failure risk is `1 - reliability`.
- Folds: the same five similarity-aware folds, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Estimator: `StandardScaler` plus L2 logistic regression (`C=1`, `lbfgs`, no class weighting, maximum 5,000 iterations, seed 42), fitted independently inside each outer training fold.
- Reported predictions: out-of-fold only.
- Bootstrap: 2,000 paired similarity-group resamples, seed 42, percentile 95% interval.
- Calibration: 15 equal-frequency bins with stable score ordering.
- Catastrophic capture: zero-Dice cases in the top 20% highest failure risk, with deterministic normalized-path tie breaking.
- Subgroups: true-category and classifier-predicted-category AUROCs are descriptive only.

## Gate and decision

R02 passes only when:

`reduced OOF AUROC >= full fourteen-feature OOF AUROC - 0.005`

Using the locked R01 comparator, this is:

`reduced OOF AUROC >= 0.9545028142589118 - 0.005 = 0.9495028142589118`

Equality passes. ECE, Brier score, training-prevalence baseline Brier, Brier skill, catastrophic-failure capture, subgroup AUROCs, coefficients, and sign stability must all be reported but do not create additional unregistered gates.

## Fail-closed rules

R02 stops without an accepted result or decision if any registered source hash differs; the candidate list or order differs; the C03 transformation does not reproduce exactly 22,891 changed pixels across 679 images; the population differs from 680; a fold or similarity group differs; the fourteen-feature baseline cannot be reproduced within `1e-12`; a feature receives reference-mask, classifier-probability, raw-image, path, identity, or metadata input; a non-finite value appears; the threshold changes; any sealed-test image is accessed; or any classifier/neural segmenter is retrained.

Reference masks create Dice targets and evaluation metrics only. Reliability-estimator fitting is the registered experiment and does not count as neural-model retraining. Gates will not be relaxed after results are seen.
