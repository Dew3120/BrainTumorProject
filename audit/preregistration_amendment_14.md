# Pre-registration amendment 14 — deployable R02 reliability estimator

**Status:** REGISTERED BEFORE OPERATIONAL FIT

**Registered:** 2026-09-28

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 6 parent amendment:** `audit/preregistration_amendment_13.md`, SHA256 `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5`

This amendment prospectively authorizes exactly one deterministic full-population fit-and-freeze operation for the already selected R02 reliability estimator. It resolves Amendment 13's `BLOCKED_ARTIFACT_MISSING` condition. It does not authorize neural-model training, model selection, hyperparameter search, external-data access, BRISC sealed-test access, calibration refitting, segmentation-threshold changes, or replacement of the reported R02 out-of-fold results.

## Fixed purpose

The fit exists only to make the selected R02 estimator deployable on future cases. The reported reliability performance remains the five-fold out-of-fold R02 evaluation. Metrics computed on the 680 fitting cases by this operational estimator are not evaluation results and must not be reported as evidence of generalization.

## Fixed population and inputs

- Population: all 680 C03 out-of-fold-calibrated E08 BRISC development-validation cases.
- Target: `1` when the C03-calibrated threshold-0.50 hard-mask Dice is at least `0.50`; otherwise `0`.
- Expected target counts: 598 reliable and 82 unreliable cases.
- Expected similarity groups: 583.
- Expected C03 hard-mask changes: 22,891 pixels across 679 images.
- Reference masks are used only to reconstruct the already registered fitting target. They are never estimator features.
- Feature values come only from the C03-calibrated probability map and its unchanged threshold-0.50 prediction.

The feature order is locked:

1. `empty_prediction`
2. `predicted_area_fraction`
3. `whole_probability_mean`
4. `whole_probability_sd`
5. `predicted_region_entropy_mean`
6. `boundary_entropy_mean`
7. `high_confidence_foreground_fraction`
8. `uncertain_pixel_fraction`
9. `bounding_box_fill_ratio`
10. `normalized_perimeter`

## Fixed estimator

Fit `StandardScaler`, followed by scikit-learn `LogisticRegression` with L2 regularization, `C=1.0`, solver `lbfgs`, `class_weight=None`, `max_iter=5000`, and `random_state=42`. Fit once on all 680 cases. No search, selection, resampling, external data, data-dependent choice, or alternative estimator is allowed.

## Frozen provenance

- R02 result: `experiments/r02_reduced_features/result.json`, SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`.
- R02 manifest: `experiments/r02_reduced_features/artifacts/experiment_manifest.json`, SHA256 `1dc386a3cd342f14fcb827960c1ec89f749b2413962c768aec1c22bdf74cb819`.
- R02 OOF table: `experiments/r02_reduced_features/artifacts/oof_model_comparison.csv`, SHA256 `9b986435b0da0fdb340a16730209ebb895115ece694386c298830e3956b2fc6a`.
- R07 result: `experiments/r07_coefficient_bootstrap/result.json`, SHA256 `fc8b1bd80826913ff990ea993bf1c0c8bdf028b614101f967afc4282c01cb948`.
- R07 coefficient table: `experiments/r07_coefficient_bootstrap/artifacts/coefficient_summary.csv`, SHA256 `b1f72477a76afa2ed74cb01481d3e514d339d405f38cc2ced5576a22cbbe2b52`.
- R07 manifest: `experiments/r07_coefficient_bootstrap/artifacts/experiment_manifest.json`, SHA256 `afddd3023655ec9fc539611a50eef3335d36923894e49aa99e831f9743e5dd10`.
- C03 mapping: `experiments/c03_isotonic/artifacts/isotonic_mappings.json`, SHA256 `fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa`.

## Verification and stopping rule

R07 recorded the standardized full-population coefficients, but it did not persist a full-population probability vector, intercept, or scaler state. Therefore a literal comparison against historical saved probabilities is impossible. This absence is recorded rather than concealed.

Before serialization, the operation must:

1. reproduce every R07 full-population standardized coefficient with maximum absolute error at most `1e-12`;
2. independently rebuild the same feature matrix and fit the fixed estimator a second time;
3. reproduce scaler means, scaler scales, coefficients, intercept, and all 680 full-population probabilities between the two independent fits with maximum absolute error at most `1e-12`;
4. reload the serialized scaler and logistic estimator and reproduce all 680 reference probabilities with maximum absolute error at most `1e-12`;
5. verify all frozen hashes, population counts, feature order, feature contract, target counts, C03 change counts, and threshold `0.50`.

Any mismatch stops the operation without declaring a deployable model and without proceeding to external reliability inference.

## Serialization

On success, create:

- `models/r02_reliability_scaler.pkl`
- `models/r02_reliability_logistic.pkl`
- `models/r02_reliability_manifest.json`

The manifest records the feature order, scaler mean and scale, standardized coefficients, intercept, class order, estimator parameters, runtime versions, source hashes, artifact hashes, verification errors, the fitting-population definition, and the distinction between operational fitting and OOF evaluation. A reference prediction table is retained inside `experiments/x00_r02_deployable_model/artifacts/` for exact future verification.

## Interpretation

This is reliability-estimator operational fitting, not neural retraining and not a performance evaluation. The classifier, E08 segmenter components, and C03 mapping remain frozen. No external image or BRISC sealed-test pixel may be touched. The segmentation threshold remains exactly `0.50`; the reliability policy threshold remains exactly `0.9490250173235835`.
