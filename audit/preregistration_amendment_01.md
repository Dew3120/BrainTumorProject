# Pre-registration amendment 01 — glioma improvement and reliability program

**Status:** REGISTERED BEFORE NEW ANALYSIS

**Registered:** 2026-09-24

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

This amendment registers the user-requested glioma decomposition, controlled model interventions, calibration comparisons, reliability comparisons, and review policy. It does not change any completed result or parent gate. Experiment-specific gates below determine whether an intervention is locally accepted; the parent G1 gate still determines whether a final candidate has demonstrated a material glioma improvement.

All experiments use the fixed BRISC inventory/split SHA256 `d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28`, fixed similarity candidates SHA256 `dab1cdcef3173315544d7d39807b54cd356d0a39223885df840f83ebd5cca1a0`, v1 manifest SHA256 `c8a36879aa6e717d60641c5930ebb083aa3b362d98d10f2601584633d4e88a80`, hard-mask threshold `0.50`, bootstrap seed `42`, and 2,000 similarity-group resamples unless an experiment config states otherwise. The sealed test remains unopened.

## A. Glioma failure decomposition

### A1. Case audit

The population is the 60 glioma validation cases with v1 hard-mask Dice `<0.50`. Geometry is measured in the 128×128 model coordinate system with 8-connectivity. Bounding-box overlap is bounding-box IoU. Centroid distance is Euclidean pixels; empty-prediction geometry is missing rather than imputed. Orientation is parsed from filename tokens `ax`, `co`, and `sa` as axial, coronal, and sagittal.

Reference-mask components and prediction components are recorded both as raw counts and as counts with area `>=4` pixels. Focality is single if exactly one reference component has area `>=4`, multifocal if more than one, and indeterminate if none. MRI edema and contrast enhancement are recorded as `not_assessable` when sequence/contrast metadata are insufficient; they will not be guessed from an unverified single JPEG. Ventricle proximity is a visual research annotation (`near`, `not_near`, or `indeterminate`) and is not a clinical finding.

### A2. Boundary sharpness

For all 204 glioma validation cases, use the same resized grayscale `[0,1]` model image and unchanged reference mask. The boundary is `mask XOR binary_erosion(mask)` using an 8-connected 3×3 structure. Gradient magnitude is `skimage.filters.sobel(image)`. Record mean, median, maximum, 10th percentile, and the fraction of boundary pixels with gradient `>=0.10`. A boundary is predeclared **sharp** when median gradient `>=0.10` and sharp-pixel fraction `>=0.50`; otherwise it is **diffuse**. Empty boundaries invalidate the case.

### A3. Matched case study

Failure means Dice `<0.50`; strong success means Dice `>=0.80`. Select 10 failure/success pairs without replacement. Candidates must be within ±20% reference area. Prefer identical orientation, then minimize absolute log area ratio, then use normalized path as deterministic tie-break. If fewer than 10 same-orientation pairs exist, allow orientation mismatch and report it. No pair is selected using visual appearance.

### A4. Subgroups

- Size: small `<100` reference pixels, medium `100–500`, large `>500`.
- Boundary: sharp/diffuse by A2.
- Focality: single/multifocal by A1; indeterminate is reported separately.
- Location: normalize centroid distance from image centre by the centre-to-corner distance. Central is `<=0.35`; peripheral is `>0.35`.

For every characteristic, report count, mean Dice, and 95% similarity-group bootstrap percentile CI across all cases and separately for failure/success labels. The failure/success mean-Dice contrast is outcome-defined and therefore descriptive, not causal. To assess prediction, report failure rate, risk difference, odds ratio with Haldane–Anscombe `+0.5` correction where needed, and bootstrap CI. A characteristic is called associated with failure only when its risk-difference CI excludes zero. No multivariable causal claim is allowed.

### A5. Literature review

Review 15–20 primary papers. For each, record method, reported Dice and target definition, subgroup reporting, glioma-difficulty statement, dataset, and limitations. BRISC evidence is kept separate from BraTS whole-tumor/tumor-core/enhancing-tumor metrics because their labels are not interchangeable.

## B. Controlled glioma interventions

Every intervention receives its own experiment directory and is evaluated on the same 680 validation pairs against frozen v1. Category change means candidate minus v1 image-mean Dice. Experiments run in the listed order even if an earlier experiment passes.

| ID | Intervention | Fixed implementation | Local acceptance gate |
|---|---|---|---|
| E01 | Largest component | Keep only the largest 8-connected predicted component; path tie-break is top-left pixel | Glioma change `>=+0.02`; meningioma and pituitary each `>=-0.01` |
| E02 | Morphology | Closing then opening with square 3×3 and 5×5 kernels; variants reported separately | A kernel passes if glioma `>=+0.02` and every category `>=-0.01`; select 3×3 over 5×5 on an exact tie |
| E03 | TTA | Identity, 90/180/270 rotations, horizontal/vertical flip, transpose, anti-transpose; invert transforms, average probabilities | Glioma `>=+0.03`; every other category `>=-0.01` |
| E04 | Multiscale | Inference at 128 and 256; resize 256 probability to 128 with bilinear interpolation, average, threshold 0.50 | Glioma `>=+0.02`; every other category `>=-0.01` |
| E05 | Glioma-weighted retraining | Fresh v1 architecture and schedule, seed 42, glioma case loss multiplier 3 | Glioma `>=+0.05`; meningioma/pituitary each `>=-0.01` |
| E06 | Boundary-loss retraining | Fresh v1 training with balanced BCE + Dice + Kervadec boundary loss, equal top-level weights | Glioma `>=+0.05`; glioma boundary F1 change `>=+0.05`; other categories reported |
| E07 | Glioma specialist | Fresh glioma-only segmenter; route using classifier-predicted type, general v1 otherwise | Routed glioma `>=+0.08`; classifier routing accuracy `>=0.95`; meningioma/pituitary `>=-0.01` |
| E08 | Ensemble | Average probability maps of E05–E07; threshold 0.50 | Glioma `>=+0.02` over the best eligible single model; meningioma/pituitary `>=-0.01` versus v1 |

Boundary F1 uses predicted/reference one-pixel boundaries, a two-pixel tolerance via binary dilation, precision and recall with zero division returning zero, and harmonic mean. Retrained experiments repeat the frozen validation, visual audit, failure analysis, pixel calibration, and case reliability protocols; no sealed test access is allowed.

### Final candidate selection

This is predeclared model selection, despite the requested “post-hoc” label. Exclude candidates that violate their category-harm constraint. Rank remaining candidates lexicographically by: (1) highest glioma mean Dice, (2) lowest accepted pixel ECE, (3) highest case-reliability AUROC, (4) highest macro-type Dice, and (5) experiment ID ascending. A selected candidate must also pass parent G1 before replacing v1. If none passes parent G1, v1 remains final.

## C. Calibration comparisons on the selected model

All reported development metrics are five-fold similarity-aware out-of-fold metrics. Scaling/mapping is fitted inside training folds. No method may use a case's reference mask to choose an operational transform at inference.

| ID | Method | Acceptance gate |
|---|---|---|
| C01 | Vector scaling: diagonal scale and bias for classifier; two-logit binary vector map for pixels | Foreground/background/whole ECE each `<=0.05`; corresponding NLL improves |
| C02 | Dirichlet calibration on two-class pixel probabilities | ECE `<=0.05`; NLL improves; hard masks unchanged at 0.50 |
| C03 | Isotonic regression | ECE `<=0.04`; Brier does not worsen; hard-mask changes reported |
| C04 | Separate foreground/background scalar temperatures | Foreground ECE `<=0.05`; background ECE `<=0.02` |
| C05 | Reliability-weighted calibration | Aggregate ECE `<=0.04`; low-reliability-quintile ECE and NLL do not worsen |
| C06 | Per-predicted-category pixel temperatures | ECE `<=0.05` within each predicted category |
| C07 | Nested hierarchical combination of C04–C06 | Aggregate ECE `<=0.03`; every predicted-category ECE `<=0.05` |

C04 is an oracle diagnostic when foreground/background fitting depends on the reference-centred region; it cannot be selected operationally unless routing features are prediction-only. C07 configuration selection occurs in an inner training-fold loop; its outer held-out fold supplies the reported metric.

All calibration studies additionally report NLL, Brier, ECE, MCE, hard-mask changes, per-category results, and bootstrap intervals. Recomputing earlier work with 15 equal-frequency bins is a secondary standardized comparison and does not retroactively alter any original gate or decision. Six category diagrams are descriptive artifacts.

Among operational methods passing their gates, select lowest aggregate ECE, then lowest NLL, then lowest Brier, then experiment ID. If none passes, raw probabilities remain accepted.

## D. Reliability comparisons on the selected model

The verified v1 reliability model has 14 features, not 15. R01 therefore starts with the exact 14-feature list in `audit/segmenter_case_reliability.json`. Empty prediction and border-touch fraction are the two zero-variance features in v1.

| ID | Method | Acceptance gate or decision rule |
|---|---|---|
| R01 | Leave-one-feature-out ablation and 2,000 group-bootstrap permutation importance | Descriptive; no model selection |
| R02 | Remove the two zero-variance features | AUROC `>=` same-fold 14-feature logistic AUROC; ECE does not worsen by `>0.01` |
| R03 | Add circularity, solidity, boundary fractal dimension, boundary image-gradient, centre distance, vertical symmetry, brain-boundary distance | AUROC change `>=+0.01`; ECE does not worsen by `>0.01` |
| R04 | Separate category models routed by classifier prediction | AUROC `>=0.80` in each predicted category; report CIs and case/failure counts |
| R05 | One model with classifier-predicted category one-hot | Glioma AUROC `>=0.88`; aggregate AUROC not lower than baseline |
| R06 | XGBoost, 2×32 MLP, and 100-tree random forest versus logistic regression | AUROC change `>=+0.02`; ECE not worse by `>0.01`; otherwise retain logistic regression |
| R07 | Coefficient bootstrap | Descriptive 2,000-resample mean, SD, percentile CI and sign stability |
| R08 | Fold-wise prospective report | Report each of five existing outer held-out folds, mean and sample SD; no new fit on held-out data |
| R09 | Fold sensitivity | Three new StratifiedGroupKFold assignments with seeds 17, 42, 93; report mean, SD, range |
| R10 | Seed stability | Fit seeds 7, 21, 42, 84, 168 on identical folds; report variance; deterministic equality is a valid result |

The raw image-gradient and brain-boundary-distance features in R03 are image-derived, not prediction-only; they must be labeled accordingly. True tumor type and reference-mask geometry are prohibited as features. All preprocessing and hyperparameters are fitted within each outer training fold.

Final reliability selection filters to models passing their local gate, then ranks by aggregate AUROC, glioma AUROC, lower ECE, higher Brier skill, and simpler model class. A nonlinear model must earn its `+0.02` gate; otherwise interpretability wins.

## E. Reliability policy

The development target is segmentation failure `Dice <0.50`. Reliability is `P(Dice>=0.50)`.

### Cost model

- Flagged/reviewed and actually bad: cost `1`.
- Flagged/reviewed and actually acceptable: cost `1`.
- Accepted and acceptable: cost `0`.
- Accepted and actually bad: cost `10`.

These weights are an engineering preference, not a clinical utility estimate. The tenfold missed-failure penalty encodes the declared preference to spend review time rather than silently accept a bad mask.

### Binary review threshold

Sweep every distinct out-of-fold reliability value plus endpoints. Predict review-required when reliability is below the threshold. Minimize mean cost. Ties use fewer accepted failures, then lower review rate, then higher threshold. Report the complete curve and do not refit the reliability model during threshold selection.

### Three-state policy

For `theta_low <= theta_high`: review required below `theta_low`, warning from `theta_low` up to but excluding `theta_high`, standard at or above `theta_high`. Exhaustively evaluate observed-score pairs using costs: review-required `1`; warning acceptable `0.25`; warning bad `5`; standard acceptable `0`; standard bad `10`. Select minimum mean cost; ties use fewer standard-state failures, fewer warning-or-standard failures, lower review rate, then higher thresholds. Report review rate, warning rate, false alarms, and missed failures.

Threshold sensitivity shifts each selected threshold by `-0.10` and `+0.10`, clipped to `[0,1]`, and evaluates the Cartesian combinations that preserve `theta_low<=theta_high`. User-facing copy must call outputs research-only, state that reliability is not diagnosis, and give a glioma-specific caution.

## F. Multiple-comparison and stopping rules

1. Every listed experiment is run and reported; an earlier pass does not suppress later results.
2. Local gates are fixed and conjunctive. No gate is relaxed after seeing results.
3. The final candidate and calibrator use only the predeclared rankings above.
4. Standardized ECE and additional seeds are sensitivity analyses, not replacements for primary metrics.
5. A failed experiment is closed. Any unlisted variant requires amendment 02 before execution.
6. Missing hashes, changed split membership, non-finite metrics, or test access produce `INVALID/STOP`.
7. The sealed test remains reserved for the final locked system under the parent registration.
