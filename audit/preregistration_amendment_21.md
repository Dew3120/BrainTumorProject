# Pre-registration amendment 21 — bounded response to external AI adversarial review

**Status:** REGISTERED AFTER EXTERNAL AI REVIEW AND BEFORE R11, R12, O07, OR P06 EXECUTION

**Registered:** 2026-09-30

**Authorizer:** User

**Evidence state before registration:** commit `ed00c71464698c7a21d920250b2e17997ba6995d`

**Frozen-stack manifest SHA256:** `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`

**Frozen-stack Git tree:** `439b4fca4b47f12b173c24eaa039b1258f9ddf03`

## Reason for amendment

Three user-solicited external AI adversarial reviews challenged the interpretation of internal performance, the E08 selection pathway, the size–Dice association, external reliability ranking, O05 source sensitivity, multiplicity, sealed-test enforcement, and the lack of a joint policy evaluation. These reviews are critique evidence, not human peer review, clinical review, or new empirical evidence.

This amendment authorizes a bounded response: four read-only or comparator-only analyses and documentation corrections. It does not reopen completed gates, reinterpret failed hypotheses as passed, modify the final stack, or authorize broad new model development.

The experiment registry contained **49 experiment packages** at the evidence cutoff. Of these, **33 are numbered E/C/R/O experiments**: E01–E08, C01–C09, R01–R10, and O01–O06. The remaining 16 are parent, selection, policy, external-validation, sealed-evaluation, application-integrity, and measurement packages. This distinction must be preserved in later documentation.

## Registered interpretation changes

The following are documentation corrections, not changes to results:

1. Small-target wording becomes: “Small target area is the strongest observed association in this model and dataset. Because Dice is mathematically coupled to object size, this is partly a metric artifact and is not a causal claim.” Surface Dice, Hausdorff distance, and lesion-wise detection are listed as size-robust complementary metrics.
2. Every summary, table, or figure caption that gives external reliability AUROC must give the corresponding ECE in the same entry: OpenNeuro AUROC `0.8719`, ECE `0.2551`; UPENN-GBM AUROC `0.8316`, ECE `0.3391`. No unqualified “reliability transferred” statement is permitted.
3. The narrative leads with H8 rejection and independent spatial failure: OpenNeuro strict subject-macro Dice `0.1272` and UPENN-GBM strict subject-macro Dice `0.2208`. Internal values are development diagnostics that did not establish external generalization.
4. Documentation discloses 49 total registered experiment packages and the 33 numbered E/C/R/O experiments.
5. “BRISC sealed test not accessed” is documented as a process commitment supported by scripts, file-path discipline, and the append-only review log; it is not a cryptographic or operating-system access-control guarantee.
6. The absence of joint OOD/reliability risk-coverage analysis remains a limitation until P06 reports the fixed-policy behavior. P06 is retrospective and cannot establish prospective safety.
7. No nnU-Net, published BRISC reproduction, or other strong compute-matched segmentation baseline was evaluated.
8. External spatial failure has no positive control trained on an external subset through the same adapter, so model weakness and protocol/label mismatch are not fully separable.
9. R02 feature reduction, O05 secondary ensemble selection, P02 threshold selection, the post-H1 H7 pathway, and the glioma subgroup predictor search are labeled exploratory or development model selection rather than independent confirmation.

## H10 — size-robust directional comparison

On the fixed 204-case BRISC glioma validation subset, E08's mean surface Dice at a two-model-pixel tolerance will be greater than v1's mean surface Dice, matching the already observed positive direction of ordinary Dice.

H10 passes only if `mean_surface_dice_E08 - mean_surface_dice_v1 > 0`. Equality does not pass because it does not preserve the positive direction. HD95 and lesion-wise detection are secondary descriptive metrics and cannot reverse the gate.

## R11 — `r11_size_robust_metrics`

### Population and models

- Exactly the 204 fixed BRISC development-validation glioma image/mask pairs used by v1 and E08 comparison.
- Frozen v1 and frozen E08 inference only. No weights, calibration maps, routes, thresholds, or preprocessing may change.
- Hard masks use the unchanged segmentation threshold `0.50` in 128×128 model space.
- BRISC sealed-test membership is prohibited.

### Metrics

1. **Surface extraction:** an 8-connected binary foreground surface is `mask XOR binary_erosion(mask, 3×3 all-ones structure, border_value=0)`.
2. **Surface Dice at 2 pixels:** the sum of predicted-surface pixels within Euclidean distance `<=2` of the reference surface and reference-surface pixels within `<=2` of the predicted surface, divided by the total number of surface pixels in both masks.
3. **HD95:** the 95th percentile of the union of bidirectional nearest-surface Euclidean distances in model pixels.
4. **Empty-mask rules:** a nonempty reference with empty prediction receives surface Dice `0` and HD95 equal to the 128×128 pixel-centre diagonal `sqrt(127^2 + 127^2)`. Both empty receives surface Dice `1` and HD95 `0`, although an empty glioma reference is an integrity failure. Non-finite values stop the analysis.
5. **Lesions:** reference and prediction components use 8-connectivity. Components are matched one-to-one by maximum component Dice using a deterministic optimal assignment; index order breaks exact ties. A reference lesion is detected when its matched component Dice is `>=0.50`. Lesion detection rate is detected reference lesions divided by all reference lesions.
6. Ordinary hard Dice is recomputed to verify the registered positive E08-minus-v1 direction and exact agreement with the frozen source artifact within `1e-12`.

### Uncertainty

Paired similarity-group bootstrap: 2,000 resamples, seed `42`, percentile 95% CI, same sampled groups for v1 and E08. The reported differences are E08 minus v1 for Dice, surface Dice, and lesion detection, and v1 minus E08 for HD95 so positive always favors E08.

## H11 — external ranking beyond trivial predictors

On each of X02 OpenNeuro and X03 UPENN-GBM separately, frozen R02 failure-detection AUROC will exceed both BRISC-fitted one-feature comparator AUROCs by at least `0.03`.

The gate is conjunctive across two cohorts and two comparators. For each cohort and comparator, `AUROC_R02 - AUROC_comparator >= 0.03`; equality passes. Failure of any comparison rejects H11. No pooled external AUROC may replace a cohort result.

## R12 — `r12_r02_external_baselines`

### Comparator fitting

- Two independent one-feature comparators are authorized: `predicted_area_fraction` and `maximum_calibrated_foreground_probability`.
- Each uses `StandardScaler` followed by L2 logistic regression with `C=1.0`, solver `lbfgs`, `max_iter=10000`, and fixed seed `42` where applicable.
- Target is `P(Dice >= 0.50)` under frozen E08+C03 at threshold `0.50`; reported failure risk is `1 - P(success)`.
- Development metrics use the existing five fixed similarity-aware folds and strictly out-of-fold predictions on the same 680 BRISC cases as R02.
- After OOF generation, each comparator is fitted once on all 680 BRISC cases, serialized and hashed. This operational comparator fit must occur before external target labels or performance values are joined in the execution script.
- The serialized comparators are applied unchanged to X02 and X03. External cases, labels, Dice values, or patient groups may not influence scaling, coefficients, intercepts, feature direction, hyperparameters, or missing-value handling.

This narrowly authorized comparator fitting is not an R02 refit, not calibration, not operational model selection, and not a final-stack change. R02, C03, E08, and all policy artifacts remain frozen. No other reliability fitting is authorized.

### Evaluation

- BRISC: 680 fixed development-validation cases, with the existing similarity-aware folds.
- X02 primary strict population: all 425 registered tumor-bearing slices, including target-not-representable cases with Dice `0` under Amendment 15.
- X03 primary strict population: all 9,464 registered tumor-bearing slices, including target-not-representable cases with Dice `0` under Amendment 17.
- Failure is frozen calibrated hard-mask Dice `<0.50`.
- Report failure prevalence, AUROC, failure-class average precision, and patient/group-bootstrap 95% CIs.
- Report risk-coverage curves and accepted-case mean Dice at retained coverages `0.10, 0.20, …, 1.00`. Cases are accepted from lowest predicted risk upward; normalized case ID breaks ties.
- Also report R02's fixed-policy accepted coverage and accepted-case Dice. Comparator values at that same coverage are rank-matched descriptive comparisons, not deployable thresholds.
- External bootstrap uses patient as the unit; BRISC uses similarity group. Use 2,000 percentile resamples with seed `42` and paired samples for R02-minus-comparator differences.

The external cohorts were previously analyzed in Stage 6. R12 is an exploratory post-review baseline analysis, not untouched confirmation.

## H12 — OOD score/failure association description

H12 is descriptive. It asks whether higher frozen O05 OOD score is associated with lower E08+C03 Dice and enriched Dice-below-0.50 failures within X02 and X03. It has no pass/fail gate and cannot tune O05 or the O06 threshold.

## O07 — `o07_ood_failure_correlation`

- Populations are the same 425 X02 and 9,464 X03 primary strict tumor-bearing slices used by R12.
- Recompute the selected frozen O05 score through the final-stack manifest and verify registered fixture outputs before cohort inference.
- Report Spearman correlation between O05 score and calibrated Dice. Higher OOD score and worse Dice imply a negative correlation.
- Report AUROC and failure-class average precision for O05 score detecting Dice `<0.50`.
- Define the high-OOD group as the `ceil(0.20*N)` largest O05 scores, ties broken by normalized case ID. Report its failure prevalence, the cohort failure prevalence, enrichment ratio, failures captured, and a patient-bootstrap 95% CI.
- Use 2,000 patient-bootstrap percentile resamples with seed `42`.
- No fitting, threshold selection, O05 modification, cohort pooling, or external-data tuning is allowed.

## H13 — fixed joint-policy risk-coverage description

H13 is descriptive. It asks how the already frozen O06 and R02 policies behave jointly on BRISC validation, X02, and X03. It has no pass/fail gate and cannot select or change a threshold.

## P06 — `p06_joint_policy_risk_coverage`

### Fixed rule

1. Compute frozen O05 score.
2. O06 is `ACCEPT` only when OOD score `<0.36854275476100085`; otherwise the case is `REVIEW_REQUIRED`. REFUSE remains disabled.
3. For O06-accepted cases, R02 is `STANDARD` only when reliability `>=0.9490250173235835`; otherwise it is `REVIEW_REQUIRED`.
4. The joint status is `STANDARD` only when both rules accept. Every other case is `REVIEW_REQUIRED`.
5. Missing, invalid, or non-finite scores fail closed to `REVIEW_REQUIRED`.

### Populations and reporting

- BRISC uses the 680 development-validation cases, held-out O05 scores, and R02 out-of-fold reliability scores. These are internal development estimates.
- X02 uses all 425 primary strict tumor-bearing slices and the frozen full-population R02 operational artifact.
- X03 uses all 9,464 primary strict tumor-bearing slices and the frozen full-population R02 operational artifact.
- Report STANDARD and REVIEW_REQUIRED counts and rates; mean Dice and Dice-below-0.50 rate among STANDARD cases; failures caught, failures missed, sensitivity for failure review, and accepted-case category results.
- BRISC intervals resample similarity groups; external intervals resample patients. Use 2,000 percentile resamples with seed `42`.
- Report results separately by cohort. No pooled result may replace a cohort result.
- “Prevents poor masks” may be stated only as the observed proportion of failures assigned REVIEW_REQUIRED. It must not be described as safety, clinical protection, or prospective validation.

## Deferred structural work

The following are documented future work and are not authorized by this amendment:

1. external positive control trained or fine-tuned on a UPENN subset;
2. patient-independent BRISC resplitting without verified IDs;
3. nnU-Net, published-BRISC, or other strong baseline comparison;
4. encoding-harmonized O05 re-evaluation;
5. OSF or signed external timestamping of future preregistration;
6. compute-matched four-seed v1 ensemble;
7. E08 routing ablation;
8. nested cross-validation for the reliability threshold;
9. higher-resolution or 3D neural retraining; and
10. any change to the frozen final stack.

## Execution and stopping rules

Execution order is fixed:

1. register this amendment and all four experiment packages; commit and pause;
2. execute R11, document, commit, and pause;
3. execute R12, document, commit, and pause;
4. execute O07, document, commit, and pause;
5. execute P06, document, commit, and pause;
6. apply documentation-only corrections and commit;
7. write the Stage 10 revision addendum, commit, push, and pause before application work.

Each experiment must retain its own hypothesis, config, result, decision, artifacts, manifest, review-log entry, and commit. Experiments may not be skipped, merged, substituted, or used to alter another experiment's gate.

Any source-hash mismatch, case-membership mismatch, fold mismatch, metric reproduction mismatch, non-finite required value, sealed-test access, threshold drift, final-stack change, or unauthorized fit is `INVALID/STOP`. It is not a failed scientific gate and does not authorize an automatic variant.

## Constraints retained

- BRISC sealed-test access remains prohibited.
- Neural retraining remains prohibited.
- C03, R02, O05, and all operational artifacts remain frozen. The only new fits are the two explicitly registered, non-operational R12 comparator models trained on BRISC development data.
- Segmentation threshold remains exactly `0.50`.
- Reliability threshold remains exactly `0.9490250173235835`.
- OOD review threshold remains exactly `0.36854275476100085`; REFUSE remains disabled.
- `runs/final_stack/` is immutable.
- No gate relaxation is permitted after results are inspected.
- Results from these post-review analyses are exploratory and must be labeled as such.
