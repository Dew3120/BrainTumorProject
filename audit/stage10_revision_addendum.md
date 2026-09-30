# Stage 10 revision addendum: response to external AI adversarial reviews

## Status and scope

Amendment 21 is complete. This addendum records the project’s response to adversarial reviews supplied by the user from Claude, Grok, and DeepSeek. These are external AI critiques, not human peer review, clinical review, or publication acceptance. The response preserves every negative result, leaves `runs/final_stack/` unchanged, and does not alter any model, threshold, split, or prior audit artifact.

The binding conclusion remains that the frozen E08+C03+R02 stack failed independent spatial transport on the two evaluable cohorts. OpenNeuro subject-macro Dice was 0.1272 and UPENN-GBM subject-macro Dice was 0.2208. BRISC results remain internal development diagnostics.

## Criticisms accepted

The reviews identified several material weaknesses. The project accepts them as follows:

1. **Small-target wording was too causal.** Dice is mathematically coupled to target area. The corrected claim is: “Small target area is the strongest observed association in this model and dataset. Because Dice is mathematically coupled to object size, this is partly a metric artifact and is not a causal claim.”
2. **External reliability ranking was described too positively.** Every external AUROC is now paired with calibration: OpenNeuro AUROC 0.8719 with ECE 0.2551; UPENN AUROC 0.8316 with ECE 0.3391. The report no longer says reliability transferred without this qualification.
3. **The narrative gave internal performance too much priority.** H8 rejection and the two independent Dice results now lead the abstract, introduction, conclusion, README, and review package.
4. **Researcher degrees of freedom needed a count.** At the original external-review cutoff, the repository contained 49 experiment packages, including 33 numbered E/C/R/O packages and 16 policy, validation, or governance packages. Amendment 21 added four packages, producing 53 current registry entries.
5. **Sealed-test enforcement was overstated.** Non-access is a procedural commitment supported by scripts, path discipline, and the review log. It is not cryptographic or operating-system isolation.
6. **Joint policy behavior was unknown.** P06 now measures the fixed O06+R02 policy. On both independent cohorts it reviews every case, so the observed external protection is total abstention with zero standard-display coverage.
7. **Strong baselines are missing.** The study did not run nnU-Net, reproduce a published BRISC baseline, or compare against a strong compute-matched segmenter. R12 adds only simple reliability baselines.
8. **External evaluation lacks a positive control.** No model was fitted on an external subset to distinguish model weakness from protocol, preprocessing, or target-definition mismatch.
9. **Several later analyses are exploratory.** R02 feature reduction, O05 secondary-ensemble selection, P02 threshold selection, the post-H1 H7 pathway, and the glioma subgroup predictor search are now labeled exploratory or development model selection.

## Registered response experiments

### R11 — size-robust metrics

R11 compared v1 and E08 on the same 204 internal glioma cases grouped into 181 similarity groups. Ordinary Dice reproduced the committed source values exactly.

| Metric | v1 | E08 | Registered difference | 95% paired group-bootstrap CI |
|---|---:|---:|---:|---:|
| Ordinary Dice | 0.6090 | 0.6324 | +0.0234 | +0.0006 to +0.0453 |
| Surface Dice at 2 px | 0.5949 | 0.6092 | +0.0143 | −0.0108 to +0.0391 |
| HD95, pixels | 14.2392 | 11.8433 | +2.3960 improvement | +1.1069 to +3.8260 |
| Lesion-detection sensitivity | 0.7243 | 0.7290 | +0.0047 | −0.0383 to +0.0490 |

H10 passed because the preregistered point-direction gate required E08 surface Dice to exceed v1. The surface-Dice and lesion-detection intervals cross zero, so this is not evidence of a uniform improvement across all complementary metrics. HD95 provides the clearest size-robust evidence in E08’s favor. [R11 result](../experiments/r11_size_robust_metrics/result.json)

### R12 — simple external reliability baselines

Two one-feature logistic comparators were fitted on BRISC development data only, cross-fitted with the existing similarity-aware folds, frozen before external outcome joins, and applied unchanged to X02 and X03.

| Cohort | Comparator | R02 AUROC | Comparator AUROC | Difference | Gate ≥ +0.03 |
|---|---|---:|---:|---:|---|
| OpenNeuro | Predicted area fraction | 0.8719 | 0.8524 | +0.0195 | Fail |
| OpenNeuro | Maximum calibrated foreground probability | 0.8719 | 0.8455 | +0.0263 | Fail |
| UPENN | Predicted area fraction | 0.8316 | 0.8231 | +0.0085 | Fail |
| UPENN | Maximum calibrated foreground probability | 0.8316 | 0.7871 | +0.0445 | Pass |

Only one of four conjunctive checks passed, so H11 was rejected. R02 retained useful ranking on these cohorts, but this experiment does not show a consistently large advantage over trivial prediction-derived signals. [R12 result](../experiments/r12_r02_external_baselines/result.json)

### O07 — OOD score versus spatial failure

O07 asked whether the frozen O05 score ranks individual external masks by Dice failure. It did not tune O05 or any threshold.

| Cohort | Spearman OOD score vs Dice | Failure-detection AUROC | Highest-quintile failure enrichment | Failures captured by top quintile |
|---|---:|---:|---:|---:|
| OpenNeuro | −0.0833 | 0.5998 | 1.050× | 75/357 |
| UPENN | −0.1219 | 0.5638 | 1.073× | 1,631/7,600 |

The association is weak. O05 is supported as a retrospective source-shift detector on its registered benchmark, not as a replacement for R02 or as a reliable case-level mask-failure score. H12 was descriptive and is complete. [O07 result](../experiments/o07_ood_failure_correlation/result.json)

### P06 — joint fixed-policy risk coverage

P06 applied the frozen O06 and R02 thresholds unchanged. A case entered `STANDARD` only when O06 accepted it and R02 reliability was at least `0.9490250173235835`.

| Cohort | Standard | Review | Standard mean Dice | Failures caught | Failures missed |
|---|---:|---:|---:|---:|---:|
| BRISC validation | 450/680 | 230/680 | 0.8894 | 81/82 | 1 |
| OpenNeuro | 0/425 | 425/425 | Not estimable | 357/357 | 0 |
| UPENN | 0/9,464 | 9,464/9,464 | Not estimable | 7,600/7,600 | 0 |

On BRISC, the joint gate reduced standard-display coverage to 66.2% and left one observed failure in the standard path. On both independent cohorts it achieved 100% observed failure review by sending every case to review. This is total abstention, not evidence that the policy selectively accepts reliable external predictions, and not a clinical safety result. H13 was descriptive and is complete. [P06 result](../experiments/p06_joint_policy_risk_coverage/result.json)

## Claims revised

| Earlier framing | Revised framing |
|---|---|
| Small masks drive glioma failure. | Small target area is the strongest observed association; Dice-size coupling makes it partly a metric artifact, and no causal claim is made. |
| Reliability transfers externally. | Ranking AUROC remained above chance on X02/X03, but ECE was poor and R12’s simple-baseline gate failed. |
| OOD detection mitigates external failure. | O05 detects the registered source shifts, O07 shows weak per-mask failure ranking, and P06 achieves external protection only through total abstention. |
| Internal metrics lead the project result. | H8 rejection leads; internal metrics are development diagnostics. |
| The experiment trail is preregistered. | The trail is preregistered with amendments, but multiplicity and exploratory selection remain: 49 packages at review cutoff, 33 numbered. |
| The sealed test is protected. | Non-access is procedurally documented, not technically isolated. |

## Issues not fixed by Amendment 21

The following work remains deferred because it requires new data, neural training, a larger compute study, or a new prospective protocol:

- patient-independent BRISC reconstruction;
- nnU-Net, published BRISC, or compute-matched strong segmentation baselines;
- an external positive-control model fitted on a training subset and evaluated on different patients;
- O05 evaluation after removing encoding and file-statistic shortcuts;
- prospective timestamping and third-party artifact escrow;
- compute-matched E08 controls and routing ablation;
- nested cross-validation for feature, model, and reliability-threshold selection;
- prospective multi-site clinician-supervised validation.

These are material limitations. Their deferral prevents claims of state-of-the-art performance, causal failure mechanisms, broad generalization, clinical utility, or prospective safety.

## Governance and integrity

Amendment 21 was committed before R11–P06 results were generated. Each response analysis has its own hypothesis, config, result, decision, artifacts, tests, review-log entry, and commit. H10 passed its exact point gate; H11 failed and was not relaxed; H12 and H13 remained descriptive. No experiment was merged, substituted, or silently omitted.

The BRISC sealed test remained procedurally closed. Neural checkpoints, C03, R02, O05, the segmentation threshold `0.50`, reliability threshold `0.9490250173235835`, and OOD threshold `0.36854275476100085` were unchanged. The final-stack manifest SHA256 remains `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`, and its Git tree remains `439b4fca4b47f12b173c24eaa039b1258f9ddf03`.

## Final assessment

The external reviews improved the work by narrowing its claims and forcing four unanswered questions into registered tests. They did not convert the project into a clinically valid or externally generalizable tumor system. The defensible contribution is an undergraduate research-engineering and reproducibility study that exposes internal success, independent transport failure, calibration failure, baseline sensitivity, OOD shortcut risk, policy abstention, and governance limits in one auditable chain.

**Research prototype — not for clinical use.**
