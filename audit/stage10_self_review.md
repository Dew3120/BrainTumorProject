# Stage 10 hostile self-review

**Review mode:** simulated workshop reviewer; skeptical, technically competent, and actively looking for rejection-level weaknesses.

**Review target:** the frozen research stack and the claims in [`docs/report.md`](../docs/report.md), not a clinical device.

**Evidence cutoff:** commit `b7cfdfd608412b87e5dadd7954ce9c06b910d9a3`.

**Scope rule:** this is an internal adversarial review. It is not an external peer review and does not claim endorsement by a clinician, radiologist, regulator, or independent researcher.

## Overall recommendation

**Recommendation if submitted as a clinical or generalizable medical-AI paper: reject.** The project lacks verified patient independence, prospective clinical evaluation, a sealed patient-level test result, and successful independent spatial transport. The strongest internal scores cannot support claims about patient care or general deployment.

**Recommendation if submitted as an undergraduate research-engineering or reproducibility case study: weak accept to borderline.** The work is unusually complete in preregistration, failure analysis, negative-result preservation, calibration/reliability separation, external failure reporting, OOD safeguards, and artifact freezing. Acceptance would depend on narrow claims: the contribution is the audited research process and the project-specific evidence chain, not a new segmentation architecture or clinically useful tumor detector.

## Severity scale

- **Fatal:** defeats a central claim or makes the claimed study design invalid.
- **Major:** materially limits interpretation and must be prominent.
- **Moderate:** important qualification or reproducibility gap that does not invalidate the entire study.
- **Minor:** clarity, presentation, or completeness issue.

## A. Methodological criticisms

### S10-M01 — Patient independence is unverified

- **Severity:** Fatal for generalization claims; major for internal-development claims.
- **Criticism:** BRISC does not provide complete patient identifiers, and visual-similarity groups are not patient IDs. Related slices from the same person may appear in different development folds. Cross-fitting, bootstrap intervals, calibration, and reliability evaluation can all be optimistic if patient clusters cross folds.
- **Evidence:** [`audit/preregistration.md`](preregistration.md); [`audit/classifier_training_findings.json`](classifier_training_findings.json); [`audit/segmentation_training_findings.json`](segmentation_training_findings.json).
- **Consequence:** Internal macro F1, Dice, ECE, and reliability AUROC cannot be described as unseen-patient performance. Confidence intervals do not incorporate unknown patient duplication.
- **Disposition:** **Acknowledge and document.** This cannot be repaired without trustworthy patient identities or a new patient-independent cohort.
- **Action:** Keep “internal development validation” next to every headline BRISC result. Reject any wording such as “generalizes to new patients.”
- **Residual risk:** The magnitude of inflation is unknown.

### S10-M02 — The neural models were trained on one assembled dataset

- **Severity:** Major.
- **Criticism:** All neural optimization used BRISC, which aggregates upstream sources and can encode source-specific label cues. A high internal classifier score may reflect framing, contrast, compression, or acquisition differences rather than tumor biology.
- **Evidence:** [`audit/classifier_training_findings.json`](classifier_training_findings.json) reports a geometry-only AUROC of 0.8190; [`docs/report.md`](../docs/report.md) documents source confounding.
- **Consequence:** The classifier result is a BRISC task result, not a screening result. The ensemble may share source biases across components.
- **Disposition:** **Acknowledge and document.** Fixing it requires prospective multi-site training or external model development, which would change the frozen stack.
- **Action:** Preserve the geometry diagnostic and source-shortcut warning in the main report and reviewer package.
- **Residual risk:** Other unmeasured shortcuts can remain even if geometry is controlled.

### S10-M03 — Independent spatial transport failed badly

- **Severity:** Fatal for claims of external segmentation utility.
- **Criticism:** Strict subject-macro Dice was 0.1272 on OpenNeuro pituitary and 0.2208 on UPENN-GBM. These are not small degradations; they demonstrate that the frozen segmenter is not transportable to the evaluated independent cohorts.
- **Evidence:** [`experiments/x02_openneuro_pituitary/result.json`](../experiments/x02_openneuro_pituitary/result.json); [`experiments/x03_upenn_gbm/result.json`](../experiments/x03_upenn_gbm/result.json); [`audit/stage6_cross_dataset.md`](stage6_cross_dataset.md).
- **Consequence:** H8 is rejected. Internal E08 Dice cannot justify deployment outside BRISC-like inputs.
- **Disposition:** **Acknowledge and foreground.** No documentation-only fix can change this result.
- **Action:** State external failure before discussing OOD mitigation. Avoid language implying that OOD detection repairs segmentation.
- **Residual risk:** Other external sources may perform better or worse; only new frozen evaluations can determine that.

### S10-M04 — External cohorts are incomplete and not symmetric across categories

- **Severity:** Major.
- **Criticism:** OpenNeuro covers pituitary, UPENN is GBM-only with a coarse glioma mapping, Cheng overlaps BRISC, and Meningioma-SEG-CLASS was access blocked. There is no independent meningioma result and no representative no-tumor screening cohort.
- **Evidence:** [`audit/stage6_cross_dataset.md`](stage6_cross_dataset.md).
- **Consequence:** The project cannot make a four-class external-performance claim, estimate specificity on independent no-tumor cases, or claim independent meningioma transport.
- **Disposition:** **Acknowledge and document.**
- **Action:** Keep every external metric cohort-specific and prohibit pooled external summaries.
- **Residual risk:** Category conclusions can be confounded with dataset and acquisition differences.

### S10-M05 — The reliability threshold is fragile and cost-dependent

- **Severity:** Major.
- **Criticism:** `0.9490250173235835` minimizes the registered internal cost, but the cost surface is fragile. Worst cost rises from 205 to 628 within ±0.05. The cost ratio 10:1 is an engineering assumption, not a clinical utility estimate.
- **Evidence:** [`experiments/p02_threshold_selection/result.json`](../experiments/p02_threshold_selection/result.json); [`experiments/p05_threshold_sensitivity/result.json`](../experiments/p05_threshold_sensitivity/result.json).
- **Consequence:** The exact threshold is reproducible but not externally justified. A reviewer may view the many decimal places as false precision.
- **Disposition:** **Acknowledge and document.**
- **Action:** Explain that full precision reproduces the observed optimum; it does not imply universal precision. Any new use must preregister costs and reselect policy.
- **Residual risk:** Policy behavior can change sharply with prevalence or score drift.

### S10-M06 — Reliability ranking transfers, but reliability probability does not

- **Severity:** Major.
- **Criticism:** External failure-detection AUROC remained useful, but ECE rose to 0.2551 on OpenNeuro and 0.3391 on UPENN. Calling this “reliability transfer” without qualification can imply that the probability itself remains honest.
- **Evidence:** [`audit/stage6_cross_dataset.md`](stage6_cross_dataset.md).
- **Consequence:** The internal R02 score and threshold are not validated as external operating probabilities.
- **Disposition:** **Fix wording, acknowledge, and document.**
- **Action:** Use “failure ranking retained discrimination on two evaluated cohorts” rather than an unqualified “reliability transfers externally.”
- **Residual risk:** AUROC can remain high while the chosen operating threshold performs poorly.

### S10-M07 — OOD detection may be source recognition

- **Severity:** Major.
- **Criticism:** O02 dominates O05 and includes image/file statistics. AUROC 0.9957 may primarily distinguish dataset encoding, contrast, crop, or file size. Easy far-OOD sources further increase separability.
- **Evidence:** [`experiments/o05_ensemble_ood/result.json`](../experiments/o05_ensemble_ood/result.json); [`audit/stage7_completion_report.md`](stage7_completion_report.md).
- **Consequence:** H9 supports separation on the registered benchmark, not semantic OOD detection or safety at an unseen hospital.
- **Disposition:** **Acknowledge and document.** A shortcut ablation would be new experimental work and is prohibited in Stage 10.
- **Action:** Present O05 as a source-sensitive protective gate. Recommend prospective site-held-out evaluation and a version excluding encoding features.
- **Residual risk:** A shifted but BRISC-styled image may pass, and a valid new-site MRI may be flagged.

### S10-M08 — Stage 7 near-OOD sources were historically seen by the project

- **Severity:** Moderate.
- **Criticism:** Cheng, OpenNeuro, and UPENN had been analyzed in Stage 6 before OOD work. Stage 7 fitting excluded earlier masks and performance values, but the sources were not historically unseen to the research process.
- **Evidence:** [`audit/stage7_completion_report.md`](stage7_completion_report.md).
- **Consequence:** The benchmark is held out within Stage 7 but not a prospective test of a detector frozen before source discovery.
- **Disposition:** **Acknowledge and document.**
- **Action:** Recommend a new untouched hospital/source after freezing O05.
- **Residual risk:** Design decisions may indirectly reflect prior familiarity with the sources.

### S10-M09 — The 2D 128×128 design discards clinically relevant context

- **Severity:** Major.
- **Criticism:** Single slices and one grayscale channel omit volumetric continuity, multimodal MRI, physical spacing, and neighboring anatomy. The adapter erased five OpenNeuro and 82 UPENN targets.
- **Evidence:** [`experiments/x02_openneuro_pituitary/result.json`](../experiments/x02_openneuro_pituitary/result.json); [`experiments/x03_upenn_gbm/result.json`](../experiments/x03_upenn_gbm/result.json).
- **Consequence:** Small-lesion sensitivity and external failure are partly architectural/data-contract limitations. The system cannot support volume, millimetre diameter, or depth.
- **Disposition:** **Acknowledge and document.**
- **Action:** Recommend a high-resolution or 3D multi-sequence successor with preserved spacing.
- **Residual risk:** Increasing resolution alone may not resolve acquisition and target-definition shift.

### S10-M10 — Small-mask “driver” language risks causal overstatement

- **Severity:** Moderate.
- **Criticism:** Small mask area had the strongest registered association with glioma failure, but this observational subgroup analysis does not prove causation. Size correlates with contrast, location, annotation, and slice selection.
- **Evidence:** [`audit/glioma_subgroup_analysis.md`](glioma_subgroup_analysis.md); [`audit/glioma_failure_findings.md`](glioma_failure_findings.md).
- **Consequence:** “Primary failure driver” can be read causally.
- **Disposition:** **Fix wording and document.**
- **Action:** Prefer “strongest measured failure association in this model and dataset.”
- **Residual risk:** Unmeasured confounding remains.

### S10-M11 — Ensemble selection adds complexity for a modest gain over v1

- **Severity:** Moderate.
- **Criticism:** E08 requires four segmenter checkpoints plus classifier routing. Its gains over v1 are +0.0234 glioma, +0.0119 meningioma, and +0.0096 pituitary. A reviewer may question whether operational complexity is justified.
- **Evidence:** [`experiments/e08_ensemble/result.json`](../experiments/e08_ensemble/result.json); [`runs/final_stack/manifest.json`](../runs/final_stack/manifest.json).
- **Consequence:** More failure points, latency, memory, and routing dependence are introduced without demonstrating deployment benefit.
- **Disposition:** **Acknowledge and document.**
- **Action:** State that E08 satisfied the registered H7 selection rule but was not optimized for latency or maintainability. Provide the exact component roles.
- **Residual risk:** No measured runtime/energy comparison is registered.

### S10-M12 — The classifier-dependent route can fail under shift

- **Severity:** Moderate internally; major externally.
- **Criticism:** The third E08 component routes predicted glioma to E07 and all other predictions to v1. UPENN glioma recall was only 0.4218, so the intended specialist route may often be missed under shift.
- **Evidence:** [`experiments/e08_ensemble/result.json`](../experiments/e08_ensemble/result.json); [`experiments/x03_upenn_gbm/result.json`](../experiments/x03_upenn_gbm/result.json).
- **Consequence:** Segmentation and classification errors are coupled under external shift.
- **Disposition:** **Acknowledge and document.**
- **Action:** Recommend soft routing or route-independent ensembles only as future registered work.
- **Residual risk:** Internal route accuracy does not bound external route accuracy.

### S10-M13 — Calibration regions use reference information for evaluation

- **Severity:** Moderate.
- **Criticism:** Lesion-centered ECE is an analysis metric constructed around the reference lesion. It is appropriate for evaluation but unavailable for a new unlabeled case. Some numerically strong calibration variants were inference-ineligible for this reason.
- **Evidence:** [`experiments/c03_isotonic/result.json`](../experiments/c03_isotonic/result.json); [`audit/stage3_calibration_report.md`](stage3_calibration_report.md).
- **Consequence:** Lesion-centered ECE should not be described as an online measurement.
- **Disposition:** **Document.**
- **Action:** Continue distinguishing evaluation region from the globally applicable C03 mapping.
- **Residual risk:** Whole-image ECE can be deceptively low because of easy background.

### S10-M14 — The C03 operational mapping uses all validation cases

- **Severity:** Moderate.
- **Criticism:** Reported C03 metrics are out of fold, but the frozen operational isotonic map is fitted to all 680 validation cases. It has no untouched internal estimate after this full-population fit.
- **Evidence:** [`experiments/c03_isotonic/result.json`](../experiments/c03_isotonic/result.json).
- **Consequence:** OOF values are valid for method assessment, but the exact serialized operational mapping is not independently evaluated.
- **Disposition:** **Acknowledge and document.**
- **Action:** Preserve the distinction between OOF metrics and operational full-population fit in every summary.
- **Residual risk:** The mapping can overfit fine-grained validation probability structure.

### S10-M15 — The deployable R02 model is also a full-population fit

- **Severity:** Moderate.
- **Criticism:** R02’s reported metrics are OOF, while the serialized scaler/logistic model was deterministically fitted on all 680 C03-calibrated validation cases under Amendment 14.
- **Evidence:** [`audit/preregistration_amendment_14.md`](preregistration_amendment_14.md); [`models/r02_reliability_manifest.json`](../models/r02_reliability_manifest.json).
- **Consequence:** OOF performance should not be attributed to the exact full-population deployment coefficients as an independent evaluation.
- **Disposition:** **Acknowledge and document.**
- **Action:** Retain the operational-versus-evaluation distinction.
- **Residual risk:** A prospective reliability cohort is still required.

### S10-M16 — No clinician adjudication of surprising errors

- **Severity:** Major for clinical interpretation; moderate for engineering claims.
- **Criticism:** Visual reviews checked technical alignment but did not clinically adjudicate whether reference masks or labels were medically correct.
- **Evidence:** [`audit/segmentation_visual_review_guide.md`](segmentation_visual_review_guide.md); [`audit/segmentation_failure_modes.md`](segmentation_failure_modes.md).
- **Consequence:** A “model failure” is failure against the dataset reference, not necessarily against expert consensus.
- **Disposition:** **Acknowledge and document.**
- **Action:** Use “reference-mask disagreement” or “dataset-label error” where appropriate; do not infer pathology.
- **Residual risk:** Label noise can affect every supervised metric.

### S10-M17 — Statistical intervals do not cover the full adaptive workflow

- **Severity:** Moderate.
- **Criticism:** Bootstrap CIs cover resampling of the observed groups or patients, not uncertainty from model selection, amendment-driven analyses, multiple experiments, or choosing hypotheses after earlier findings.
- **Evidence:** [`audit/preregistration.md`](preregistration.md); amendment sequence in [`docs/appendix.md`](../docs/appendix.md).
- **Consequence:** CIs may look more definitive than the full research process warrants.
- **Disposition:** **Acknowledge and document.**
- **Action:** Label intervals as conditional resampling intervals and distinguish confirmatory from descriptive results.
- **Residual risk:** No correction summarizes all selection multiplicity.

## B. Claim criticism

### Claims that would be overclaims

| Potential claim | Reviewer verdict | Required correction |
|---|---|---|
| “The system detects brain tumors with 97.8% accuracy.” | **Reject.** Macro F1 is not accuracy; BRISC internal validation is not clinical screening; patient independence is unverified. | State metric, dataset, split, and limitation together. |
| “The model generalizes to external MRI.” | **False for segmentation.** H8 was rejected on both evaluated independent cohorts. | State cohort-specific failures. |
| “Reliability transfers externally.” | **Too broad.** Ranking AUROC retained discrimination, while calibration and threshold behavior failed to transfer. | Narrow to the two evaluated cohorts and ranking only. |
| “Small masks cause glioma failure.” | **Causal overstatement.** Size is the strongest observed association. | Replace “cause/driver” with “measured association.” |
| “OOD detection makes the system safe.” | **Unsupported.** O05 is a retrospective source-sensitive gate. | Describe mitigation on the registered benchmark only. |
| “C03 fixes confidence.” | **Too broad.** It improves internal pixel calibration and cannot repair spatial errors or external shift. | Name region, split, and limitation. |
| “E08 is clinically accurate.” | **Unsupported.** E08 is an internally selected research ensemble. | Remove clinical language. |

### Claims that are appropriately narrow

- The classifier achieved macro F1 0.9779 on BRISC internal development validation.
- E08 improved all three BRISC internal category means over v1 under the registered evaluation.
- C03 improved internal lesion-centered ECE while preserving the hard mask.
- R02 achieved OOF AUROC 0.9555 on the registered internal failure target.
- H8 was rejected for the evaluated independent cohorts.
- O05 separated the registered benchmark sources with AUROC 0.9957.
- The frozen stack reconstructs registered fixtures and OOD states within the stated tolerance.

### Claims that are currently underclaimed

1. **Negative-result preservation:** The project records failed interventions and does not relabel them as successes. This is stronger than typical undergraduate reporting.
2. **Operational eligibility as a selection criterion:** C04 and C07 were not selected despite favourable numbers because they depended on reference information. This is an important methodological contribution.
3. **Fail-closed audit behavior:** Multiple X03 stops exposed real analysis-code defects and target-representability issues before final metrics were accepted.
4. **Artifact reconstruction discipline:** O03 recovery was gated at 1e-12 and would have stopped the freeze if it failed.
5. **Separation of tasks:** Classification, spatial overlap, pixel calibration, case reliability, OOD status, and display policy are treated as distinct claims.

These are contributions to research engineering and auditability. They should not be reframed as algorithmic novelty.

### Findings that restate prior work

- Small lesions receive unstable overlap scores and are vulnerable to downsampling.
- Glioma segmentation is harder than more circumscribed tumor categories.
- Boundary losses, ensembling, calibration, and uncertainty estimation are standard techniques.
- Internal medical-imaging performance can collapse under domain shift.
- OOD detectors can exploit acquisition and dataset shortcuts.

The project does not claim discovery of these general phenomena. Its contribution is the quantified, registered demonstration within this specific BRISC pipeline and the way those findings determine the frozen policy.

## C. Reproducibility criticisms

### S10-R01 — Neural training is controlled but not bitwise deterministic

- **Severity:** Moderate.
- **Criticism:** Seeds and configs are recorded, but all deterministic GPU algorithms were not globally enforced for the historical neural runs.
- **Fix:** None without retraining, which is prohibited.
- **Response:** **Acknowledge.** Distinguish training reproducibility from frozen inference reproducibility.

### S10-R02 — Windows/CUDA/platform dependence can alter behavior

- **Severity:** Moderate.
- **Criticism:** The frozen environment was verified on Windows with an RTX 4060 Laptop GPU and specific CUDA/PyTorch versions. Future drivers and kernels can produce different floating-point behavior.
- **Fix:** Documentation can add a supported-environment statement; containerization would be future work.
- **Response:** **Document.** Synthetic fixtures and tolerances detect divergence but do not guarantee cross-platform identity.

### S10-R03 — Raw datasets are not fully distributable in the repository

- **Severity:** Major for third-party reproduction.
- **Criticism:** Dataset access, licensing, size, and controlled-access restrictions mean a reviewer cannot clone the repository and reproduce all training/external evaluations immediately.
- **Fix:** Preserve dataset manifests, hashes, acquisition instructions, and access-blocked status. Do not redistribute unauthorized data.
- **Response:** **Acknowledge and document.** X04 is a direct example of access limiting reproduction.

### S10-R04 — Some historical intermediate artifacts are local or ignored

- **Severity:** Moderate.
- **Criticism:** Not every early run payload is tracked in Git. Baseline archives and large artifacts may depend on the local workspace.
- **Fix:** The final stack and registered experiment results are tracked; a future archival release could use object storage with immutable checksums.
- **Response:** **Acknowledge.** Reproduction of the final inference stack is stronger than reproduction of every historical training step.

### S10-R05 — O03 state was reconstructed, not originally serialized

- **Severity:** Moderate.
- **Criticism:** A reviewer may distrust recovery of a missing precision matrix.
- **Fix:** The reconstructed state is now frozen, hashed, and verified against all registered distances with maximum error 0.0.
- **Response:** **Document.** Treat this as recovery with a strict equivalence proof, not evidence that the original serialization was complete.

### S10-R06 — Environment aging

- **Severity:** Moderate.
- **Criticism:** `uv.lock` captures package versions, not permanent availability of wheels, CUDA runtimes, or driver compatibility.
- **Fix:** Future archival work could add a container image digest, software bill of materials, and CPU reference outputs.
- **Response:** **Acknowledge and document.** Current fixtures offer drift detection.

### S10-R07 — The report depends on generated figures and source paths

- **Severity:** Minor.
- **Criticism:** A reviewer needs to know whether figures were manually edited.
- **Fix:** Figure generation is scripted, all inputs and hashes are in `docs/figures/figure_manifest.json`, and images are regenerated at 300 DPI.
- **Response:** **Fixed and documented.**

### S10-R08 — A commit cannot contain its own hash

- **Severity:** Minor.
- **Criticism:** A completion report that claims its own commit hash creates an impossible self-reference.
- **Fix:** Record the reviewed evidence commit in the report, provide a command for resolving the containing commit, and report the final hash outside the commit.
- **Response:** **Document.** This matches earlier repository convention.

## D. Novelty criticism

### Standard components

None of the following is algorithmically novel: ResNet-18 classification, compact U-Net segmentation, weighted BCE/Dice training, boundary loss, isotonic regression, logistic reliability estimation, permutation importance, empirical-CDF normalization, Mahalanobis distance, Ledoit–Wolf covariance, model ensembling, bootstrap confidence intervals, or cost-sensitive thresholding.

The project should not be submitted as a new architecture or state-of-the-art segmentation method. Its E08 Dice is not competitive evidence across incompatible datasets, and direct comparison with BraTS or other 3D multimodal work would be misleading.

### Project-specific contribution

The defensible contribution is an integrated research-engineering case study with these elements:

1. a preregistered sequence of gates with amendments preserved before authorized restarts;
2. explicit rejection of interventions that fail or require unavailable inference information;
3. quantitative glioma failure decomposition with small-target association;
4. separation of classifier correctness from spatial segmentation reliability;
5. internal calibration, reliability, and policy studies with OOF evaluation;
6. independent external evaluation that reports severe transport failure without adapting the stack;
7. a source-sensitive OOD mitigation evaluated after the transport failure;
8. a complete manifest-only frozen package with deterministic reconstruction; and
9. a narrative that treats negative results as first-class evidence.

This contribution is strongest for an undergraduate methods, reproducibility, or responsible-AI workshop. It is weak for a venue demanding clinical novelty or state-of-the-art algorithmic performance.

## E. Documentation-only fixes authorized by this review

The following wording fixes are justified without changing the frozen stack or any registered value:

| ID | Existing risk | Approved documentation fix |
|---|---|---|
| F01 | “Reliability transfers” can imply calibrated probability transfer. | Say that failure ranking retained discrimination on two evaluated independent cohorts, while calibration did not transfer. |
| F02 | “Small-mask driver” can sound causal. | Use “strongest measured failure association in this model and dataset.” |
| F03 | “Contribution” can imply algorithmic novelty. | Describe the contribution as an audited research-engineering evidence chain. |
| F04 | OOD mitigation can sound like a safety solution. | State that O05 mitigates a registered source-shift pathway but does not establish safety. |
| F05 | E08 improvement can obscure added complexity. | Pair selection with the four-checkpoint/routing cost and lack of runtime study. |

No metric, CI, model, calibration map, feature order, threshold, hypothesis verdict, or frozen artifact may change under these fixes.

## F. Questions a hostile reviewer should still ask

1. How much would every internal result change under verified patient-level splits?
2. Does the classifier retain performance when source, geometry, and compression cues are balanced?
3. Can a higher-resolution or three-dimensional model remove small-target failure without increasing false positives?
4. Does R02 retain both discrimination and calibration on an untouched prospective cohort?
5. How does O05 perform when encoding features are removed and sites are prospectively held out?
6. Does E08’s modest Dice gain justify its operational complexity and latency?
7. Would clinician adjudication change the failure taxonomy or reveal reference-mask error?
8. What review and miss costs would real users assign, and how would that change the threshold?

## Final adversarial assessment

The project survives hostile review only under a narrow identity: a rigorously audited undergraduate research prototype and negative-results case study. It does not survive as a clinical tumor detector, a generalizable segmentation system, or an algorithmically novel architecture. Its strongest evidence is the transparency of its failure chain and the reproducibility of its frozen inference package. Its largest unresolved threats are unverified patient independence, severe independent transport failure, source-sensitive OOD discrimination, weak glioma/small-target behavior, and lack of prospective clinical review.
