# Consolidated limitations

This document consolidates limitations that are otherwise distributed across dataset audits, experiment decisions, stage reports, and the frozen-stack manifest. It applies to every result and artifact in this repository.

## Scope statement

The project is a research prototype. It is not a clinical tool or deployable medical device, has not been validated for patient care, has not established generalizability beyond BRISC, and is not FDA approved or CE marked. It must not be used for diagnosis, exclusion of disease, grading, prognosis, treatment, surgical planning, or monitoring.

## Evidence limitations

### BRISC patient independence

Complete patient independence is unverified. Similarity-aware groups reduce obvious duplicate leakage, but they cannot prove that two visually different slices come from different people. Internal classification, segmentation, calibration, and reliability metrics may therefore be optimistic.

### Sealed test set

The BRISC supplied test set remains sealed. This is a process commitment supported by path discipline, scripts, and the review log; it is not cryptographic or operating-system access control. The seal protects the set from intentional iterative development, but the project reports development-validation performance rather than a final test estimate. Opening the supplied test would not by itself solve the missing patient-identity problem.

### Dataset-source confounding

BRISC aggregates multiple upstream sources. Acquisition style, image framing, resolution, compression, contrast, annotations, and source-specific formatting may correlate with labels. The classifier and OOD detector can exploit these correlations.

### External cohort coverage

Only two independent tumor-positive cohorts were evaluated successfully: OpenNeuro pituitary and UPENN-GBM. Meningioma-SEG-CLASS was access blocked. Cheng overlaps BRISC and is only an overlap control. There is no independent no-tumor screening cohort and no independent meningioma estimate.

### Retrospective design

All evaluations are retrospective. There is no prospective enrollment, consecutive-case sampling, reader study, clinical comparator, time-to-decision study, or workflow evaluation.

## Data and label limitations

### Category meaning

Glioma, meningioma, pituitary, and no tumor are dataset categories. They are not a complete taxonomy of intracranial masses. The classifier does not cover metastases, schwannoma, ependymoma, medulloblastoma, craniopharyngioma, inflammatory lesions, vascular lesions, or postoperative change as explicit categories.

### Tumor-only segmentation

The segmentation training/evaluation population contains positive tumor masks. It does not establish false-positive spatial behavior on a representative no-tumor population. Classification and segmentation tasks therefore have different denominators.

### Reference-mask uncertainty

Reference masks are treated as ground truth for overlap metrics, but annotation protocol, inter-reader variability, and clinical adjudication are not fully characterized. A lower Dice can reflect model error, label uncertainty, or target-definition mismatch.

### Physical metadata

Complete verified pixel spacing is not available across BRISC. Millimetre-scale size, volume, depth below skull, and physical distance are unsupported. Pixel measurements must not be presented as clinical dimensions.

## Model limitations

### Two-dimensional context

The system processes one two-dimensional image. It cannot use through-plane continuity, full three-dimensional tumor extent, or relationships across sequences. It may treat adjacent slices inconsistently.

### Single-channel appearance

The frozen pipeline does not jointly model standard multi-sequence MRI such as T1, post-contrast T1, T2, and FLAIR. This limits its ability to distinguish enhancing tumor, non-enhancing tumor, edema, necrosis, and normal structures.

### Low resolution and target representability

The segmenter uses 128×128 model space. Five OpenNeuro targets and 82 UPENN targets disappeared after the frozen nearest-neighbor mask transform. Strict analyses count them as Dice zero. Native-resolution restoration showed that these cases do not explain all external failure, but they establish a small-target limit.

Small target area is the strongest observed association in this model and dataset. Because Dice is mathematically coupled to object size, this is partly a metric artifact and is not a causal claim. R11 found a positive surface-Dice point difference whose CI crossed zero, a clearer HD95 improvement, and almost no lesion-detection sensitivity change. These complementary views narrow the claim rather than establishing a single size-driven mechanism.

### Glioma weakness

Glioma is the weakest segmentation and reliability subgroup. E08 glioma Dice is 0.6324 internally, and small masks have a 65.6% observed failure rate under the registered size definition. Registered loss, specialist, inference, and ensemble experiments did not meet the original parent improvement target.

### Classifier-dependent routing

E08 routes one ensemble component using the classifier prediction. Under distribution shift, category error can select an inappropriate specialist/fallback route. UPENN-GBM classification transport was weak, increasing this risk.

### Calibration cannot repair geometry

C03 changes probabilities monotonically and preserves the hard mask. It cannot move a misplaced region, merge fragments, or recover an empty prediction. Improved ECE must not be described as improved spatial accuracy.

## Reliability limitations

### Internal target

R02 estimates success under the internal definition Dice at least 0.50. This is not probability of clinical acceptability, diagnostic correctness, safe treatment, or accurate tumor burden.

### Feature associations

R02 coefficients and permutation importance describe association. Features are correlated, and coefficients are not causal. Boundary entropy being dominant does not prove that boundary entropy causes failure.

### Subgroup imbalance

Some categories have few failures, especially meningioma. Per-category reliability models were unstable and degraded ranking. Aggregate AUROC hides subgroup differences; glioma is weaker.

### External calibration

OpenNeuro reliability AUROC was 0.8719 with ECE 0.2551, and UPENN reliability AUROC was 0.8316 with ECE 0.3391. Ranking transferred better than segmentation Dice, but probability honesty did not. R12 failed its conjunctive requirement that R02 beat both BRISC-fitted simple predictors by at least +0.03 AUROC on both cohorts. The internal threshold is not validated externally.

### Joint policy coverage

P06 sent every X02 and X03 case to review, giving zero external standard-display coverage. It caught all observed external failures through total abstention. This does not show that the policy can selectively accept useful external masks, and it is not evidence of clinical safety.

### Policy cost assumptions

The selected reliability threshold follows costs TP=1, FP=1, TN=0, FN=10 in the registered framing. These values are engineering assumptions, not measured clinical utilities. A different setting can select a different threshold.

### Threshold fragility

The threshold `0.9490250173235835` is fragile. Worst cost within ±0.05 is 628 compared with selected cost 205. It must be stored at full precision for reproduction and re-evaluated for any new population.

## OOD limitations

### Benchmark specificity

O05 AUROC 0.9957 applies to the registered retrospective mixture of BRISC, three near-OOD sources, and four far-OOD sources. It does not cover all scanners, hospitals, artifacts, orientations, file corruptions, or adversarial inputs.

### Easy far-OOD sources

CIFAR-10, OrganAMNIST, random noise, and uniform gray are intentionally easy shifts. Strong performance on them can inflate an overall impression of OOD capability.

### Source shortcuts

O02 image statistics dominate the selected logistic ensemble. File size, encoding, contrast, crop, and dataset style can separate sources. The detector may flag valid MR images from a new site or accept a shifted image that mimics BRISC formatting.

### Historical exposure

Near-OOD datasets were analyzed in Stage 6 before the Stage 7 detector study, although their performance values and masks were excluded from detector fitting. The benchmark is held out within Stage 7 but is not historically unseen by the project.

### False flags and no REFUSE state

O06 flags 8.82% of BRISC validation images. REFUSE is disabled because its false-refusal rate exceeded the registered limit. OOD review is a research warning, not a safety guarantee.

O07 further shows that O05 is weak as an individual mask-failure detector: external failure AUROC was 0.5998 on OpenNeuro and 0.5638 on UPENN. O05 is supported only as a registered source-shift detector on its retrospective benchmark.

## Statistical limitations

### Confidence intervals

Bootstrap intervals are conditional on the observed dataset and grouping assumptions. They do not include unknown patient duplication, systematic label error, future-domain shift, or uncertainty created by the full sequence of model-selection decisions.

### Multiple experiments

At the external-review cutoff, 49 experiment packages had been created, including 33 numbered E/C/R/O packages. Amendment 21 added four more. Separate gates and a visible registry reduce but do not eliminate multiplicity and researcher degrees of freedom. R02 feature reduction, O05 secondary-ensemble selection, P02 threshold selection, the post-H1 H7 pathway, and the glioma subgroup predictor search are exploratory or development model-selection analyses.

### Missing strong baselines

The study did not run nnU-Net, reproduce a published BRISC baseline, or compare E08 with another strong compute-matched segmentation system. R12 adds two simple reliability baselines, but it does not close the segmentation-baseline gap. Claims concern this frozen project stack rather than superiority over current medical-image segmentation practice.

### No external positive control

No model was fine-tuned on a subset of OpenNeuro or UPENN to show that the registered adapter and target definitions can support high performance under the same evaluation. The external failures therefore combine neural-model weakness with possible protocol, preprocessing, and label-definition mismatch. These causes cannot be fully separated by the current evidence.

### Metric dependence

Dice is sensitive to target size; a small positional error can cause a large decrease for a small mask. AUROC ignores calibration and operating prevalence. ECE depends on binning. Brier skill depends on its baseline. No metric alone establishes usefulness.

### External mapping differences

UPENN-GBM is a GBM-only whole-tumor cohort and maps only coarsely to heterogeneous BRISC glioma. OpenNeuro target definition and acquisition differ from BRISC pituitary. Direct numerical comparisons require these caveats.

## Engineering and reproducibility limitations

### Training determinism

Seeds, config, and data order were controlled, but the original neural training did not enforce every deterministic GPU operation. Exact training reruns are not guaranteed bitwise identical.

### Frozen-environment aging

The final stack includes project locks and code, but future drivers, hardware, operating systems, and package availability may change execution behavior. Hash verification detects artifact changes, not every platform difference.

### O03 reconstruction

O03 state was reconstructed exactly from frozen embeddings rather than copied from an originally serialized estimator. The 1e-12 reproduction gate passed, making it a valid deterministic recovery; earlier serialization should still have been more complete.

### Application not yet integrated

The final stack is an inference package and policy contract. Stage 9 is documentation. A polished end-user application, security model, upload validation, access control, logging, monitoring, and update mechanism remain future work.

## What would change these conclusions

- A new patient-independent cohort passing the registered transport gates would revise H8’s current rejection scope.
- Prospective multi-site results with stable Dice, calibration, reliability, and OOD policy would strengthen generalization claims.
- Reliable BRISC patient identifiers could remove or confirm the internal leakage concern.
- A controlled high-resolution or 3D model eliminating the small-mask risk difference would revise the identified failure mechanism.
- A preregistered study showing calibrated reliability probabilities externally would change the present ranking-only interpretation.
- An unseen-site study showing O05 detects subtle shifts without relying on encoding would reduce the shortcut concern.
- A clinician-led prospective study would be required before any patient-care claim could be reconsidered.

Until such evidence exists, the binding statement is: **Research prototype — not for clinical use.**
