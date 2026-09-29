# Stage 6 cross-dataset transport

**H8 verdict:** `REJECTED_FOR_EVALUATED_INDEPENDENT_COHORTS`

The frozen E08 + C03 + R02 stack was evaluated without retraining, refitting, threshold adjustment, or external-data-driven selection. X02 and X03 are independent retrospective cohorts and both failed the registered conjunctive transport gates. X01 is informative overlap-control evidence but cannot establish independent generalization. X04 is access-blocked and contributes no performance metric.

## Dataset disposition

| ID | Dataset | Evidence role | Outcome |
|---|---|---|---|
| X01 | Cheng/Figshare v8 | BRISC source-overlap control | Completed; gates failed; not independent |
| X02 | OpenNeuro ds006248 v1.0.0 | Independent pituitary cohort | Completed; H8 rejected |
| X03 | UPENN-GBM registered derivative | Independent GBM-only cohort | Completed; H8 rejected |
| X04 | Meningioma-SEG-CLASS v1 | Controlled-access meningioma candidate | `ACCESS_BLOCKED`; metrics not estimable |

Glioblastoma is a subtype of glioma, but X03's GBM-only population and whole-tumor target are a coarse mapping and are not directly comparable to the heterogeneous BRISC glioma category.

### Disk ledger

| Checkpoint | Drive E free bytes |
|---|---:|
| After X01 audit | 348,887,539,712 |
| After X02 audit | 348,584,640,512 |
| After X03 audit | 348,143,616,000 |
| X04 access decision | 348,132,126,720 |
| Final Stage 6 synthesis | 348,131,704,832 |

The final free space is `324.223 GiB`. The projected footprint never exceeded the registered 250 GB sequential-processing trigger.

## Classification transport

| Cohort | Target-category recall | 95% CI | Top-label ECE | Gate outcome |
|---|---:|---:|---:|---|
| X01 overlap control | 0.9326 (weakest category) | See category-specific X01 artifact | 0.0428 | Recall/ECE pass |
| X02 pituitary | 0.9400 | 0.8600–1.0000 | 0.0762 | Recall pass; ECE fail |
| X03 GBM→glioma | 0.4218 | 0.3469–0.5102 | 0.2454 | Recall/ECE fail |
| X04 meningioma | Not estimable | — | — | Access blocked |

X02 shows that category recognition can remain high while spatial transport fails. X03 shows simultaneous category and spatial domain failure. Tumor/no-tumor AUROC and specificity are not estimable on the tumor-only independent cohorts.

## Segmentation transport

| Cohort/view | Subject-macro Dice | 95% CI | Registered gate |
|---|---:|---:|---|
| X01 glioma | 0.4142 | 0.3656–0.4616 | Fail (≥0.55) |
| X01 meningioma | 0.8679 | 0.8441–0.8887 | Pass (≥0.75) |
| X01 pituitary | 0.7712 | 0.7494–0.7924 | Pass (≥0.75) |
| X02 pituitary, strict 425-slice | 0.1272 | 0.0824–0.1778 | Fail (≥0.75) |
| X03 GBM→glioma, strict 9,464-slice | 0.2208 | 0.2004–0.2422 | Fail (≥0.55) |
| X04 meningioma | Not estimable | — | Access blocked |

The strict and native-resolution views agree closely for X02 (`0.1272` vs `0.1273`) and X03 (`0.2208` vs `0.2205`). The five X02 and 82 X03 targets erased by the frozen 128×128 mask transform therefore do not explain the broad external failures.

## Calibration and reliability transfer

| Cohort | Lesion ECE raw→C03 | Reliability AUROC (95% CI) | Reliability ECE | Brier skill (95% CI) | Top-20% capture |
|---|---:|---:|---:|---:|---:|
| X01 overlap | 0.0802→0.0514 | 0.9387 (0.9244–0.9510) | 0.0613 | 0.5874 (0.5397–0.6273) | 64.7% |
| X02 pituitary | 0.1348→0.1414 | 0.8719 (0.8112–0.9162) | 0.2551 | 0.7050 (0.6587–0.7514) | 23.8% |
| X03 GBM→glioma | 0.1430→0.1420 | 0.8316 (0.8120–0.8501) | 0.3391 | 0.5715 (0.5453–0.5970) | 24.8% |

R02 preserves useful failure ranking on all three evaluated cohorts, but its absolute reliability probabilities are not calibrated externally. The fixed policy review rate rises to `90.6%` on X02 and `82.5%` on X03. The Stage 5 threshold is therefore not validated as an external operating policy.

## Failure taxonomy

| Cohort | Weak cases | Complete miss | Severe under | Fragmented | Disconnected FP |
|---|---:|---:|---:|---:|---:|
| X01 overlap | 817 | 41.1% | 40.0% | 48.0% | 38.4% |
| X02 pituitary | 357 | 86.6% | 9.0% | 42.9% | 41.2% |
| X03 GBM→glioma | 7,600 | 56.1% | 35.5% | 52.0% | 45.6% |

Complete spatial misses dominate X02 and X03. X03 also shows major fragmentation and disconnected false positives. Deterministic visual audits found no shared flip, transpose, shift, or mask-alignment defect, supporting model/domain failure rather than a common rendering error.

## Domain gap

Frozen-encoder centroid shift from the matched BRISC development reference is `4.1616` for X01, `18.7692` for X02, and `13.1125` for X03. The fixed policy review rate increases from the matched BRISC reference to `39.3%` on X01, `90.6%` on X02, and `82.5%` on X03. X03 references are fragmented in `52.8%` of slices versus `4.9%` in its BRISC glioma reference. These shifts are consistent with the poor independent segmentation transport.

No pooled cross-dataset performance number is reported. The cohorts differ in evidence independence, tumor category, orientation, source format, patient population, and target construction; pooling them would conceal the registered failures.

## What Stage 6 establishes

- The exact frozen pipeline can execute on audited external MRI formats with patient-cluster uncertainty estimates and reproducible failure taxonomy.
- Classification transport is dataset-dependent: strong for X02 pituitary, poor for X03 GBM→glioma.
- Segmentation does not satisfy the registered independent transport gates on X02 or X03.
- R02 transfers as a ranking signal but not as an honestly calibrated external probability or validated policy threshold.
- The controlled-access meningioma cohort remains unevaluated and explicitly access-blocked.

## What Stage 6 does not establish

- It does not establish prospective generalization, clinical validity, diagnostic safety, treatment utility, or suitability for patient care.
- It does not establish no-tumor screening performance, because the independent cohorts contain only tumor cases.
- It does not establish independent meningioma transport; X01 overlaps BRISC and X04 was inaccessible.
- It does not justify adapting the model, C03 mapping, R02 estimator, segmentation threshold, or policy threshold using these external outcomes.
- It does not open or replace the sealed BRISC test evaluation.

## Governance verification

- BRISC sealed-test images opened: 0
- Neural retraining: 0; calibrator refits: 0; reliability refits: 0
- Segmentation threshold changes: 0 (`0.50` throughout)
- Reliability-policy threshold changes: 0 (`0.9490250173235835` throughout)
- Gate relaxations: 0
- Confidence intervals: 2,000 patient-cluster bootstrap resamples, seed 42, where estimable
- Claim scope: **retrospective research evidence; not clinical validation**
