# X01 Cheng/Figshare overlap-control evaluation

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_OVERLAP_CONTROL_ONLY`  
**Audit role:** `ELIGIBLE_OVERLAP_CONTROL`  
**Independent external evidence:** No  
**Visual technical review:** Pass; no systematic alignment or rendering defect

## Population and scope

The frozen E08 ensemble, operational C03 isotonic mapping, R02 reliability model, and exact Stage 5 policy were applied once to 3,064 Cheng CE-T1 slices from 233 patient identifiers. Patient identifiers are the uncertainty-resampling unit. Cheng is a documented upstream source of BRISC, so these results are an overlap-control reproduction and cannot establish external generalization.

The pHash audit found 135 Cheng slices whose nearest exact candidate in frozen BRISC metadata is in the original test split, and 686 whose nearest candidate is in that split at any recorded distance. No BRISC test file was opened. This content-overlap warning is another reason X01 is not sealed-test evidence.

## Classifier

Patient-level accuracy was `0.9742` (95% patient-bootstrap CI `0.9528–0.9914`). Top-label ECE was `0.0428`. Recall was glioma `0.9326`, meningioma `1.0000`, and pituitary `1.0000`. All classifier reference gates passed. Tumor/no-tumor specificity and AUROC are not estimable because Cheng contains no no-tumor cases.

## Segmentation

| Category | Patients | Subject-macro Dice | 95% patient-bootstrap CI | Registered gate | Outcome |
|---|---:|---:|---:|---:|---|
| Glioma | 89 | 0.4142 | 0.3656–0.4616 | ≥0.55 | **Fail** |
| Meningioma | 82 | 0.8679 | 0.8441–0.8887 | ≥0.75 | Pass |
| Pituitary | 62 | 0.7712 | 0.7494–0.7924 | ≥0.75 | Pass |

Overall subject-macro Dice was `0.6688`. There were `817` Dice-below-0.50 slices and `336` zero-Dice slices. Whole-volume Dice, tumor volume, and depth are `NOT_ESTIMABLE` because per-slice orientation and ordering are unavailable.

## Calibration and reliability

C03 improved lesion-centred pixel NLL from `0.5391` to `0.3432`, Brier from `0.1000` to `0.0898`, and ECE from `0.0802` to `0.0514`. Whole-image ECE after C03 was `0.0021`.

R02 retained strong failure ranking: AUROC `0.9387` (95% patient-cluster bootstrap CI `0.9244–0.9510`). Its ECE was `0.0613`, above the registered `0.05` gate, so reliability calibration failed even though AUROC and Brier skill `0.5874` passed. The full-precision policy sent `1205` slices (`39.3%`) to review, caught `742` failures, and missed `75`.

## Failure taxonomy

Of the `817` weak slices, `336` (`41.1%`) were complete misses, `327` (`40.0%`) were severe under-predictions, and `392` (`48.0%`) were fragmented. Glioma contributed `737` weak slices. The visual panel confirms these are model/task failures rather than a shared transform defect.

## Domain gap

Relative to BRISC development validation, the frozen classifier encoder centroid shifted by `4.1616` and the mean per-dimension Wasserstein distance was `0.1703`. R02 score Wasserstein distance was `0.0853`. Policy review rate rose from `28.8%` on the BRISC operational reference to `39.3%` on Cheng.

## What X01 establishes

- The frozen pipeline can be transported technically to the official Cheng v8 MATLAB release without retraining, refitting, threshold changes, or geometry failures.
- Classifier category recognition transported strongly within this overlap-control cohort.
- Meningioma and pituitary segmentation passed the registered reference gates.
- Glioma segmentation did not pass, and reliability scores ranked failures but were miscalibrated under the registered ECE criterion.

## What X01 does not establish

- It does not establish independent cross-dataset generalization because Cheng is an upstream BRISC source.
- It is not a sealed BRISC test evaluation; no BRISC test file was opened and no sealed-test claim is made.
- It does not establish performance on no-tumor scans, prospective patients, other scanners, or 3D volumes.
- It does not establish clinical validity, diagnostic accuracy, safety, or suitability for patient care.

## Verification

- Independent per-case recomputation maximum difference: `2.220446049250313e-16`.
- Bootstrap: 2,000 patient-cluster resamples, seed 42.
- Segmentation threshold: exactly `0.50`; reliability-policy threshold: exactly `0.9490250173235835`.
- BRISC development-validation files opened for registered domain reference: 680; sealed-test files opened: 0.
- Neural retraining, calibrator refit, reliability refit, and threshold changes: 0.
