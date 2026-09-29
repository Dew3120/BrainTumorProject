# Phase 6 OOD workstream — repository Stage 7 completion report

**Completed:** 2026-09-30  
**Governance:** Amendment 20 / H9  
**Source commit before completion-report commit:** `ee8109a`  
**Completion commit:** resolve with `git log -1 --format=%H -- audit/stage7_completion_report.md` (a Git commit cannot contain its own hash)

## Outcome

H9 is **accepted on the registered retrospective benchmark**. The selected O05 logistic ensemble achieved held-out AUROC **0.995718** (95% group-bootstrap CI **0.992726–0.997975**) against the registered `0.92` gate.

This acceptance means the detector separated BRISC-like images from the specific registered near- and far-OOD sources. It does not establish clinical safety, universal OOD recognition, or performance at an unseen hospital.

## Experiment status

| ID | Method | Registered gate | Result | Status |
|---|---|---:|---:|---|
| O01 | Frozen OOD benchmark | Descriptive | 2,680 images; 1,300 fit / 1,380 evaluation | Complete / accepted |
| O02 | Image-statistics logistic regression | AUROC ≥ 0.85 | **0.996828** (CI 0.993411–0.999291) | **Pass** |
| O03 | Frozen segmenter bottleneck Mahalanobis | AUROC ≥ 0.90 | **0.915418** (CI 0.897710–0.932144) | **Pass** |
| O04a | Classifier `1-max softmax` | AUROC ≥ 0.85 | **0.849817** (CI 0.824121–0.872436) | **Fail** |
| O04b | Segmenter mean entropy | AUROC ≥ 0.85 | **0.774992** (CI 0.745421–0.802056) | **Fail** |
| O05 primary | Equal mean of four normalized scores | AUROC ≥ 0.92 for selected method | **0.971169** (CI 0.961085–0.980056) | Candidate passed |
| O05 secondary | Logistic combination of four normalized scores | AUROC ≥ 0.92 for selected method | **0.995718** (CI 0.992726–0.997975) | **Selected / pass** |
| O06 | Threshold and application policy | Descriptive | ACCEPT/REVIEW policy | Complete / accepted |

The O05 secondary model was selected by the preregistered rule: improvement over the primary was **+0.024549**, and the paired group-bootstrap CI **0.016908–0.033085** was entirely above zero and exceeded the required `+0.01` improvement.

## Benchmark

- ID fitting: 600 BRISC development-training images, sampled across distinct similarity groups.
- ID evaluation: all 680 positive BRISC segmentation-development-validation images.
- Near OOD: 100 fitting and 100 evaluation images each from Cheng/Figshare, OpenNeuro ds006248, and UPENN-GBM, with patient-disjoint detector splits.
- Far OOD: 100 fitting and 100 evaluation images each from OrganAMNIST, CIFAR-10, seeded random noise, and seeded uniform gray.
- Every row records source provenance, source/group identifier, SHA256, and standardized-pixel SHA256.
- Cross-domain and fit/evaluation standardized-pixel contamination: **0**.
- Two repeated standardized hashes occurred only inside the BRISC evaluation partition and were retained as an audited within-source property.

The near-OOD cohorts had already been processed during Stage 6. No Stage-6 performance value, reference mask, or diagnostic label entered OOD selection or fitting. They are held out from OOD detector fitting, but they are not historically unseen by the project.

## Selected detector

The selected O05 detector combines four inference-time scores:

1. O02 image-statistics probability
2. O03 frozen-segmenter bottleneck Mahalanobis distance
3. Classifier `1 - max softmax`
4. Segmenter whole-image mean Bernoulli entropy

Each score is normalized by its empirical CDF on the 600 BRISC fitting images. A fixed L2 logistic regression combines the four values. O02 dominates the fitted combination; file encoding/size and image statistics are strong source cues. This is effective on the benchmark but may partly detect dataset acquisition and storage differences rather than all clinically relevant shifts.

## Operating policy

The numeric boundaries were computed only from BRISC detector-fitting scores:

- **ACCEPT:** OOD score below `0.36854275476100085`.
- **REVIEW:** OOD score at or above `0.36854275476100085`.
- Invalid, missing, non-finite, or out-of-range scores fail closed to **REVIEW**.
- The 99th-percentile REFUSE candidate was `0.8615167211351659`, but REFUSE is **disabled**. It had OOD precision `0.977941` but falsely refused `2.2059%` of BRISC validation images, above the fixed `1%` limit.

Held-out coverage:

- BRISC validation falsely flagged: **60/680 (8.82%)**.
- OOD evaluation images detected: **699/700 (99.86%)**.
- Cheng: 99/100; all six other OOD sources: 100/100 each.

The required user warning is:

> This input does not match the model's training distribution. Predictions may be unreliable.
>
> **Research prototype — not for clinical use**

## Constraints and verification

- BRISC sealed-test access: **0**.
- Neural retraining or refitting: **0**.
- Reference-mask or diagnostic-label detector features: **0**.
- External performance values used for detector fitting: **0**.
- Segmentation threshold changes: **0**; remains `0.50`.
- Reliability-policy threshold changes: **0**; remains exactly `0.9490250173235835`.
- Gate relaxations: **0**.
- O01–O06 are separately registered, executed, documented, reviewed, committed, and pushed.
- Independent artifact verification: **PASS**.
- Tests: **204 passed**.
- Ruff: **PASS**.
- Working tree at final verification: expected clean after the completion-report commit.

## Limits that remain

1. O02 can exploit file encoding and acquisition-source shortcuts, including file size.
2. Far-OOD sources are intentionally easy and do not cover all invalid medical inputs.
3. The BRISC ID evaluation cohort contains positive tumor slices; it is not a complete prospective upload distribution.
4. Near-OOD images come from cohorts already used in Stage 6 analysis, although their prior results were excluded from detector fitting.
5. Threshold coverage is retrospective and must be re-evaluated on new sites and scanners before any broader claim.
6. The policy is a protective research gate, not a clinical safety guarantee.

**Stage 7 complete. Ready for Stage 8.**
