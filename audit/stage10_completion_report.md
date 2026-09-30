# Stage 10 adversarial-review completion report

**Status:** COMPLETE  
**Evidence commit reviewed:** `940a4b57e086ad4f4375cdbc73b82d53025e2693`  
**Review type:** simulated hostile self-review and external-review preparation  
**External-review status:** no external review is claimed; the review package is ready for the user to send manually.

## Scope completed

Stage 10 produced three documentation-only outputs:

1. **Phase A — hostile self-review:** [`stage10_self_review.md`](stage10_self_review.md) evaluates methodology, claims, reproducibility, and novelty from the perspective of a skeptical workshop reviewer.
2. **Phase B — review package:** [`stage10_review_package.md`](stage10_review_package.md) gives Claude, Grok, Gemini, or another independent reviewer a self-contained summary, limitations, hypotheses, evidence paths, severity scale, and seven required questions.
3. **Phase C — consolidation:** this report records the criticisms, fixes, unresolved limitations, verification results, and completion state.

No model, calibration mapping, reliability estimator, OOD estimator, threshold, metric, confidence interval, hypothesis verdict, experiment result, or frozen-stack artifact was changed.

## Criticisms generated

The simulated hostile review recorded **25 formal criticisms**:

| Class | Fatal | Major | Moderate | Minor | Total |
|---|---:|---:|---:|---:|---:|
| Methodology | 2 | 8 | 7 | 0 | 17 |
| Reproducibility | 0 | 1 | 5 | 2 | 8 |
| **Total** | **2** | **9** | **12** | **2** | **25** |

The two fatal ratings apply to broad clinical or generalization claims, not to the narrow identity of an undergraduate research-engineering case study:

- unverified patient independence defeats a claim of unseen-patient internal performance; and
- severe independent spatial transport failure defeats a claim of general segmentation performance.

The principal major criticisms are single-dataset neural training, asymmetric external evidence, fragile policy threshold, external reliability miscalibration, source-sensitive OOD discrimination, classifier-dependent ensemble routing, absence of clinician adjudication, and incomplete third-party access to raw datasets.

## Fixed through documentation

Five claim-level issues were fixable without changing evidence:

| ID | Revision made |
|---|---|
| F01 | Replaced broad “reliability transfers” wording with the narrower result that failure ranking retained discrimination on two evaluated independent cohorts while probability calibration did not transfer. |
| F02 | Replaced causal “small-mask driver” wording with “strongest measured failure association in this model and dataset.” |
| F03 | Defined novelty as an audited research-engineering evidence chain rather than a new architecture or state-of-the-art algorithm. |
| F04 | Narrowed the OOD claim to detection of a registered source-shift pathway and stated that it does not establish safety. |
| F05 | Added E08's four-checkpoint, classifier-routing complexity and the absence of a latency, energy, or deployment-burden study. |

The corresponding wording was updated in [`docs/report.md`](../docs/report.md), [`README.md`](../README.md), and the regenerable external-ranking figure. No numeric result changed.

## Acknowledged limitations that remain unresolved

The following cannot be fixed by documentation and remain binding limitations:

- BRISC patient independence is unverified.
- The BRISC sealed test partition remains unopened.
- Neural training used one assembled development dataset.
- Independent segmentation transport failed on OpenNeuro and UPENN-GBM.
- Independent meningioma transport remains unmeasured because X04 access was blocked.
- The 2D 128×128 adapter loses spatial and volumetric context and has a measured small-target limit.
- Glioma remains the weakest segmentation and reliability subgroup.
- C03 and R02 full-population artifacts are operational fits; their reported metrics remain OOF estimates.
- R02 failure ranking retained external discrimination, but its probability calibration and policy threshold did not transfer.
- The reliability threshold is cost-dependent and empirically fragile.
- O05 may exploit source, file, crop, or encoding cues and has not been tested prospectively on an unseen hospital.
- No clinician adjudication, prospective study, reader study, privacy/security evaluation, or clinical validation was performed.
- Historical neural training is controlled but not bitwise reproducible across platforms.
- Raw and controlled-access datasets cannot all be redistributed with the repository.

These limitations are stated in the report, the dedicated limitations document, the hostile self-review, and the external-review package.

## Claim revisions

| Earlier shorthand or risk | Final defensible wording |
|---|---|
| “Reliability transfers externally.” | Failure ranking retained discrimination on X02 and X03; external probability calibration did not transfer. |
| “Small masks drive glioma failure.” | Small target area was the strongest measured failure association in this model and dataset; causality remains untested. |
| “The project contributes an advanced model.” | The contribution is the integrated preregistered, fail-closed, negative-result-preserving evidence chain. |
| “OOD detection mitigates external failure.” | O05 flags a registered source-shift pathway on a retrospective benchmark; it does not establish general shift detection or safety. |
| “E08 is the best model.” | E08 passed the registered selection rule, but requires four checkpoints and routing, and its operational cost was not measured. |

No claim was strengthened as a result of the self-review.

## Novelty verdict

The architecture and component algorithms are standard: ResNet-18, U-Net, Dice/BCE and boundary losses, probability averaging, isotonic regression, logistic regression, Mahalanobis distance, Ledoit-Wolf covariance, bootstrap intervals, and cost-sensitive thresholds.

The defensible contribution is the integration and audit discipline: preregistration with amendments, explicit gates, preserved failures, glioma-specific decomposition, separation of calibration from spatial correctness, external evaluation without adaptation, source-shift safeguards, exact policy contracts, and manifest-only frozen reconstruction. This supports an undergraduate methods, reproducibility, or responsible-AI submission. It does not support a clinical, state-of-the-art, or algorithmic-novelty claim.

## External review preparation

The review package begins with instructions for independent reviewers to:

- read the summary, findings, and limitations;
- answer seven evidence-specific questions;
- identify overclaimed, underclaimed, ambiguous, or misrepresented statements;
- identify methodological flaws missing from the current limitation list;
- rate each issue as fatal, major, moderate, or minor; and
- state what evidence would change the assessment.

It states that this is an undergraduate project built with AI coding agents under user-directed methodology, uses a preregistered amendment trail, and rejects clinical claims. It asks for honest criticism rather than endorsement.

## Verification

Verification was run after the documentation revisions and before completion:

| Check | Result |
|---|---|
| Ruff | PASS |
| Test suite | **207 passed** |
| Stage 9 documentation verifier | PASS; 14 figures, 9,713 words, 32 page-equivalents |
| Frozen-stack verifier | PASS; 31 artifacts and 3 fixtures |
| O03 reconstruction maximum absolute error | `0.0` |
| O05 reconstruction maximum absolute error | `0.0` |
| Frozen Git tree | `439b4fca4b47f12b173c24eaa039b1258f9ddf03` — unchanged |
| Frozen manifest SHA256 | `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a` — unchanged |
| Self-review SHA256 | `416911ab09609f876c5cf862fc7f83df4bbb9afe484ed2d1a3aeeb66026cf78c` |
| Review-package SHA256 | `b716ff148466c829215b4d8f95f74a23349d992b2a99e9392520c0bee2f511e1` |
| Reviewed report SHA256 | `05b75c23e077e085e88b7e5431275e8627ee810d3c7ec23830a71efba2f5c1a9` |

The evidence commit was clean immediately after creation. The commit containing this completion report can be resolved with:

```powershell
git log -1 --format=%H -- audit/stage10_completion_report.md
```

## Constraint confirmation

- BRISC sealed-test accesses during Stage 10: **0**
- Neural retraining events during Stage 10: **0**
- Calibration or reliability refits during Stage 10: **0**
- Segmentation-threshold changes: **0**; remains `0.50`
- Reliability-threshold changes: **0**; remains `0.9490250173235835`
- OOD-threshold changes: **0**; remains `0.36854275476100085`
- Writes under `runs/final_stack/`: **0**
- Gate relaxations: **0**
- External reviews claimed: **0**

## Completion decision

Stage 10 complete. Ready for Phase 6.

The next action is external review by the user-selected reviewers. Their criticism must be treated as new evidence and must not be described as completed until it is actually received.
