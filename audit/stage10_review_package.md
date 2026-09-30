# Stage 10 external adversarial review package

## How to use this package

This package is designed for the user to send manually to independent AI reviewers such as Claude, Grok, or Gemini. It does not claim that any external review has already occurred.

### Reviewer instructions

1. Read the **project summary**, **key findings**, and **known limitations** below.
2. Answer all seven questions in **Recommended questions for external reviewers**.
3. Identify any claim that is overclaimed, underclaimed, ambiguous, or misrepresented.
4. Identify methodological flaws that are missing from the known-limitations section.
5. Assign each criticism a severity: **fatal**, **major**, **moderate**, or **minor**.
6. Cite the exact statement or result you are criticizing.
7. State what evidence or experiment would change your assessment.
8. Separate criticism of the **research method** from criticism of the **clinical applicability**, because the project explicitly rejects clinical claims.

### Context for reviewers

- This is an **undergraduate research project**.
- It was built with **AI coding agents under user-directed methodology and approval**.
- Experiments were **preregistered**, with an append-only amendment and review trail.
- The project **explicitly rejects clinical claims**. It is not a medical device, is not validated for patient care, and is not FDA approved or CE marked.
- The BRISC sealed test partition was not accessed.
- The final model stack is frozen and hash verified.
- The aim of this review is not praise. We want **honest criticism, specific findings, and questions the project has not asked**.

### Requested response format

For each issue, return:

```text
ID:
Severity: fatal | major | moderate | minor
Claim or method criticized:
Why it matters:
Evidence from this package:
Recommended correction or experiment:
What would change your assessment:
```

Finish with:

- a recommendation for an undergraduate workshop submission;
- a separate recommendation if the same work were presented as a clinical-AI paper;
- the three strongest contributions;
- the three most serious unresolved weaknesses; and
- any important question that this package did not ask.

## One-page project summary

This repository implements a research pipeline for two-dimensional brain MRI classification and tumor-mask segmentation using BRISC. The frozen stack contains a four-class ResNet-18 classifier, an E08 ensemble of compact 2D U-Nets, C03 isotonic pixel calibration, an R02 ten-feature logistic case-reliability model, an O05 out-of-distribution detector, and fixed review policies. The segmentation threshold is `0.50`, the reliability threshold is `0.9490250173235835`, and the OOD review threshold is `0.36854275476100085`.

The classifier predicts glioma, meningioma, pituitary, or no tumor. On BRISC internal development validation it achieved macro F1 0.9779. Patient independence is unverified, and a geometry-only diagnostic reached AUROC 0.8190, so this score may include dataset-source shortcuts. It must not be interpreted as clinical tumor-screening performance.

The historical v1 segmenter achieved image-mean Dice 0.7694 internally, with a large category gap: glioma 0.6090, meningioma 0.8702, and pituitary 0.8088. The selected E08 ensemble improved the category means to 0.6324, 0.8822, and 0.8183. E08 did not satisfy the original H1 parent improvement target but later became final under preregistered H7 because its selected calibration and reliability conditions passed.

C03 isotonic calibration reduced internal lesion-centered ECE from 0.0505 to 0.0335 while preserving the hard mask. R02 achieved out-of-fold failure-detection AUROC 0.9555 (95% CI 0.9387–0.9704), ECE 0.0171, and Brier skill 0.5033. Glioma remained the weakest reliability subgroup. A registered cost model selected the high reliability threshold above; it reduced observed internal cost from 406 at a conventional 0.50 threshold to 205. Sensitivity analysis classified that threshold as fragile.

Independent spatial transport failed. OpenNeuro pituitary subject-macro Dice was 0.1272 (CI 0.0824–0.1778); UPENN-GBM subject-macro Dice was 0.2208 (CI 0.2004–0.2422). H8 was rejected. R02 failure ranking retained discrimination on those cohorts—AUROC 0.8719 and 0.8316—but its probabilities were poorly calibrated externally, so neither the internal probability interpretation nor threshold transferred.

O05 achieved AUROC 0.9957 (CI 0.9927–0.9980) on a registered retrospective ID/OOD benchmark and flagged 699/700 OOD evaluation images. It also flagged 60/680 BRISC validation images. Its dominant component uses image and encoding statistics, so the detector may distinguish dataset sources rather than every medically relevant shift. It is a protective research gate, not a safety guarantee.

The project’s main contribution is not a new architecture. It is a preregistered, failure-aware evidence chain: failed experiments remain failed; external transport failure is reported; operationally ineligible calibration methods are excluded; model/reliability/OOD claims are separated; and the final stack reconstructs from a hash-verified manifest. The repository explicitly states: **Research prototype — not for clinical use.**

## Frozen system under review

| Role | Selected component | Reviewer concern |
|---|---|---|
| Classifier | v1 ResNet-18, raw softmax | Source shortcuts; patient independence |
| Segmenter | E08 = E05 + E06 + classifier-routed E07/v1 | Four checkpoints, routing dependence, modest internal gain |
| Pixel calibrator | C03 isotonic mapping | Internal-only mapping; cannot repair geometry |
| Reliability | R02 ten-feature L2 logistic | Internal target; externally miscalibrated |
| OOD | O05 secondary logistic ensemble | O02 source/encoding dominance |
| OOD policy | O06 ACCEPT/REVIEW | Retrospective benchmark; REFUSE disabled |

Frozen manifest: [`runs/final_stack/manifest.json`](../runs/final_stack/manifest.json), SHA256 `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`.

## Key findings with uncertainty

| Finding | Estimate | 95% CI | Interpretation | Source |
|---|---:|---:|---|---|
| Classifier macro F1 | 0.9779 | Not registered | BRISC internal development validation; patient independence unverified | [`classifier_training_findings.json`](classifier_training_findings.json) |
| V1 image-mean Dice | 0.7694 | Not registered | Historical internal baseline | [`segmentation_training_findings.json`](segmentation_training_findings.json) |
| V1 glioma Dice | 0.6090 | Not registered | Systematic internal weakness | Same source |
| E08 glioma Dice | 0.6324 | Not registered for level | Selected ensemble internal value | [`e08_ensemble/result.json`](../experiments/e08_ensemble/result.json) |
| E08 glioma gain over best eligible single model | +0.0522 | 0.0322–0.0744 | Passed local E08 gate | Same source |
| R02 failure AUROC | 0.9555 | 0.9387–0.9704 | Similarity-aware OOF internal ranking | [`r02_reduced_features/result.json`](../experiments/r02_reduced_features/result.json) |
| OpenNeuro Dice | 0.1272 | 0.0824–0.1778 | Independent spatial transport failed | [`x02 result`](../experiments/x02_openneuro_pituitary/result.json) |
| OpenNeuro reliability AUROC | 0.8719 | 0.8112–0.9162 | Ranking retained; ECE 0.2551 | Same source |
| UPENN-GBM Dice | 0.2208 | 0.2004–0.2422 | Independent coarse GBM→glioma transport failed | [`x03 result`](../experiments/x03_upenn_gbm/result.json) |
| UPENN reliability AUROC | 0.8316 | 0.8120–0.8501 | Ranking retained; ECE 0.3391 | Same source |
| O05 OOD AUROC | 0.9957 | 0.9927–0.9980 | Registered retrospective source benchmark | [`o05 result`](../experiments/o05_ensemble_ood/result.json) |

No new confidence intervals were computed for this package. “Not registered” is reported instead of inventing an interval after the fact.

## Principal failure evidence

### Glioma and small targets

- V1 glioma Dice was 0.6090; E08 improved it to 0.6324.
- Among glioma references smaller than 100 model-space pixels, 21/32 failed Dice 0.50.
- The registered failure-risk difference for small versus larger masks was +0.430 (CI +0.258 to +0.592).
- Size was the strongest measured association; this does not establish a causal mechanism.
- Boundary class, focality, and central/peripheral location did not independently clear the registered predictor rule.

Source: [`audit/glioma_failure_findings.md`](glioma_failure_findings.md).

### Segmentation topology

Among 83 weak v1 cases, 70 were fragmented and 67 contained disconnected false positives. Seventeen were complete misses. Seventy-eight of the 83 weak masks still had correct classification, demonstrating that category correctness does not establish spatial reliability.

Sources: [`segmentation_failure_modes.md`](segmentation_failure_modes.md) and [`classifier_segmenter_agreement.md`](classifier_segmenter_agreement.md).

### Adapter limit

The frozen 128×128 mask transform erased five OpenNeuro targets and 82 UPENN targets. Strict primary metrics assign these targets Dice zero. Native-resolution restored metrics remained similarly poor, so erased targets do not explain the broad external failure.

## Hypotheses and outcomes

| Hypothesis | Registered question | Outcome | Reviewer interpretation |
|---|---|---|---|
| H1 | Can a controlled intervention materially improve glioma while protecting other categories? | **Failed** | No Stage 2 intervention met the +0.05 parent criterion. E08 later passed a separate local/amended selection rule. |
| H7 | Can E08 become final if registered calibration and reliability conditions pass? | **Satisfied** | C03 and R02 passed the amended conditions. This is internal model selection, not external validation. |
| H8 | Does the frozen E08+C03+R02 stack transport to independent external cohorts? | **Rejected for evaluated independent cohorts** | X02 and X03 failed spatial gates. X04 was access blocked. |
| H9 | Can a frozen OOD detector pass its retrospective registered benchmark? | **Satisfied** | O05 passed, but source/encoding shortcuts and historical source exposure constrain the claim. |

## Known limitations

1. **No clinical validation:** no prospective study, radiologist reader study, treatment study, or patient-care evaluation.
2. **Patient independence unverified:** similarity groups are not patient identifiers.
3. **Single development dataset:** neural weights were optimized on BRISC.
4. **Dataset-source confounding:** geometry, crop, contrast, compression, and encoding may correlate with labels.
5. **Sealed test unopened:** results are development-validation outcomes.
6. **Two-dimensional single-channel input:** no volume or multimodal MRI context.
7. **128×128 target limit:** very small masks may disappear.
8. **Glioma weakness:** segmentation, calibration, and reliability are weakest for glioma.
9. **Independent transport failure:** OpenNeuro and UPENN spatial results failed registered gates.
10. **No independent meningioma transport:** X04 access was blocked; Cheng overlaps BRISC.
11. **No independent screening specificity:** external cohorts are tumor-positive.
12. **Reference-mask uncertainty:** annotations were not independently clinician-adjudicated.
13. **Calibration limitation:** C03 cannot fix geometry and did not transport externally.
14. **Reliability limitation:** R02 ranks external failures but is externally miscalibrated.
15. **Threshold fragility:** the exact internal reliability policy is sensitive to score movement.
16. **Cost assumptions:** 10:1 missed-failure cost is not a measured clinical utility.
17. **Ensemble complexity:** E08 requires four segmenter checkpoints and category routing.
18. **OOD shortcut risk:** O02 dominates and may recognize source/encoding.
19. **OOD benchmark composition:** includes easy far-OOD sources and historically examined near-OOD sources.
20. **REFUSE disabled:** the candidate refusal boundary failed its BRISC false-refusal limit.
21. **Conditional confidence intervals:** bootstrap intervals omit unknown duplication and full workflow adaptivity.
22. **Training not bitwise deterministic:** configs and seeds are recorded, but historical GPU training is not exactly replayable.
23. **Platform aging:** package locks do not guarantee future CUDA/driver availability.
24. **Large-data reproducibility:** raw and controlled-access datasets are not fully distributed in Git.
25. **No application security study:** authentication, privacy, logging, monitoring, and update safety are outside the frozen inference study.

Full limitations: [`docs/limitations.md`](../docs/limitations.md). Hostile internal review: [`audit/stage10_self_review.md`](stage10_self_review.md).

## What is standard and what is contributed

### Standard methods

ResNet-18, U-Net, Dice/BCE loss, boundary loss, ensembling, isotonic regression, logistic regression, Mahalanobis distance, Ledoit–Wolf covariance, bootstrap intervals, and cost-sensitive threshold selection are established methods.

### Specific contribution

The contribution is the integrated, auditable study design: preregistration and amendments; explicit failure gates; retained negative results; a project-specific glioma decomposition; separate calibration, reliability, OOD, and policy claims; independent transport evaluation without adaptation; and a hash-verified frozen stack. Reviewers should decide whether that integration is sufficient novelty for an undergraduate research or reproducibility venue.

## Recommended questions for external reviewers

### 1. Are the internal results credible without verified patient independence?

Please assess which internal claims remain valid as dataset-development results and which should be rejected until a patient-level split exists. State what patient-identity evidence would change your assessment.

### 2. Is “small-mask sensitivity is the strongest measured glioma failure association” supported?

Consider target-size/Dice coupling, confounding, subgroup definitions, and whether the conclusion is descriptive or causal. Suggest a controlled experiment that could test mechanism.

### 3. Is E08 justified relative to v1?

Compare the category improvements with four-checkpoint complexity, classifier routing, reproducibility, latency, and maintenance. State whether a simpler model should remain preferred despite H7.

### 4. What exactly transfers externally?

Evaluate the distinction between failed segmentation Dice, retained failure-ranking AUROC, poor external reliability ECE, and unvalidated policy thresholds. Identify any wording that still overstates transfer.

### 5. Does O05 provide meaningful protection or mostly detect dataset source?

Assess the benchmark composition, O02 dominance, historical source exposure, far-OOD difficulty, false-flag rate, and disabled REFUSE state. Propose an unseen-site benchmark that could falsify the current interpretation.

### 6. Are the statistical and governance claims sufficient?

Assess whether preregistration, amendments, group bootstrap, and fail-closed restarts adequately control researcher degrees of freedom. Identify analyses that remain exploratory despite being registered later.

### 7. What is the true novelty and appropriate venue?

Separate algorithmic novelty, clinical contribution, undergraduate engineering value, reproducibility value, and responsible reporting. Recommend accept/reject for an undergraduate workshop and explain what would change the decision.

## Evidence navigation

- Main report: [`docs/report.md`](../docs/report.md)
- Methods: [`docs/methods.md`](../docs/methods.md)
- Limitations: [`docs/limitations.md`](../docs/limitations.md)
- Artifact appendix: [`docs/appendix.md`](../docs/appendix.md)
- Metric ledger: [`docs/stage9_source_ledger.json`](../docs/stage9_source_ledger.json)
- Pre-registration: [`audit/preregistration.md`](preregistration.md)
- Review log: [`audit/review_log.md`](review_log.md)
- Cross-dataset synthesis: [`audit/stage6_cross_dataset.md`](stage6_cross_dataset.md)
- OOD completion: [`audit/stage7_completion_report.md`](stage7_completion_report.md)
- Final freeze: [`audit/stage8_final_freezing.md`](stage8_final_freezing.md)

## Binding interpretation

This package requests criticism of a frozen research prototype. It does not request medical advice or certification. A positive review would not establish clinical validity. A negative review should distinguish fatal flaws in the narrow undergraduate research claim from limitations already acknowledged by the project.
