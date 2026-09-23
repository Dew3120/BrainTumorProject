# Brain-tumor MRI project pre-registration

**Registration version:** 1.0

**Registered:** 2026-09-24T03:13:55+05:30

**Evidence cut-off:** Git commit `3b25787f148c2c7b0ed23e19f451df5d5c98a382`

**Frozen comparison baseline:** `runs/baseline_v1/manifest.json`, SHA256 `c8a36879aa6e717d60641c5930ebb083aa3b362d98d10f2601584633d4e88a80`

## Registration boundary

This document governs experiments performed after the evidence cut-off. Results already observed before registration are retrospective baseline evidence and are not described as preregistered. The frozen v1 classifier, segmenter, calibration results, and validation findings may be used only as fixed comparators.

No unlisted hypothesis, dataset, endpoint, subgroup gate, or analysis may be presented as preregistered. A new item requires a versioned amendment committed and added to `audit/review_log.md` before its labels, predictions, or results are inspected. Exploratory work is permitted only when labeled exploratory and cannot change a registered gate.

**The numeric gates in this document will not be relaxed after results are seen.** A failed primary gate remains a failure. Secondary metrics, post-hoc subgroups, and qualitative examples cannot overturn it.

## Registered scope

The current system supports four classifier outputs (`glioma`, `meningioma`, `pituitary`, `no_tumor`), binary tumor/no-tumor output, tumor segmentation for the three tumor classes, prediction-derived case reliability, and research-only measurements. Seven-type classification, causal diagnosis, danger or organ-failure prediction, depth beneath the skull, and physical millimetre measurements without verified spacing are outside this registration.

## Falsifiable hypotheses

| ID | Falsifiable claim | Population and comparison | Primary decision |
|---|---|---|---|
| H1 — glioma improvement | A locked candidate segmenter improves glioma image-mean Dice by at least `0.05` absolute over frozen v1 without materially reducing aggregate or other-type performance. | BRISC internal validation, paired image predictions; v1 versus one locked candidate. | Pass only if every G1 condition passes. |
| H2 — final classification | The locked final classifier maintains strong four-class and binary discrimination on the untouched BRISC test partition. | All 1,000 sealed classification-test images. | Pass only if every G2 condition passes. |
| H3 — final segmentation | The locked final segmenter retains useful overlap on the untouched BRISC mask-test partition, including glioma. | All 860 sealed positive-case mask pairs. | Pass only if every G3 condition passes. |
| H4 — reliability generalization | The locked case-reliability model ranks segmentation failures and remains calibrated on sealed cases. | The same 860 sealed segmentation cases; target is `Dice >= 0.50`. | Pass only if every G4 condition passes. |
| H5 — application integrity | The application gives repeatable inference, rejects unsupported inputs, and never invents physical units when spacing is absent. | Fixed technical test suite plus 100 warmed runs on the registered laptop. | Pass only if every G5 condition passes. |
| H6 — measurement correctness | Mask-derived pixel area and maximum diameter are implemented correctly independent of model quality. | Deterministic synthetic masks with analytically known answers. | Pass only if every G6 condition passes. |

H2–H4 are evaluated once, together, after all model and policy artifacts are locked. If H1 is not attempted, frozen v1 becomes the locked final segmenter. The sealed partition is not used to choose between candidates.

## Predeclared numeric gates

### G1 — candidate segmenter development gate

All conditions are conjunctive:

1. Paired glioma mean-Dice change `candidate - v1 >= +0.050`.
2. The paired similarity-group bootstrap 95% CI lower bound for glioma mean-Dice change is `> 0.000`.
3. Overall image-mean Dice change is `>= -0.010`, and its CI lower bound is `>= -0.020`.
4. Meningioma and pituitary mean-Dice changes are each `>= -0.020`.
5. Glioma zero-Dice count does not exceed v1's count of `12` on the fixed 204-case validation subset.
6. Exactly one candidate may be nominated for final evaluation. If several development runs pass, selection uses highest glioma mean Dice; ties within `1e-12` use higher macro-type Dice, then lexicographically smaller experiment ID.

The `+0.05` target is large enough to be material relative to v1 glioma Dice `0.6090`, rather than a rounding fluctuation. The `-0.01/-0.02` non-inferiority limits prevent trading broad performance for a narrow subgroup gain. Other-type limits of `-0.02` prevent hidden category harm.

### G2 — one-time sealed classifier gate

All conditions are conjunctive:

1. Four-class macro F1 `>= 0.900`.
2. Tumor sensitivity `>= 0.950` at the unchanged binary threshold `0.50`.
3. No-tumor specificity `>= 0.900` at threshold `0.50`.
4. Recall for each of the four classes `>= 0.850`.
5. Raw top-label ECE `<= 0.050` using 10 equal-frequency bins.
6. The similarity-group bootstrap 95% CI lower bound for macro F1 is `>= 0.850`.

These thresholds demand balanced class behavior while allowing a realistic decrease from the known internal-validation macro F1 `0.9779`. Sensitivity is stricter because missed tumors are the most consequential classifier error. This remains an internal-source benchmark, not a clinical sensitivity claim.

### G3 — one-time sealed segmenter gate

All conditions are conjunctive:

1. Overall image-mean Dice `>= 0.700`.
2. Macro-type mean Dice `>= 0.700`.
3. Glioma image-mean Dice `>= 0.550`.
4. Each of meningioma and pituitary image-mean Dice `>= 0.750`.
5. Overall zero-Dice fraction `<= 0.050`.
6. The similarity-group bootstrap 95% CI lower bounds are `>= 0.680` for overall image-mean Dice and `>= 0.500` for glioma mean Dice.

The thresholds are below v1 internal-validation values (`0.7694` overall and `0.6090` glioma) to allow distribution shift while still requiring useful overlap. The zero-Dice limit caps complete spatial misses at 5%.

### G4 — one-time sealed case-reliability gate

All conditions are conjunctive:

1. Failure-detection AUROC `>= 0.750`, where failure is hard-mask Dice `< 0.50` and risk is `1 - P(Dice >= 0.50)`.
2. AUROC similarity-group bootstrap 95% CI lower bound `>= 0.700`.
3. Reliability ECE `<= 0.050` using 10 equal-frequency bins.
4. Brier skill `> 0.000`, and its bootstrap 95% CI lower bound is `> 0.000`.
5. The highest-risk 20% of cases capture at least `ceil(0.50 * N_failure)` Dice-below-0.50 cases.
6. If at least one zero-Dice case exists, the same highest-risk quintile captures at least `ceil(0.50 * N_zero)` zero-Dice cases. If `N_zero = 0`, this condition passes but is reported as not estimable.

AUROC `0.75`, ECE `0.05`, and positive Brier skill are the previously declared minimum useful discrimination, calibration, and baseline improvement. Capturing more than half of catastrophic failures in one fifth of cases makes the score operationally useful for review prioritization.

### G5 — application integrity gate

All conditions are conjunctive:

1. Ten repeated inferences for every case in a fixed 20-case fixture produce the same predicted class and bitwise-identical hard mask; maximum absolute probability difference is `<= 1e-6`.
2. All malformed or unsupported fixtures are rejected: `100%` of at least 20 fixtures, with no model result emitted.
3. When pixel spacing is missing or unverified, `100%` of fixtures omit millimetres and use pixels only.
4. When verified spacing is supplied to synthetic fixtures, `100%` of displayed unit conversions match the registered calculation within `1e-6`.
5. On the registered RTX 4060 laptop, after 10 warm-up runs, 100 single-image runs have median end-to-end latency `<= 2.0 s` and 95th percentile latency `<= 5.0 s`.
6. Any checkpoint, split, model-version, or baseline-manifest hash mismatch stops inference in `100%` of tests.

The tolerances distinguish numerical repeatability from display rounding. Latency limits define a responsive local demonstration and are not portable performance claims.

### G6 — measurement-engine gate

All conditions are conjunctive across at least 50 deterministic synthetic masks:

1. Pixel area equals the foreground-pixel count exactly for `100%` of masks.
2. Maximum Euclidean diameter absolute error is `<= 1.0` pixel for every mask.
3. Bounding-box width and height absolute errors are each `<= 1.0` pixel for every mask.
4. With verified synthetic spacing, physical area relative error is `<= 1%` and physical diameter relative error is `<= 2%` for every nonzero mask.
5. Empty masks produce an explicit `not measurable` state in `100%` of cases.

One-pixel geometric tolerances cover discrete raster boundaries. The 1–2% physical tolerances test arithmetic and coordinate conversion only; they do not validate predicted tumor boundaries.

## Metric definitions

Let `TP_c`, `FP_c`, and `FN_c` be one-vs-rest counts for class `c`; `N` is the number of cases; `C=4`; `y_ic` is a one-hot label; `p_ic` is a predicted probability. Tumor probability is the sum of glioma, meningioma, and pituitary probabilities. No-tumor is negative.

- **Accuracy:** `sum_i 1(argmax_c p_ic = y_i) / N`.
- **Precision for class c:** `TP_c / (TP_c + FP_c)`; zero division returns `0`.
- **Recall for class c:** `TP_c / (TP_c + FN_c)`; zero division returns `0`.
- **F1 for class c:** `2 * precision_c * recall_c / (precision_c + recall_c)`; zero division returns `0`.
- **Macro F1:** `(1/C) * sum_c F1_c`.
- **Balanced accuracy:** `(1/C) * sum_c recall_c`.
- **Tumor sensitivity:** `TP_tumor / (TP_tumor + FN_tumor)` at threshold `0.50`.
- **No-tumor specificity:** `TN_tumor / (TN_tumor + FP_tumor)` at threshold `0.50`.
- **AUROC:** probability that a randomly chosen positive receives a higher score than a randomly chosen negative, with ties contributing `0.5`; computed by the standard rank statistic.
- **Multiclass NLL:** `-(1/N) * sum_i log(clip(p_i,y_i, 1e-15, 1))`.
- **Multiclass Brier:** `(1/N) * sum_i sum_c (p_ic - y_ic)^2`.
- **Binary NLL:** `-(1/N) * sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)]`, probabilities clipped to `[1e-15, 1-1e-15]`.
- **Binary Brier:** `(1/N) * sum_i (p_i-y_i)^2`.
- **ECE:** `sum_b (n_b/N) * |accuracy_b - confidence_b|`. Classifier and case-level reliability use 10 equal-frequency bins with deterministic index tie-breaking. Pixel reports use 15 equal-width bins on `[0,1]` with equal total weight per image.
- **MCE:** `max_b |accuracy_b - confidence_b|` over occupied bins.

For segmentation, hard prediction `P_i` is `sigmoid(logit) >= 0.50`; reference mask is `R_i`; `I_i = |P_i intersect R_i|`.

- **Dice per image:** `2*I_i / max(1, |P_i| + |R_i|)`.
- **IoU per image:** `I_i / max(1, |P_i| + |R_i| - I_i)`.
- **Image-mean Dice/IoU:** arithmetic mean over cases.
- **Type-mean Dice:** arithmetic mean within one tumor type.
- **Macro-type Dice:** unweighted mean of glioma, meningioma, and pituitary type means.
- **Zero-Dice fraction:** `sum_i 1(Dice_i = 0) / N`.
- **Failure:** `Dice_i < 0.50`; **success:** `Dice_i >= 0.50`.
- **Risk capture at 20%:** failures among the `ceil(0.20*N)` highest-risk cases divided by all failures. Ties are broken by normalized path ascending.
- **Reliability Brier skill:** `1 - Brier_model / Brier_baseline`. In cross-fitting, each held-out case's baseline is the training-fold success prevalence; on the sealed set, the baseline is the locked development success prevalence.

For measurements:

- **Pixel area:** foreground-pixel count.
- **Physical area:** pixel area multiplied by verified row spacing and column spacing.
- **Maximum diameter:** maximum Euclidean distance between boundary-pixel centres; physical distance uses the two spacing dimensions before maximization.
- **Latency:** wall-clock time from accepted decoded input through completed result object, excluding UI rendering; median and nearest-rank 95th percentile are reported.

## Dataset and split freeze

| Item | Registered value | SHA256 |
|---|---|---|
| Dataset source descriptor | BRISC, Kaggle owner `briscdataset`, version 6, CC BY 4.0 | `53f4aba8d78f245bf95f56d3c71cbf23ec916b61b2243b2ee4a3b7407b955c35` |
| Canonical inventory and split | `audit/runs/20260921-182736-884368/inventory.csv`; contains per-file SHA256, pixel hash, label, mask path, similarity group, and split | `d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28` |
| Working split snapshot | `data/splits/draft_split.csv` | `d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28` |
| Similarity candidates | `audit/runs/20260921-182736-884368/similarity_candidates.csv` | `dab1cdcef3173315544d7d39807b54cd356d0a39223885df840f83ebd5cca1a0` |
| Frozen baseline manifest | `runs/baseline_v1/manifest.json` | `c8a36879aa6e717d60641c5930ebb083aa3b362d98d10f2601584633d4e88a80` |

Classification membership is fixed at 3,570 train, 893 validation, 1,000 sealed test, and 537 review-excluded images. The sealed classifier test contains 254 glioma, 306 meningioma, 300 pituitary, and 140 no-tumor images. Segmentation membership is fixed at 2,719 train pairs, 680 validation pairs, and 860 sealed test pairs (254 glioma, 306 meningioma, 300 pituitary); 534 additional masked cases remain review-excluded.

The data source does not guarantee patient-independent membership or modality consistency. Similarity groups reduce obvious duplicate leakage but are not patient identifiers. BRISC is the only registered dataset. No external dataset is registered, so no claim of external, patient-independent, scanner-independent, or clinical generalization is permitted.

## Analysis protocol

### Development and cross-fitting

- Random seed is `42` unless a config explicitly preregisters additional seeds before execution.
- Five-fold `StratifiedGroupKFold`, shuffled with seed `42`, is used where model fitting and evaluation share the 680-case development population.
- Groups are the fixed transitive components of exact decoded-pixel matches or perceptual-hash Hamming distance `<= 4`; the accepted fold-assignment SHA256 is `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Scaling, feature normalization, and reliability-model fitting occur inside each training fold. Reported development scores are out-of-fold only.
- Reference masks may create and evaluate Dice targets. They cannot enter reliability features.

### Bootstrap and confidence intervals

- Use `2,000` resamples with NumPy generator seed `42`.
- Resampling unit is the fixed similarity group; sampling is with replacement and retains every member of each sampled group.
- Resampling is not fold-stratified.
- For model comparisons, the same sampled groups are used for both models and the statistic is `candidate - v1`.
- The 95% interval is the percentile interval at the `2.5` and `97.5` percentiles; it is not BCa.
- A draw lacking both binary target classes is discarded and deterministically redrawn. Failure to obtain 2,000 valid draws within 10,000 attempts invalidates the analysis.
- Report point estimate, bootstrap mean, interval, number of groups, repetitions, and seed.

### Sealed evaluation

The test partition remains unopened for model development. Before its one allowed evaluation, all checkpoints, configs, scripts, thresholds, review policy, environment lock, and the exact evaluation command must be hashed in an experiment folder. Frozen v1 and the locked candidate are evaluated in the same run. After test results exist, no retraining, model selection, threshold adjustment, calibration refit, or gate amendment is allowed. A later model requires a new external test dataset registered before access.

## Pass, fail, and stopping rules

1. A gate passes only when every listed primary condition passes. Equality passes for `>=` or `<=`; strict inequalities remain strict.
2. Missing artifacts, hash differences, non-finite metrics, changed membership, unexpected labels, or test access before lock cause `INVALID/STOP`, not a pass or a fail.
3. A failed gate closes that experiment. A different model, loss, resolution, or threshold is a new experiment with a new folder and prospective hypothesis.
4. Secondary metrics and qualitative panels are reported but cannot reverse the primary decision.
5. Per-category results are always reported. Subgroup claims without a registered CI or gate are descriptive.
6. If H1 fails, v1 remains the final model. No weaker post-hoc candidate is substituted.
7. The sealed evaluation is performed once. Its failure is final for this dataset version.

## Novelty rules

| Experiment | Expected result | What may be called novel inside this project | What is required for field-level novelty |
|---|---|---|---|
| H1 glioma improvement | Incremental changes from loss, sampling, resolution, or architecture tuning | A preregistered, statistically stable `>=0.05` glioma gain without the declared harms | A new method compared with strong baselines and confirmed on patient-independent external data |
| H2 classification | Reproduction of a standard four-class transfer-learning task | Rigorous sealed evaluation, shortcut audit, and calibrated reporting | External multi-centre evidence or a substantively new method; high BRISC accuracy alone is not novel |
| H3 segmentation | Reproduction of 2D U-Net-style BRISC segmentation | A locked model meeting all subgroup and catastrophic-failure gates | External prospective or multi-centre validation and comparison with established systems |
| H4 reliability | Prediction-only uncertainty features should rank some failures | A preregistered score that passes all sealed discrimination, calibration, and capture gates | A methodologically new reliability approach with external validation and competitive baselines |
| H5 application integrity | Deterministic inference and input rejection are expected engineering | Complete fail-closed provenance and review-priority integration | Not a medical-AI novelty claim by itself |
| H6 measurements | Correct geometry on synthetic masks is expected | Traceable measurement restrictions that prevent unsupported physical claims | Clinical measurement agreement using verified imaging geometry and expert reference standards |

No current experiment is predeclared as field-level medical novelty. An internally successful result may be described as a project contribution or engineering finding only.

## Amendment rule

An amendment must state its reason, exact changes, date, authorizer, and hashes; update the experiment registry and review log; and be committed before affected data or results are inspected. Amendments cannot alter the interpretation of completed results. This version's gates are permanent for the registered experiments.
