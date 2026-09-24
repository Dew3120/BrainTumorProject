# Formal review log

This log is append-only. Each decision records the evidence available at the time. Corrections are new entries that cite the superseded entry; prior text is not silently rewritten.

## 2026-09-24T03:23:39+05:30 — Register the prospective research protocol

- **Decision made:** Accept registration version 1.0 for H1–H6 and freeze every numeric gate. Previously observed results remain retrospective baseline evidence.
- **Triggering artifact:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`.
- **Rationale:** Separating known evidence from prospective tests prevents post-hoc thresholds from being described as preregistered. The protocol defines exact metrics, splits, resampling, stopping rules, and novelty language.
- **Alternatives considered:** Treat all earlier work as preregistered; rejected because its results were already known. Leave future gates informal; rejected because it permits result-dependent changes.
- **Who approved:** User, by explicit foundation-reset instruction; Codex verified internal consistency mechanically.

## 2026-09-24T03:23:39+05:30 — Seal the accepted models as baseline_v1

- **Decision made:** Freeze complete accepted classifier and segmenter run folders, environment files, and consolidated evidence in a create-once, read-only archive.
- **Triggering artifact:** `runs/baseline_v1/manifest.json`, SHA256 `c8a36879aa6e717d60641c5930ebb083aa3b362d98d10f2601584633d4e88a80`; 59 hashed files; 60,198,748 copied bytes.
- **Rationale:** A complete run archive supports exact future comparisons and protects against checkpoint, config, split, source, result, or figure drift.
- **Alternatives considered:** Copy only the winning checkpoints; rejected because preprocessing, configuration, and evaluation evidence would be separable from the model. Track the 60 MB binary archive in Git; rejected in favor of a tracked manifest plus local immutable artifacts.
- **Who approved:** User, by explicit baseline-freeze instruction; Codex approved the copy after source/copy hash equality and read-only verification passed.

## 2026-09-24T03:23:39+05:30 — Adopt file-based experiment tracking beside MLflow

- **Decision made:** Register four prospective experiment directories and require `hypothesis.md`, `config.json`, `result.json`, `decision.md`, and `artifacts/` in each.
- **Triggering artifact:** `experiments/registry.json`, SHA256 `6a19640b662aec69cbad394d2c6fd236bcb435e9767cf2ec5c3b06b13e31ea26`.
- **Rationale:** MLflow records executions, while the file contract records pre-run intent, immutable gates, final decisions, and reviewable artifacts in Git.
- **Alternatives considered:** MLflow alone; rejected because it does not enforce a prospective hypothesis or decision record. One shared results folder; rejected because experiment-specific provenance and artifacts would be ambiguous.
- **Who approved:** User, by explicit experiment-tracking instruction; Codex approved the registry after the directory-contract validator passed.

## 2026-09-24T03:23:39+05:30 — Keep the sealed test partition unopened

- **Decision made:** Record the one-time sealed evaluation as `REGISTERED_NOT_STARTED`; do not evaluate or inspect test predictions during the foundation reset.
- **Triggering artifact:** `experiments/sealed_final_evaluation/config.json`, SHA256 `0f6f7aa06c3c888efcd068642734c831dd0faad1fec427dffb58fefaade79bfd`.
- **Rationale:** The test partition must remain independent of model, threshold, and policy selection. Its only registered use is the final locked evaluation.
- **Alternatives considered:** Evaluate v1 now and a candidate later; rejected because the same test result would then influence candidate development. Use test labels to tune subgroup gates; rejected as leakage.
- **Who approved:** Codex by the preregistered sealed-evaluation rule, within the user's no-test-access constraint.

## 2026-09-24T03:27:45+05:30 — Accept the foundation reset as complete

- **Decision made:** Accept the pre-registration, immutable baseline, experiment registry, and review log as the project's new foundation.
- **Triggering artifact:** `audit/foundation_reset_verification.json`, SHA256 `3593651bee53f0d94dbacc612b26cbb19ca1f2b44bc219b0aa696eed39232418`.
- **Rationale:** The baseline verifier and registry validator pass, Ruff passes, all 60 tests pass, and no test prediction, retraining, threshold change, or application integration occurred.
- **Alternatives considered:** Begin model experimentation before verifying the foundation; rejected because later comparisons would lack a trusted reference. Delay completion until test evaluation; rejected because test access is explicitly reserved for the final locked system.
- **Who approved:** Codex by the predeclared mechanical completion rule; user approval supplied the foundation-reset specification.

## 2026-09-24T03:44:18+05:30 — Register the glioma improvement and reliability program

- **Decision made:** Accept amendment 01 as the prospective protocol for the glioma decomposition, eight controlled interventions, calibration comparisons, reliability comparisons, and review policy. All local gates and final-selection rules are frozen before new outcome analysis.
- **Triggering artifact:** `audit/preregistration_amendment_01.md`, SHA256 `62abb1d13d3ffd16d3fdf41b461ad48aafc746caf4639f18e4545467ebf8c0fb`; updated `experiments/registry.json`, SHA256 `7f480bec5cf5f8b2dc23f6b0c3c5d7e99dfd84f311521f6b31461572caf5ba9c`.
- **Rationale:** The requested program contains many correlated interventions and subgroup analyses. Registering exact implementations, harm controls, deterministic tie-breaks, operational limitations, and model-selection order before inspecting new results limits outcome-driven choices.
- **Alternatives considered:** Run the analyses first and formalize later; rejected because observed outcomes could influence definitions. Treat single-slice JPEG edema and enhancement as visually diagnosable; rejected because sequence and contrast metadata are insufficient. Compare success and failure mean Dice alone; rejected as outcome-defined, so failure rates and group-bootstrap associations are also required.
- **Who approved:** User, by explicitly authorizing the complete Stage 1–5 program and any necessary earlier-phase redo; Codex verified the amendment and registry mechanically.

## 2026-09-24T04:04:18+05:30 — Accept the glioma failure decomposition

- **Decision made:** Accept the frozen-v1 glioma decomposition as complete. Small reference masks are the only predeclared subgroup with a clearly higher failure risk; fragmentation, disconnected predictions, and localization error are common failure mechanisms. Diffuse boundaries and multifocal references remain plausible but do not clear the registered group-bootstrap association rule.
- **Triggering artifact:** `audit/glioma_failure_artifact_manifest.json`, SHA256 `8bac35936e5ceb848d096dfcf6476b83519bf3e7b992d086c62e2c2ea801088a`; 60 failure rows, 204 boundary rows, 10 matched pairs, and 19 primary papers.
- **Rationale:** All 680 validation predictions reproduce stored Dice and IoU with maximum difference 0.0, all 60 failure panels passed image/reference alignment review, all requested geometric fields are populated, and the verifier records zero sealed-test access. The small-mask risk difference is +0.430 with a 95% similarity-group bootstrap interval of +0.258 to +0.592.
- **Alternatives considered:** Attribute failures primarily to diffuse glioma boundaries; rejected because the sharp/diffuse risk-difference interval includes zero. Record edema, enhancement, or ventricle proximity from appearance; rejected because unverified single JPEG slices are insufficient. Treat fragmentation as a preprocessing error; rejected because file pairing and spatial alignment checks passed.
- **Who approved:** Codex by the registered mechanical and evidence rules; the user authorized Stage 1 execution and requested an evidence-based explanation.

## 2026-09-24T10:55:31+05:30 — Reject largest-component filtering

- **Decision made:** Reject E01 as a glioma improvement. The conjunctive local gate failed and will not be relaxed.
- **Triggering artifact:** `experiments/e01_lcc_filter/artifacts/experiment_manifest.json`, SHA256 `7143c62deb6490d813f016243fa4ca51f0d69ba0e82d45b66abeb7fa9df281c2`; all 680 validation cases evaluated.
- **Rationale:** Glioma mean Dice changed from 0.6090 to 0.6050 (`-0.0040`, bootstrap 95% CI `-0.0149` to `+0.0074`) instead of the required `+0.0200`. Meningioma and pituitary changes were `+0.0011` and `+0.0020`, so their harm controls passed. Removing every smaller component deletes true foreground in enough glioma cases to offset false-positive cleanup.
- **Alternatives considered:** Keep the filter because it improves some individual cases; rejected because 55 cases worsened beyond tolerance and the primary glioma mean declined. Tune a minimum component size or apply filtering by category; deferred because those are unregistered variants and cannot rescue E01 after its result is known.
- **Who approved:** Codex by amendment 01's fixed E01 gate; the user authorized sequential execution of all registered interventions.

## 2026-09-24T11:01:45+05:30 — Reject fixed morphology postprocessing

- **Decision made:** Reject E02 as a glioma improvement. Neither preregistered closing-then-opening variant passed the conjunctive local gate, and the gate will not be relaxed.
- **Triggering artifact:** `experiments/e02_morphology/artifacts/experiment_manifest.json`, SHA256 `95b0f46047d2ca97710bffb24e1763507597e9cf3e716905954cab1418839258`; both fixed variants evaluated on all 680 validation cases.
- **Rationale:** The 3x3 variant changed glioma Dice by only `+0.0031` (95% similarity-group bootstrap CI `+0.0004` to `+0.0061`) instead of the required `+0.0200`. The 5x5 variant changed glioma Dice by `-0.0037` and pituitary Dice by `-0.0111`, exceeding the allowed category harm. Model weights and the probability threshold were unchanged, and no sealed-test image was accessed.
- **Alternatives considered:** Accept 3x3 because its small gain is statistically stable; rejected because statistical stability does not satisfy the preregistered effect-size gate. Tune kernel shape, order, or case-specific morphology after seeing these results; rejected as unregistered post-hoc rescue variants.
- **Who approved:** Codex by amendment 01's fixed E02 gate; the user authorized sequential execution of all registered interventions.

## 2026-09-24T11:09:56+05:30 — Reject eight-view test-time augmentation

- **Decision made:** Reject E03 as a glioma improvement. The fixed eight-view probability ensemble failed the glioma benefit gate and the pituitary harm control; its gate will not be relaxed.
- **Triggering artifact:** `experiments/e03_tta/artifacts/experiment_manifest.json`, SHA256 `135fffc1b19a90f32b72422525d3e8e5c37733991b4d0288daa3a77f36edad18`; all 680 validation cases and all eight registered views evaluated.
- **Rationale:** Glioma Dice changed by `-0.1603` (95% similarity-group bootstrap CI `-0.2083` to `-0.1170`) rather than the required `+0.0300`; pituitary Dice changed by `-0.2580`, violating the `-0.0100` harm limit. Meningioma changed by `+0.0075`. Every transform passed an exact inverse-coordinate test, all eight synthetic views were distinct, and the identity view reproduced stored metrics within `2.98e-08`. The result therefore demonstrates poor frozen-model equivariance to the registered rotations/reflections rather than a coordinate inversion error.
- **Alternatives considered:** Keep only transformations that look favorable after inspecting these results; rejected as post-hoc transform selection. Weight the identity view more heavily; rejected as an unregistered rescue. Retrain with corresponding augmentation; remains a separate future intervention rather than a reinterpretation of E03.
- **Who approved:** Codex by amendment 01's fixed E03 gate; the user authorized sequential execution of all registered interventions.

## 2026-09-24T11:16:50+05:30 — Reject two-scale inference

- **Decision made:** Reject E04 as a glioma improvement. The fixed equal-weight 128/256 probability ensemble failed the glioma benefit gate and the pituitary harm control; its gate will not be relaxed.
- **Triggering artifact:** `experiments/e04_multiscale/artifacts/experiment_manifest.json`, SHA256 `bac31e54081338073e2f45ac19bfd3bcacc35dbbf4431b5c1047cb79816eabe3`; all 680 validation cases evaluated at both registered scales.
- **Rationale:** Glioma Dice changed by `-0.0403` (95% similarity-group bootstrap CI `-0.0675` to `-0.0165`) rather than the required `+0.0200`; pituitary Dice changed by `-0.0419`, violating the `-0.0100` harm limit. Meningioma changed by `+0.0083`. The 256-only diagnostic was substantially worse for every category, consistent with using a model trained at 128 pixels outside its training scale. The identity-scale prediction reproduced stored metrics within `2.98e-08`.
- **Alternatives considered:** Tune the 128/256 averaging weight after seeing the result; rejected as post-hoc optimization. Use 256-only inference; rejected because its descriptive category Dice values are all below baseline. Retrain at 256 pixels; that would be a new registered training intervention, not a rescue of E04.
- **Who approved:** Codex by amendment 01's fixed E04 gate; the user authorized sequential execution of all registered interventions.

## 2026-09-24T11:55:11+05:30 — Reject glioma-weighted retraining

- **Decision made:** Reject E05 as a glioma improvement after completing training, checkpoint reload, visual review, failure decomposition, pixel calibration, and case-reliability evaluation. All three category checks failed, and the gate will not be relaxed.
- **Triggering artifact:** `experiments/e05_glioma_weighted/artifacts/experiment_manifest.json`, SHA256 `6b75eee5729aab275d9c852ea77057f8cbd5fb214672cf118b936119c4688d07`; run manifest SHA256 `0e1da3a167990370391bb59bc9f08031c3394e6c8d9e0bcc525cec7ab142acfc`; 680 validation cases and zero sealed-test images.
- **Rationale:** The macro-selected epoch-45 checkpoint changed glioma, meningioma, and pituitary Dice by `-0.0117`, `-0.0312`, and `-0.0287`. The required values were at least `+0.0500`, `-0.0100`, and `-0.0100`. Similarity-group bootstrap intervals were `[-0.0362,+0.0111]`, `[-0.0463,-0.0169]`, and `[-0.0439,-0.0143]`. Weak cases increased from 83 to 103. Visual review found no systematic coordinate or pairing defect. Pixel temperature scaling missed its foreground ECE gate (`0.0580` versus `0.0500`), while the prediction-only case reliability rerun passed (AUROC `0.9423`, ECE `0.0155`, Brier skill `0.5073`, zero-Dice capture `13/17`).
- **Alternatives considered:** Select epoch 46 because its glioma Dice was higher; rejected because the registered checkpoint metric is macro-type Dice. Increase the glioma weight beyond 3 or alter normalization after seeing the result; rejected as post-hoc rescue. Keep E05 for its passing reliability model; rejected because reliability does not replace the failed segmentation-performance gate.
- **Who approved:** Codex by amendment 01's fixed E05 gate; the user authorized the full sequential program and required gates to remain fixed.
