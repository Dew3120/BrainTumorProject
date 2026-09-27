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

## 2026-09-24T16:14:16+05:30 — Reject boundary-aware retraining

- **Decision made:** Reject E06 as a glioma improvement after completing checkpoint reload, visual review, failure analysis, two-pixel-tolerance boundary-F1 comparison, pixel calibration, and case-reliability evaluation. Both primary benefit checks failed, and the gate was not relaxed.
- **Triggering artifact:** `experiments/e06_boundary_loss/artifacts/experiment_manifest.json`, SHA256 `181584ee6f14e5d95bde4cd369ec7d56724f16166999cd0cbc269fbb5e97c773`; 680 validation cases and zero sealed-test images.
- **Rationale:** The selected epoch-40 checkpoint changed glioma Dice by `-0.0296` (95% similarity-group bootstrap CI `[-0.0533,-0.0070]`) and glioma boundary F1 by `-0.0470` (CI `[-0.0766,-0.0195]`). The conjunctive gate required both changes to be at least `+0.0500`. Meningioma and pituitary Dice also changed by `-0.0103` and `-0.0110`. Visual review found no systematic technical defect. Pixel calibration missed its foreground ECE gate (`0.0584` versus `0.0500`), while prediction-only case reliability passed (AUROC `0.9153`, ECE `0.0295`, Brier skill `0.3381`, zero-Dice capture `11/17`).
- **Alternatives considered:** Increase or normalize the boundary term after observing the result; rejected as an unregistered rescue. Select a checkpoint by glioma or boundary F1 instead of the registered macro-Dice rule; rejected as post-hoc selection. Keep E06 because reliability passed; rejected because reliability does not replace failed segmentation and boundary gates.
- **Who approved:** Codex by amendment 01's fixed E06 gate; the user authorized the full sequential program and required gates to remain fixed.

## 2026-09-24T16:17:32+05:30 — Register glioma-specialist routed segmentation

- **Decision made:** Register E07 before implementation. Train one fresh glioma-only segmenter and route only frozen-classifier `glioma` predictions to it; all other predictions use frozen v1.
- **Triggering artifact:** `experiments/e07_glioma_specialized/config.json`, SHA256 `d98d46650bfc6ee3773ce8f6c02f44c52c928086267ce9285bdce1b156991113`; hypothesis SHA256 `8c57bf5d9bc095e18e1714e3bad4ebca8d8680803032f45ef014eb6c9323a01d`; registry SHA256 `252a7186d5112303f7113901e041a04ea3e9172023776aaa8047f09a4f1b859c`.
- **Rationale:** The protocol isolates specialization from routing. Checkpoint selection uses only the 204 glioma validation pairs, while the fixed conjunctive system gate requires routed glioma Dice improvement of at least `+0.08`, exact four-class routing accuracy of at least `0.95`, and no more than `0.01` Dice loss for meningioma or pituitary.
- **Alternatives considered:** Route by the dataset's true label; rejected because it is unavailable operationally and would inflate performance. Train on all tumor types with an auxiliary head; rejected because E07 specifically registers a glioma-only specialist. Select checkpoints by routed system performance; rejected because repeated routing evaluation would add another outcome-driven selection layer.
- **Who approved:** User through the authorized Stage 2 program; Codex fixed the previously unspecified checkpoint-selection and routing definitions before implementation.

## 2026-09-24T16:42:31+05:30 — Reject classifier-routed glioma specialist

- **Decision made:** Reject E07 after completing glioma-only training, exact checkpoint reload, frozen-classifier routing, visual review, failure analysis, pixel calibration, and case-reliability evaluation. The conjunctive gate failed and was not relaxed.
- **Triggering artifact:** `experiments/e07_glioma_specialized/artifacts/experiment_manifest.json`, SHA256 `643449c6d645113209ab5c90bfb7bc024a265c20e3e4a9736946e9d57187ee50`; 680 validation cases and zero sealed-test images.
- **Rationale:** Frozen-classifier routing accuracy passed at `0.9721`, and meningioma/pituitary harm controls passed with Dice changes of `-0.0034` and `0.0000`. Routed glioma Dice changed from `0.6090` to `0.5802` (`-0.0287`, 95% similarity-group bootstrap CI `[-0.0596,+0.0015]`) instead of the required `+0.0800`. Weak cases increased from 83 to 89. Visual review found no systematic technical defect. Pixel calibration failed its foreground ECE/background NLL gate, while prediction-only case reliability passed (AUROC `0.9410`, ECE `0.0171`, Brier skill `0.4490`, zero-Dice capture `10/13`).
- **Alternatives considered:** Route using the true tumor label; rejected as target leakage. Select a specialist checkpoint by routed system performance or combine specialist and general probabilities after observing the result; rejected as post-hoc rescue. Keep E07 because routing and reliability passed; rejected because they do not replace the failed glioma segmentation gate.
- **Who approved:** Codex by amendment 01's fixed E07 gate; the user authorized the full sequential program and required gates to remain fixed.

## 2026-09-24T16:45:30+05:30 — Register fixed E05–E07 probability ensemble

- **Decision made:** Register E08 before ensemble inference. Average the frozen E05, E06, and routed E07 raw probability maps with equal one-third weights and threshold once at `0.50`.
- **Triggering artifact:** `experiments/e08_ensemble/config.json`, SHA256 `4a6ca00b0aae03a75f6efef8f2f68e318a3415c8d30e4e41cc8afc1a61368bdb`; hypothesis SHA256 `a479d158af3b2449f3bfa785483325aa709c57653214295c105995be4f49b3e4`; registry SHA256 `7d7e1b20ccc2ef90ede06fda14449964db75c879d37ca46303aff3b72d60db8d`.
- **Rationale:** Amendment 01 requires all listed experiments to run. The previously ambiguous comparator is resolved using its own final-selection rule: E05 is excluded for registered category harm, E06 and E07 remain eligible, and E07 is the higher-glioma-Dice comparator. The fixed ensemble must reach at least `0.6002066652` glioma Dice and preserve both other categories within `0.01` of v1.
- **Alternatives considered:** Skip E08 because no single model passed its full local gate; rejected because the multiple-comparison rule requires every listed experiment. Choose components or weights after inference; rejected as post-hoc selection. Treat true tumor type as the E07 route; rejected as leakage.
- **Who approved:** User through the authorized Stage 2 program; Codex resolved the comparator using amendment 01's predeclared category-harm filter before ensemble inference.

## 2026-09-24T17:47:23+05:30 — Accept fixed segmentation ensemble locally

- **Decision made:** Accept E08 under its local gate after exact component reproduction and visual review. Do not replace frozen v1 because parent G1 remains unmet.
- **Triggering artifact:** `experiments/e08_ensemble/artifacts/experiment_manifest.json`, SHA256 `496d689a37ca98293542cdcd56f320cd04ee60d30a6fb01177bc7c6f6d1183e6`; 680 validation cases and zero sealed-test images.
- **Rationale:** The equal-weight ensemble achieved glioma Dice `0.6324`, a `+0.0522` change over the best eligible v2 single system E07, exceeding the local `+0.0200` requirement. Meningioma and pituitary improved over v1 by `+0.0119` and `+0.0096`, so both harm controls passed. Every component reproduced stored Dice/IoU with maximum difference `0.0`, and visual review found no systematic technical defect. The ensemble's glioma gain over frozen v1 was only `+0.0234`, below parent G1's `+0.0500` replacement gate.
- **Alternatives considered:** Promote E08 because it passed locally; rejected because parent G1 governs model replacement. Tune ensemble weights to seek a larger gain; rejected as an unregistered post-hoc variant. Open the sealed test to choose between v1 and E08; rejected because test access remains reserved for the final locked system.
- **Who approved:** Codex by amendment 01's fixed E08 local gate and parent G1; the user authorized execution of the full sequential program.

## 2026-09-24T17:54:01+05:30 — Select E08 as experimental v2; retain frozen v1

- **Decision made:** Select E08 as the best category-harm-safe experimental v2 system for the requested calibration and reliability studies. Retain frozen v1 as the accepted final model because E08 fails parent G1.
- **Triggering artifact:** `audit/v2_model_selection.json`, SHA256 `105eb71939a002d222b17fbf1cd4137bb4b4a136b934dae18aec5f561ccd45c4`; report SHA256 `e6602bf691ce3e30d281419cfb6e46596eaee6a4a1e4617cf048a2ef34126c73`.
- **Rationale:** E08 has the highest harm-safe glioma Dice (`0.6324`) and improves the other two categories. Its `+0.0234` glioma change over v1 is below parent G1's `+0.0500` requirement, so it cannot replace v1. Pixel-calibration and reliability tie-breaks were unnecessary because the primary glioma ranking was not tied.
- **Alternatives considered:** Promote E08 on its local pass; rejected because the parent replacement gate is controlling. Select E02 because it is simpler; rejected because E08 has higher glioma and macro Dice while remaining harm-safe. Treat later calibration results as permission to reverse model selection; rejected because those analyses are downstream and cannot retroactively change the frozen performance gate.
- **Who approved:** Codex by amendment 01's fixed selection and parent-gate rules; the user authorized the full staged program.

## 2026-09-24T17:59:09+05:30 — Register vector-scaling calibration

- **Decision made:** Register C01 before fitting. Evaluate E08 segmenter binary affine-logit vector scaling and descriptive frozen-classifier vector scaling with five-fold similarity-aware cross-fitting.
- **Triggering artifact:** `experiments/c01_vector_scaling/config.json`, SHA256 `c2a795b371246970f82bcb40406d0c2f4aad149b711af410604bba0ea861c0f2`; hypothesis SHA256 `610fd580dd12e38ba96a7099e48cd54ee5d99501ca6780936d29fb71b056763f`; registry SHA256 `4ad3b182f85e149fda3ef6dd4059a33414ed821a98ce8de8d40bd8b2de80f548`.
- **Rationale:** The segmenter gate requires NLL improvement and ECE at most `0.05` in each of three fixed regions. Reported metrics use out-of-fold outputs; all-validation parameters are operational proposals only. The existing raw-classifier acceptance remains unchanged unless a separately registered rule says otherwise.
- **Alternatives considered:** Fit and evaluate on all validation cases; rejected as calibration leakage. Use reference masks as operational inputs; rejected because lesion-centered regions are evaluation constructs only. Reuse equal-width ECE bins; rejected for this new program because Stage 3 predeclares 15 equal-frequency bins for unified comparison.
- **Who approved:** User through the authorized Stage 3 program; Codex fixed the identifiable binary parameterization and metric definitions before fitting.

## 2026-09-24T18:08:00+05:30 — Accept vector-scaling calibration experiment

- **Decision made:** Accept C01 under its predeclared segmenter calibration gate. Preserve the fixed 0.50 decision threshold and treat the learned all-validation parameters as proposed operational calibration parameters only; they were not used to compute reported metrics.
- **Triggering artifact:** `experiments/c01_vector_scaling/artifacts/experiment_manifest.json`, SHA256 `ee4c1e1bd15742115a5a6c08440fc2f2579f59d5665675a428abac6c6992d45d`; 680 segmentation-validation cases, 893 classifier-validation cases, and zero sealed-test images.
- **Rationale:** Five-fold similarity-aware out-of-fold vector scaling improved lesion-centered NLL from `0.3172` to `0.2274`, background NLL from `0.002732` to `0.002439`, and whole-image NLL from `0.02323` to `0.01776`. Corresponding ECE values were `0.0382`, `0.00214`, and `0.00033`, all at or below the registered `0.05` ceiling. The affine mapping changed 43,935 hard-mask pixels across all 680 images; this is explicitly recorded, so C01 is accepted as a calibration result and does not silently replace the experimental segmentation mask policy. Descriptive classifier vector scaling worsened binary NLL, Brier, and ECE, so the existing raw-classifier policy remains unchanged. A metadata-only case-count field was corrected from 680 to the verified 893 classifier rows; no metric changed.
- **Alternatives considered:** Relax the gate or hide hard-mask changes; rejected. Promote calibrated classifier probabilities despite worse binary metrics; rejected. Evaluate the all-validation fit on the same cases used to fit it; rejected as calibration leakage.
- **Who approved:** Codex by the registered C01 rule; the user authorized sequential execution of Stage 3 and required fixed gates.

## 2026-09-24T18:16:04+05:30 — Register conditional E08 final-model pathway

- **Decision made:** Register amendment 02 and H7 before any Stage 4 result is inspected. E08 may be nominated for the one-time sealed evaluation only when an eligible operational calibration passes its fixed gate and the selected gate-passing E08 reliability model has out-of-fold AUROC at least `0.941`.
- **Triggering artifact:** `audit/preregistration_amendment_02.md`, SHA256 `0153a8709f16ee38e4e2e2cea6044c840bb8b775971e2d090a02537a473e6a43`; selection config SHA256 `f326f3385b4e9bc8363140fa5d8f9112945e0fa945e06dfc190870553c35d78c`.
- **Rationale:** This is a new prospective selection pathway requested after Stage 2 closed. It does not change G1. C01 was already known to pass locally; C02–C09 and all E08 reliability results remain unobserved. Both H7 conditions are conjunctive, and passing only nominates E08 for sealed evaluation.
- **Alternatives considered:** Retroactively declare E08 final from its local ensemble pass; rejected. Lower G1; rejected. Use a reference-mask-dependent calibrator operationally; prohibited.
- **Who approved:** User explicitly requested and defined the new conditional hypothesis; Codex fixed the operational selection and tie-break rules before Stage 4.

## 2026-09-24T18:16:04+05:30 — Register two-class Dirichlet pixel calibration

- **Decision made:** Register C02 before implementation using the identifiable binary Dirichlet map and five-fold similarity-aware cross-fitting.
- **Triggering artifact:** `experiments/c02_dirichlet/config.json`, SHA256 `36c4247047098f303a727fe38f3a5c3ca912de0ac7af9592cf8751bc3b145bc1`; hypothesis SHA256 `88798ca78ea1199dd00cd0ae69fe43e5886e9e8fa8f0484b6a7fc2530d3e9007`.
- **Rationale:** C02 must improve NLL in all three fixed regions, keep all three ECE values at or below `0.05`, and change zero hard-mask pixels at threshold `0.50`. The gate is conjunctive and will not be relaxed.
- **Alternatives considered:** Fit and evaluate on all validation pixels; rejected as leakage. Omit the hard-mask identity rule; rejected because it is explicitly registered.
- **Who approved:** User through the sequential C02–C09 instruction; Codex fixed the identifiable parameterization before fitting.

## 2026-09-24T21:41:17+05:30 — Reject two-class Dirichlet pixel calibration

- **Decision made:** Reject C02 under its unchanged conjunctive gate. NLL improvement passed in all three registered regions, ECE at most `0.05` passed in all three regions, zero changed hard-mask pixels failed, and zero changed hard-mask images failed.
- **Triggering artifact:** `experiments/c02_dirichlet/artifacts/experiment_manifest.json`, SHA256 `af2b470c045f8376780fe96b53b81af5838a4001463e650c5994e3fb480b2dbd`; 680 segmentation-validation cases and zero sealed-test images.
- **Rationale:** Five-fold similarity-aware out-of-fold Dirichlet calibration reduced lesion-centered NLL from `0.317163` to `0.226449`, background NLL from `0.002732` to `0.002435`, and whole-image NLL from `0.023231` to `0.017734`. Calibrated ECE was `0.036148`, `0.002162`, and `0.000210`, satisfying every ECE ceiling. However, the calibrated mapping changed `43,228` hard-mask pixels across all `680` images at the unchanged `0.50` threshold. Both mandatory identity checks therefore failed, and the gate was not relaxed.
- **Alternatives considered:** Ignore the identity checks because probability metrics improved; rejected because C02 explicitly requires bitwise-identical masks. Retune the mapping to preserve the decision boundary after seeing the result; rejected as an unregistered variant. Change the threshold to recover the original masks; prohibited.
- **Who approved:** Codex by the registered C02 rule; the user authorized sequential C02–C09 execution and required failures to remain findings.

## 2026-09-24T22:04:35+05:30 — Register isotonic pixel calibration

- **Decision made:** Register C03 before implementation or fitting. Evaluate deterministic five-fold similarity-aware isotonic calibration on frozen E08 probabilities.
- **Triggering artifact:** `experiments/c03_isotonic/config.json`, SHA256 `46be5b7b0f1216a24bd7de9019e1541ffef97a69be65055544aaa2753d9c2a08`; hypothesis SHA256 `b65f6a5ad02e931498768bbbbef7df91b87fa0dd6577aea250a4f750e37f9632`.
- **Rationale:** The C03 gate is fixed as six conjunctive checks: ECE at most `0.04` and Brier not worse in each of the lesion-centred, background, and whole-image regions. The mapping uses only raw probability at inference. Hard-mask changes at the unchanged `0.50` threshold are counted overall and by category but do not control the registered C03 gate.
- **Alternatives considered:** Fit separate mappings with reference-region routing; rejected because reference masks are unavailable at inference. Sample pixels; rejected because exact deterministic aggregation of identical stored probabilities preserves all fitting information. Treat hard-mask identity as a new gate; rejected because amendment 01 requires reporting rather than identity for C03.
- **Who approved:** User explicitly authorized C03 registration and required plan review before execution; no mapping has been fitted.

## 2026-09-24T22:33:00+05:30 — Accept isotonic pixel calibration experiment

- **Decision made:** Accept C03 under its unchanged six-check conjunctive gate. Preserve threshold `0.50`; hard-mask changes remain descriptive as registered.
- **Triggering artifact:** `experiments/c03_isotonic/artifacts/experiment_manifest.json`, SHA256 `beec6f5526204b1dc5c44bef17048d56ad7f999d15b4dde308d9303f38a9abe4`; 680 segmentation-validation cases and zero sealed-test images.
- **Rationale:** Five-fold similarity-aware out-of-fold isotonic calibration reduced lesion-centred ECE from `0.050543` to `0.033509`, background ECE from `0.001761` to `0.002005`, and whole-image ECE from `0.002503` to `0.000076`; every value remained at or below `0.04`. Brier improved in all three regions: `0.070381` to `0.063371`, `0.000719` to `0.000344`, and `0.005060` to `0.004520`. The mapping changed 22,891 hard-mask pixels across 679 images; this is reported but is not a C03 gate condition.
- **Alternatives considered:** Reject C03 because hard masks changed; rejected because amendment 01 and the registered C03 rule make this descriptive. Evaluate the final all-validation fit on its fitting cases; rejected as leakage. Use reference-region routing at inference; rejected because the operational mapping accepts raw probability only.
- **Who approved:** Codex by the registered C03 rule; the user approved the C03 plan and authorized uninterrupted C03–C09 execution subject to fixed stop conditions.

## 2026-09-24T22:39:00+05:30 — Register separate-region temperature diagnostic

- **Decision made:** Register C04 before fitting as a five-fold similarity-aware oracle diagnostic with independent positive temperatures for the lesion-centred and background regions.
- **Triggering artifact:** `experiments/c04_split_temperature/config.json`, SHA256 `51e1adbf5c96a05dd6b9123e66d63b9cac4a17537437249957fe94f4f11422bf`; hypothesis SHA256 `38d4032d39a0692aa6880ad5ffb5b70cc58acf0a272c4f94c96e17d391a3d440`.
- **Rationale:** The conjunctive gate requires lesion-centred ECE at most `0.05`, background ECE at most `0.02`, and no Brier worsening in either gated region. Whole-image and per-category metrics are descriptive. Reference-defined pixel routing makes C04 ineligible as an operational H7 calibrator regardless of its local result.
- **Alternatives considered:** Route pixels using the unavailable reference mask in production; rejected as leakage. Treat all pixels with one temperature; already evaluated by earlier registered methods. Add whole-image performance to the gate; rejected because the registered C04 gate names foreground and background.
- **Who approved:** User through the approved uninterrupted C03–C09 sequence; Codex fixed the explicit gate before any C04 fit.

## 2026-09-24T22:47:00+05:30 — Accept separate-region temperature diagnostic locally

- **Decision made:** Accept C04 under its unchanged local gate, while retaining its preregistered oracle-only and H7-ineligible status.
- **Triggering artifact:** `experiments/c04_split_temperature/artifacts/experiment_manifest.json`, SHA256 `60d3dc3ac35cb9813c666e52a7600c3e6e3fb4101d54abe3d0a4d8798ca376b0`; 680 segmentation-validation cases and zero sealed-test images.
- **Rationale:** Five-fold out-of-fold lesion-centred ECE was `0.024521` (gate `<=0.05`) and background ECE was `0.001512` (gate `<=0.02`). Lesion-centred Brier improved from `0.070381` to `0.068105`, and background Brier improved from `0.000719` to `0.000667`. All four checks passed. The separate temperatures changed zero hard-mask pixels at threshold `0.50`.
- **Alternatives considered:** Treat the local pass as operational; rejected because the reference-defined routing mask is unavailable at inference. Use its whole-image result to satisfy H7; rejected by amendment 02's operational-input rule.
- **Who approved:** Codex by the registered C04 rule; the user authorized uninterrupted C03–C09 execution subject to fixed stop conditions.

## 2026-09-24T22:53:00+05:30 — Register reliability-weighted calibration

- **Decision made:** Register C05 before fitting as a five-fold similarity-aware scalar temperature whose fitting cases are weighted by frozen v1 out-of-fold reliability scores.
- **Triggering artifact:** `experiments/c05_reliability_weighted/config.json`, SHA256 `9c623b2ed999f383a52a2ae712f31c0fe9378fb92074106189fcbeb66e041b98`; hypothesis SHA256 `114f05fd37a073a3123c364db97e797c1c707a4655b40ae65670c755685f9829`; reliability source SHA256 `e017695027231e9ce08d8c363d61dbeac7baf0f33e421881ca852697783e92ff`.
- **Rationale:** The fixed gate requires whole-image ECE at most `0.04` and no whole-image ECE or NLL worsening within the 136-case lowest-reliability quintile. Weighting uses a frozen prediction-only OOF score; Dice and reference geometry do not enter weights. The resulting global temperature needs only the raw E08 logit at inference and is operationally eligible if it passes.
- **Alternatives considered:** Fit a new E08 reliability model before Stage 4; rejected as out-of-order analysis. Weight using observed Dice; rejected as target leakage. Use the final v1 in-sample reliability score; rejected in favor of the existing OOF score.
- **Who approved:** User through the approved uninterrupted C03–C09 sequence; Codex fixed score provenance, quintile membership, and the full gate before fitting.

## 2026-09-24T23:01:00+05:30 — Reject reliability-weighted calibration

- **Decision made:** Reject C05 under its unchanged conjunctive gate and close the experiment without a variant.
- **Triggering artifact:** `experiments/c05_reliability_weighted/artifacts/experiment_manifest.json`, SHA256 `d8c1d47725e1c96a06514f1b6e29c97410ed94646a73eb2da82b49699473790c`; 680 segmentation-validation cases and zero sealed-test images.
- **Rationale:** Whole-image ECE was `0.003427`, passing the `<=0.04` ceiling, and low-reliability-quintile whole-image NLL improved from `0.051441` to `0.042347`. However, low-quintile ECE worsened from `0.006481` to `0.007988`, so the mandatory no-worsening check failed. The calibration changed zero hard-mask pixels at threshold `0.50`.
- **Alternatives considered:** Change the reliability weighting function or quintile after seeing the result; rejected as an unregistered variant. Ignore the ECE failure because NLL improved; rejected because the gate is conjunctive.
- **Who approved:** Codex by the registered C05 rule; the user required honest closure without skipped or substituted experiments.

## 2026-09-24T23:08:00+05:30 — Register classifier-routed per-category calibration

- **Decision made:** Register C06 before fitting with one scalar temperature for each frozen classifier-predicted tumor route and a global fallback for the single no-tumor prediction.
- **Triggering artifact:** `experiments/c06_per_category/config.json`, SHA256 `302336b7e78689836da788f15a37340d78d3facc04a434d3ca85ef67c86572b7`; hypothesis SHA256 `6df20cf36cf0f6cfde98f20b848319f8ec5b5fee0e76b3167df433b34c05de61`; classifier predictions SHA256 `8654b76c30d5aa7eeb977cc7d5f79776d9adb0257386a187d90bfaea0d728344`.
- **Rationale:** The gate fixes nine conjunctive checks: ECE at most `0.05` in each of three regions for each predicted glioma, meningioma, and pituitary route. Routing uses classifier prediction and never true type. The single no-tumor route uses a global fallback and is descriptive because `n=1` cannot support a stable subgroup gate.
- **Alternatives considered:** Route by true label; rejected as target leakage. Fit a no-tumor temperature on one case; rejected as unstable. Omit region-specific checks; rejected in favor of an explicit interpretation fixed before fitting.
- **Who approved:** User through the approved uninterrupted C03–C09 sequence; Codex fixed routing, fallback, and gate semantics before fitting.

## 2026-09-24T23:17:00+05:30 — Reject classifier-routed per-category calibration

- **Decision made:** Reject C06 under its unchanged nine-check conjunctive gate and close it without a variant.
- **Triggering artifact:** `experiments/c06_per_category/artifacts/experiment_manifest.json`, SHA256 `85750c61fa8bda94a357eb577a45c9f24b4f2e823a619d169f00885e5348a294`; 680 segmentation-validation cases and zero sealed-test images.
- **Rationale:** Eight of nine predicted-route-by-region ECE checks passed. The predicted-glioma lesion-centred ECE was `0.055963`, exceeding the `0.05` ceiling. Predicted-meningioma and predicted-pituitary lesion-centred ECE values were `0.027328` and `0.033086`; all background and whole-image route values passed. The classifier-routed temperatures changed zero hard-mask pixels.
- **Alternatives considered:** Exclude the glioma lesion-centred check or gate only on whole-image ECE; rejected because all nine checks were fixed before fitting. Route using true type; rejected as leakage. Tune the glioma temperature after seeing the subgroup failure; rejected as an unregistered variant.
- **Who approved:** Codex by the registered C06 rule; the user required failures to remain documented findings.

## 2026-09-24T23:25:00+05:30 — Register nested hierarchical calibration

- **Decision made:** Register C07 before fitting as nested selection among seven fixed combinations of region split, reliability weighting, and classifier-predicted category routing.
- **Triggering artifact:** `experiments/c07_hierarchical/config.json`, SHA256 `9497ac7fcbd8db5781e7e7f449531789cec50ddbadcd9acfb78a95aa85fd5f9c`; hypothesis SHA256 `221dc4f1d72206483d0ab56d3a100b8dc6dfd646a25757b2e2fe0cf6c62da101`.
- **Rationale:** Candidate selection occurs strictly inside each outer training fold, ranked by inner lesion-centred ECE, NLL, Brier, then ID. The outer held-out fold supplies reported metrics. The conjunctive gate is aggregate lesion-centred ECE `<=0.03` plus lesion-centred ECE `<=0.05` within each predicted tumor route. Any selected configuration containing reference-region split `R` is oracle-only and H7-ineligible.
- **Alternatives considered:** Select using the already observed C04–C06 outer results; rejected as selection leakage. Route by true tumor type; rejected as target leakage. Use whole-image ECE as the primary selector; rejected because easy background would obscure the registered foreground calibration problem.
- **Who approved:** User through the approved uninterrupted C03–C09 sequence; Codex fixed candidates, nested selection, gate, and operational eligibility before fitting.

## 2026-09-27T09:01:37+05:30 — Accept hierarchical calibration locally as oracle-only

- **Decision made:** Accept C07 under its unchanged local gate, while marking it operationally ineligible because every outer fold selected a reference-region configuration.
- **Triggering artifact:** `experiments/c07_hierarchical/artifacts/experiment_manifest.json`, SHA256 `fc8f0c9d57dba806f20a65da90d4d9b918f7fd885f492cd7700acabde073d0fc`; 680 segmentation-validation cases and zero sealed-test images.
- **Rationale:** Nested outer-fold calibration achieved aggregate lesion-centred ECE `0.024577`, below the `0.03` gate. Predicted-route lesion-centred ECE was `0.034287` for glioma, `0.027104` for meningioma, and `0.034934` for pituitary, all below `0.05`. All four checks passed and zero hard-mask pixels changed. Inner selection chose `RC` for outer folds 1, 2, and 5 and `R` for folds 3 and 4; the final all-validation proposal also selected `RC`. Every selected configuration requires reference-mask routing and therefore cannot operate at inference or satisfy H7.
- **Alternatives considered:** Promote C07 because its local metrics passed; rejected by amendment 02's operational-input rule. Replace the nested selections with the best operational candidate after observing results; rejected as post-hoc selection. Treat the reference lesion region as available at inference; rejected as leakage.
- **Who approved:** Codex by the registered C07 gate and operational-eligibility rule; the user required fixed gates and fail-closed selection.

## 2026-09-27T09:04:05+05:30 — Register unified ECE comparison

- **Decision made:** Register C08 as a non-fitting integrity and standardization study before generating the comparison report.
- **Triggering artifact:** `experiments/c08_unified_ece/config.json`, SHA256 `160e0e967ca5778ceb70aee9be15ed8b3708ee8f1dc9e53e4fa0490730317524`; hypothesis SHA256 `5c192d9a312f288e34260724950283ecdd0bd29092df5590baef881e622a2833`.
- **Rationale:** C08 will verify all C01–C07 result hashes, require 15 non-empty equal-frequency bins with exact pixel coverage, and independently recompute every stored ECE within `1e-12`. It also standardizes the frozen classifier and accepted case-reliability ECE to 15 equal-frequency bins without changing any earlier decision.
- **Alternatives considered:** Refit the calibrators; rejected because C08 is descriptive and must not alter their outputs. Compare mixed bin definitions; rejected because the purpose is one fixed 15-bin definition.
- **Who approved:** User through the approved uninterrupted C03–C09 sequence; Codex fixed the integrity checks before reading consolidated outcomes.

## 2026-09-27T09:05:59+05:30 — Verify unified ECE comparison

- **Decision made:** Accept C08 as a verified descriptive standardization; leave every C01–C07 decision unchanged.
- **Triggering artifact:** `experiments/c08_unified_ece/artifacts/experiment_manifest.json`, SHA256 `62ed2aa80bcbf42a1e6eef49e66aaec7b86a1ffe86d3da5b71bfe21968c6ae19`; comparison report `audit/unified_ece_comparison.md`.
- **Rationale:** All seven source hashes matched. Forty-two segmenter summaries and four classifier summaries contained exactly 15 non-empty equal-frequency bins with complete observation coverage. Independently recomputed ECE differed from stored values by at most `6.94e-18`, below `1e-12`. The accepted v1 case-reliability score has standardized 15-bin ECE `0.031665` across 680 OOF cases.
- **Alternatives considered:** Refit calibrators or reinterpret failures under the unified table; rejected because C08 is descriptive. Use whole-image ECE alone for ranking; rejected because it is dominated by background pixels.
- **Who approved:** Codex by C08's registered integrity rule; the user authorized completion through C09 without relaxing earlier decisions.

## 2026-09-27T09:07:54+05:30 — Register per-category reliability diagrams

- **Decision made:** Register C09 before rendering six category-specific diagrams using the selected operational C03 calibrator and frozen v1 case-reliability OOF scores.
- **Triggering artifact:** `experiments/c09_category_diagrams/config.json`, SHA256 `ac770b919475a52cc4576b3160dc348e77b928a0f792efba9e6fbae1d0ea7bb2`; hypothesis SHA256 `130ca650deab45782d76999e38d57f084b58b0a8bb008ffac327c31df3fcc5a4`; C03 mappings SHA256 `fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa`.
- **Rationale:** C03 is the passing operational C01–C07 method with the lowest whole-image OOF ECE. Pixel diagrams use held-out fold mappings and the lesion-centred evaluation region. Case diagrams use existing OOF reliability scores. All diagrams use 15 equal-frequency bins.
- **Alternatives considered:** Use C07 because it has lower lesion-centred ECE; rejected because it needs reference-mask routing. Use true tumor type to route calibration; rejected as leakage. Fit category-specific curves; rejected because C09 is descriptive only.
- **Who approved:** User through the approved uninterrupted C03–C09 sequence; Codex applied the fixed operational ranking before rendering.

## 2026-09-27T09:10:12+05:30 — Verify per-category reliability diagrams

- **Decision made:** Accept C09 as a verified descriptive study and close Stage 3 experimentation pending consolidation review.
- **Triggering artifact:** `experiments/c09_category_diagrams/artifacts/experiment_manifest.json`, SHA256 `7e6b3d7bbe1f38b18320feddee3b87e161840db2c9bddf67674b9c5c52bd3628`; six PNG files under `audit/figures/`.
- **Rationale:** All registered hashes matched and six non-empty 15-bin OOF diagrams were produced. C03 lesion-centred ECE improved for meningioma (`0.036631` to `0.013591`) and pituitary (`0.037867` to `0.009881`) but worsened for glioma (`0.082850` to `0.085523`). Case-reliability ECE was `0.097247` for glioma, `0.020790` for meningioma, and `0.027838` for pituitary. The subgroup weakness qualifies the aggregate C03 pass and must remain explicit.
- **Alternatives considered:** Hide the glioma worsening because C03 passed in aggregate; rejected. Fit category-specific curves after seeing the diagrams; rejected because C09 is descriptive and C06 already closed under its fixed gate.
- **Who approved:** Codex by C09's registered integrity rule; the user required honest per-category reporting and no post-hoc variants.

## 2026-09-27T09:13:42+05:30 — Close Stage 3 calibration program

- **Decision made:** Close Stage 3 with C03 isotonic calibration as the selected operational E08 calibrator. Mark H7's calibration condition passed and its reliability condition pending Stage 4; do not nominate E08 or open the sealed test yet.
- **Triggering artifact:** `audit/stage3_calibration_report.json`, SHA256 `97d9c1dced6cd28fa68e1f552e8af38784fcd216aa3670df06dbc454e9b6460d`; human-readable report SHA256 `f56999605329d92808fc3925e05ebbf7ef53b87046bd66e2336096d18effc255`.
- **Rationale:** C01–C09 are complete with fixed outcomes. C03 passed its gate and ranks first among passing operational methods by whole-image OOF ECE. Its OOF transform changes 22,891 hard-mask pixels across 679 images, so later E08 reliability work must use the locked calibrated probabilities and masks. Glioma lesion-centred ECE worsens from `0.082850` to `0.085523`, which remains an explicit subgroup limitation.
- **Alternatives considered:** Select the better-calibrated C07 oracle; rejected because reference-mask routing is unavailable at inference. Keep raw E08 probabilities despite the registered selection rule; rejected because C03 passed and ranks first. Start Stage 4 automatically; rejected because the user required a pause after C09 and Stage 3 reporting.
- **Who approved:** Codex by amendments 01 and 02's fixed operational ranking; awaiting user review before Stage 4.

## 2026-09-27T18:45:14+05:30 — Register calibrated-E08 reliability ablation

- **Decision made:** Register amendment 03 and R01 before reconstructing calibrated-E08 reliability targets or fitting any Stage 4 estimator. Pause for user review before execution.
- **Triggering artifact:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`; R01 config SHA256 `92273d9fcb22f59b7d0b70c674981ff6fbc6f63b09fc51c0efdd08c7f6553eb1`; hypothesis SHA256 `8a039f1bbded90953801a1bdbd4235dee325c23a86646d722dbcd2bdc1969373`; implementation plan SHA256 `c0f9c4ac79b25f3fb640d10c73059f7e791563cb956295df55a8a7cc48a6dccf`.
- **Rationale:** Stage 4 must rebuild all features and targets from the locked C03 out-of-fold probabilities and changed masks. R01 fixes the fourteen prediction-only features, same five similarity-aware folds, full-fold ablations, 30-repeat held-out permutation importance, and 2,000 paired group-bootstrap intervals before any result is observed. R01 is descriptive and cannot be selected.
- **Alternatives considered:** Reuse raw-E08 or v1 reliability features and targets; rejected because amendment 02 locks Stage 4 to selected C03 outputs. Use reference geometry or Dice as features; prohibited. Run R01 before plan review; rejected because the user explicitly requires approval after registration and planning.
- **Who approved:** User explicitly initiated Stage 4 and specified the R01 scope; execution remains pending the requested plan approval.


## 2026-09-27T19:00:39+05:30 — Complete calibrated-E08 reliability feature ablation

- **Decision made:** Accept R01 as `DESCRIPTIVE_COMPLETE`. R01 has no selectable performance gate; preserve all fourteen-feature, ablation, permutation, coefficient, calibration, subgroup, and verification outputs as the registered evidence for R02 planning.
- **Triggering artifact:** `experiments/r01_ablation/artifacts/experiment_manifest.json`, SHA256 `7e7162d4d1b33349c7944a8d8045898dbe37966d2c407b5feec556dbcfa51a7a`; 680 internal-validation cases, 583 similarity groups, 2,000 paired group-bootstrap resamples, 30 held-out permutations per feature, and zero sealed-test images.
- **Rationale:** The fourteen-feature OOF baseline achieved failure-detection AUROC `0.954503` (95% group-bootstrap CI `0.937618–0.969855`), ECE `0.017072`, Brier `0.053708`, Brier skill `0.494401`, and captured `18/19` zero-Dice cases in the 136 highest-risk cases. Boundary entropy was the only feature whose removal reduced AUROC by at least `0.005` (`0.007872`) and had the largest permutation importance (`0.265029`, 95% CI `0.240437–0.289786`). Glioma remained the weakest true-label subgroup (AUROC `0.873916`). The feature API and pinned source hash confirmed all fourteen inputs derive only from the C03 OOF probability map and threshold-0.50 prediction; reference masks supplied targets/evaluation only, and classifier probabilities never entered the model.
- **Verification:** C03 mapping SHA256 matched; exactly `22,891` changed pixels across `679` images were reproduced; all five fold assignments matched; no similarity group crossed folds; 106 tests passed; Ruff passed. An initial pre-result execution stopped on a defensive classifier CSV label-field schema check before writing outputs; the check was corrected from the nonexistent `true_label` field to the file's `label` field, with no change to registered inputs, analysis, gates, or thresholds.
- **Alternatives considered:** Select a reduced model directly from R01; rejected because R01 is descriptive and R02 requires separate preregistration. Interpret permutation importance as causal; rejected because correlated features can share or mask importance. Treat aggregate AUROC as uniform; rejected because glioma subgroup performance is materially weaker.
- **Who approved:** User explicitly approved R01 execution after verifying registration commit `6c43af4`; Codex applied the registered descriptive rule and fail-closed checks.


## 2026-09-27T19:11:52+05:30 — Register reduced calibrated-E08 reliability model

- **Decision made:** Register R02 before fitting with the exact ten-feature candidate list produced by committed R01. Pause for user approval before execution.
- **Triggering artifact:** `audit/preregistration_amendment_04.md`, SHA256 `1603be8a4d6b4415ebac4162260968def6b9c539ec03bb67c18d7bb4016f9629`; R02 config SHA256 `69715d2cd6e8a581380d792f2d99de1b0bef3aed1e3e0970db625d2c6664c14e`; hypothesis SHA256 `bef3f90276d1005f7ed6346bda622d05408affe695d84e2831e8feba002167ba`; implementation plan SHA256 `403ca997efbe0a015fd2e6c515c63c7b6c2f9bf61c54af43c28c7a1902276497`; R01 result SHA256 `d53b160a7aec1ab2c8b0d612576530cdcc530e1b7b30bb789d3e49bee3d9499d`.
- **Rationale:** The locked candidates are `empty_prediction`, `predicted_area_fraction`, `whole_probability_mean`, `whole_probability_sd`, `predicted_region_entropy_mean`, `boundary_entropy_mean`, `high_confidence_foreground_fraction`, `uncertain_pixel_fraction`, `bounding_box_fill_ratio`, `normalized_perimeter`. R02 uses the same 680 C03-calibrated cases, five similarity-aware folds, threshold `0.50`, estimator, 2,000-resample similarity-group bootstrap, and full R01 metric family. The fixed noninferiority gate is reduced OOF AUROC at least `0.9495028142589118`, equal to the committed fourteen-feature baseline `0.9545028142589118` minus the maximum allowed loss `0.005`.
- **Alternatives considered:** Select features by permutation importance after R01; rejected because the committed R01 candidate screen used ablation delta or five-fold coefficient-sign stability. Change the AUROC margin or add post-result calibration gates; rejected because the user fixed the `0.005` gate. Fit before plan review; rejected because the user required a registration-and-plan pause.
- **Who approved:** User explicitly accepted R01 and authorized R02 registration and planning; execution remains pending approval.


## 2026-09-27T19:23:53+05:30 — Accept reduced calibrated-E08 reliability model

- **Decision made:** Accept R02 as `PASS` under the unchanged noninferiority gate. Preserve the registered ten-feature model as an eligible Stage 4 candidate and pause before R03.
- **Triggering artifact:** `experiments/r02_reduced_features/artifacts/experiment_manifest.json`, SHA256 `1dc386a3cd342f14fcb827960c1ec89f749b2413962c768aec1c22bdf74cb819`; 680 internal-validation cases, 583 similarity groups, 2,000 paired group-bootstrap resamples, and zero sealed-test images.
- **Rationale:** The fourteen-feature baseline reproduced exactly at AUROC `0.954502814259` with aggregate difference `0` and maximum per-case OOF score difference `0`. The reduced ten-feature model achieved AUROC `0.955502080104` (95% CI `0.938715915158–0.970421402184`), exceeding the fixed minimum `0.949502814259`. Reduced-minus-full AUROC was `0.000999265846`, with paired 95% CI `0.000066687465–0.001998220021`. Reduced-model ECE was `0.017133`, Brier was `0.052757`, Brier skill was `0.503348`, and the top 20% risk captured `18/19` zero-Dice cases. Glioma remained the weakest true-label subgroup at AUROC `0.876843`.
- **Verification:** All registered hashes matched; the exact ten candidates and order matched R01; exactly 22,891 C03-changed pixels across 679 images were reproduced; all five folds and 583 similarity groups were verified without cross-fold group leakage; all ten selected features were inference-time-only; reference-mask, classifier-probability, and raw-image feature leakage were absent; threshold remained `0.50`; no classifier or neural segmenter was retrained; 109 tests passed; Ruff passed.
- **Alternatives considered:** Reject the model because its gain is small; rejected because the registered gate tests noninferiority, which passed. Claim broad superiority from the paired interval; rejected because R02 is an internal development comparison on the same population that informed R01 selection. Start R03 immediately; rejected because the user required a pause after R02.
- **Who approved:** User explicitly approved R02 execution after verifying registration commit `d250135`; Codex applied the registered gate and fail-closed rules.


## 2026-09-27T19:35:59+05:30 — Register extended calibrated-E08 reliability model

- **Decision made:** Register R03 before extracting any new feature or fitting any R03 estimator. Pause for user approval before execution.
- **Triggering artifact:** `audit/preregistration_amendment_05.md`, SHA256 `a5c8ff14dea6349c7b3be334e27edf6323f1e0e23a772f530223e89f6bfc03a3`; R03 config SHA256 `0c8b02747534d511d6ee2aad8d797d08a70f58843a61005afcb8ec391bedb4bc`; hypothesis SHA256 `91d1512ca11e468545a3cebf7723848a6532f6735bf5d3ddca725ae903299c58`; implementation plan SHA256 `77ef13b52b928194c984b83657d7ff185538404bbc1013a989260b94b33ea635`; R01 result SHA256 `d53b160a7aec1ab2c8b0d612576530cdcc530e1b7b30bb789d3e49bee3d9499d`.
- **Rationale:** R03 appends exactly six locked features to the unchanged fourteen-feature R01 baseline: mask circularity, solidity, boundary fractal dimension, predicted-boundary image-gradient mean, predicted-centroid distance to image centre, and vertical symmetry. Five use only the threshold-0.50 prediction. The gradient feature uses the uploaded image after the exact registered grayscale 128×128 letterbox transform plus the predicted boundary; it never reads a reference mask. The fixed superiority gate is extended OOF AUROC at least `0.9595028142589118`, equal to baseline `0.9545028142589118` plus `0.005`.
- **Alternatives considered:** Extend the passing ten-feature R02 model; rejected because Amendment 03 and the user fix R03 against the fourteen-feature baseline. Tune box-counting scales or image preprocessing after observing results; prohibited, so exact formulas and empty-mask behavior are frozen now. Use reference-boundary image gradients; rejected as inference-time leakage.
- **Who approved:** User explicitly accepted R02 and authorized R03 registration and planning; execution remains pending approval.


## 2026-09-27T19:57:17+05:30 — Correct and verify exact R03 image preprocessing dtype

- **Decision made:** Register Amendment 06 before R03 feature extraction, correcting the model-space image conversion to `float32` before division by 255 and pinning both preprocessing implementations. Keep every R03 scientific gate and feature formula unchanged.
- **Triggering artifact:** `audit/preregistration_amendment_06.md`, SHA256 `47ee34350db79c44d0a5d74bf9cb809029b8781f930959c030705e03e4e4b394`; corrected R03 config SHA256 `69e84560a741ca75a85812f5faadbd817da5d8689c1985c3cbfebd629e05786b`; R03 feature contract SHA256 `da3b1666adb644dead7f1588b9c8434de2b18732755f470eaf8b3be00943e3b9`; frozen segmenter preprocessing SHA256 `defa28dc8f94dc6fd3c41f5928fba1dbb6959d685acc32972711526ce3341dca`.
- **Rationale:** The pre-run pixel-regression test found that Amendment 05's proposed float64-before-division ordering differed from the segmenter's float32 model input in 9,216/16,384 synthetic pixels, with maximum absolute difference approximately `2.97e-8`. After correction, the feature transform matches `segmentation_smoke_test.transform_pair` exactly pixel for pixel. No R03 feature or model result existed when this correction was registered.
- **Alternatives considered:** Accept the numerically tiny difference; rejected because the user explicitly required identical preprocessing. Change the segmenter preprocessing; rejected because the segmenter is frozen. Continue without recording the correction; rejected by fail-closed governance.
- **Who approved:** User's explicit requirement that boundary-gradient preprocessing exactly match the segmenter authorized this prospective correction; R03 execution remains pending completion of this verification commit.
