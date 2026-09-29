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


## 2026-09-27T20:03:37+05:30 — Reject extended calibrated-E08 reliability model

- **Decision made:** Reject R03 as `FAIL` under the unchanged superiority gate. Preserve the complete negative result and pause before R04 without testing feature subsets or variants.
- **Triggering artifact:** `experiments/r03_extended_features/artifacts/experiment_manifest.json`, SHA256 `5ef0b58c58a843867ff4416cca4edfa56979285123c92ef942b7523817519aee`; 680 internal-validation cases, 680 verified source-image hashes, 583 similarity groups, 2,000 paired group-bootstrap resamples, 30 held-out permutations for each of six new features, and zero sealed-test images.
- **Rationale:** The fourteen-feature baseline reproduced exactly at AUROC `0.954502814259`, with aggregate and maximum per-case OOF differences both `0`. The twenty-feature model achieved AUROC `0.951423444000` (95% CI `0.931440479745–0.969122650872`), below the fixed minimum `0.959502814259`. Extended-minus-baseline AUROC was `-0.003079370259`, with paired 95% CI `-0.014050581511–0.005847072724`. ECE was `0.017258`, Brier was `0.052866`, Brier skill was `0.502324`, and top-20% risk capture remained `18/19`. Glioma AUROC was `0.881009`, while pituitary fell to `0.919831`. Solidity had the largest new-feature permutation importance (`0.009592`, CI `0.003807–0.015968`), but no leave-one-new-feature-out ablation recovered the registered improvement.
- **Verification:** Amendments 05 and 06 and every registered source hash matched; the image-only transform matched the segmenter pixel for pixel; all 680 source images matched the split-snapshot SHA256; exactly 22,891 C03-changed pixels across 679 images were reproduced; all folds and similarity groups were valid; all six new features were inference-time computable without reference masks or classifier probabilities; threshold remained `0.50`; no classifier or neural segmenter was retrained; 115 tests passed; Ruff passed.
- **Alternatives considered:** Select only solidity or symmetry after observing importance; rejected as an unregistered post-result variant. Relax the gate because Brier skill improved; rejected because the registered gate is AUROC superiority. Promote the model for its glioma subgroup increase; rejected because aggregate and pituitary discrimination worsened and the conjunctive selection protocol requires the experiment gate.
- **Who approved:** User explicitly approved R03 execution after verifying registration commit `458add4`; Amendment 06 prospectively enforced the user's exact-preprocessing requirement before any R03 outcome existed; Codex applied the unchanged registered gate.


## 2026-09-27T20:17:54+05:30 — Register per-category calibrated-E08 reliability models

- **Decision made:** Register Amendment 07 and R04 before fitting any per-category estimator. Pause for user approval before execution.
- **Triggering artifact:** `audit/preregistration_amendment_07.md`, SHA256 `cd960ba0167f3982722667651a2dd0da12e2c2437160df7b6c0cee1f65be2a40`; R04 config SHA256 `e341b646cfc8c75f42e97780eaf7f74437e4981fc85630b1dbe13793aac516cd`; hypothesis SHA256 `6789167ea919594404eb8423118a0a655f03a7126ff9e3b573bc36aa5f3d750d`; implementation-plan SHA256 `1a2b7b1f3e0c675c41ac763b15671d69ff759d8488b8ee4ff67557f8221038f6`; R02 result SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`.
- **Rationale:** R04 fixes two independent five-fold analyses using the R02 ten-feature contract: true-category models as a non-operational descriptive comparator and frozen-classifier-predicted-category models as the operational form. The one predicted `no_tumor` case is excluded from Version B without fallback. The experiment passes only if all six category/version AUROCs are at least `0.80`. Comparisons against R02 will explicitly test whether glioma improves and whether pituitary recovers or degrades after R03's confirmed pituitary sensitivity.
- **Alternatives considered:** Use true category as the operational router; rejected as unavailable at inference. Pool the one predicted `no_tumor` case into a tumor route; rejected as post-hoc routing. Reuse the rejected R03 feature set; rejected because R04 is locked to the passing R02 ten-feature model. Gate only the operational models; rejected because the user explicitly required both versions to pass all three categories.
- **Who approved:** User accepted R03 and explicitly authorized R04 registration and planning; execution remains pending approval.


## 2026-09-27T20:42:45+05:30 — Accept per-category reliability models under local gate

- **Decision made:** Accept R04 as `PASS` under its unchanged six-check conjunctive gate. Preserve the result as a locally eligible Stage 4 candidate while recording that every per-category fit degraded relative to the identical-case R02 aggregate ten-feature model. Pause before R05.
- **Triggering artifact:** `experiments/r04_per_category/artifacts/experiment_manifest.json`, SHA256 `b5180fb40c4663502c41cce03e83f174721897b74f145d8519c677287f47a047`; result SHA256 `e47d380024953f9e77be1d1916f4eeb0f103a72a12070ecc3a5708d43fc39046`; 680 internal-validation cases, 583 similarity groups, 2,000 group-bootstrap resamples, 679 operationally routed cases, and zero sealed-test images.
- **Rationale:** Version A true-category AUROCs were glioma `0.857368`, meningioma `0.845455`, and pituitary `0.925457`. Version B classifier-predicted-route AUROCs were glioma `0.867716`, meningioma `0.902973`, and pituitary `0.957851`. All six point estimates exceeded the registered `0.80` minimum, so the conjunctive gate passed. However, Version A changed by `-0.019475`, `-0.148701`, and `-0.031646` versus R02 in glioma, meningioma, and pituitary; Version B changed by `-0.003263`, `-0.093897`, and `-0.006061`. Per-category fitting therefore did not improve glioma and did not recover pituitary beyond R02. The frozen classifier routed 679/680 cases to tumor models, with four-class accuracy `0.972059` and tumor-route accuracy `0.973490`; its single `no_tumor` prediction was excluded without fallback.
- **Verification:** Every pinned hash matched; R02 per-case OOF reliability reproduced with maximum difference `0`; all six R02 subgroup comparators reproduced within `1e-12`; exactly 22,891 C03-changed pixels across 679 images reproduced; all category training partitions contained both outcomes; all ten features were prediction-derived; true category entered Version A grouping and route evaluation only; reference masks and classifier probabilities did not enter either reliability estimator. An initial pre-result invocation stopped before loading probability data because the new runner had not forwarded two already-pinned C03 cache keys from the verified R01 config. The wiring was corrected without changing any protocol, feature, fold, gate, or result; no output existed before the correction. Independent artifact recomputation passed, 119 tests passed, and Ruff passed.
- **Alternatives considered:** Claim category-specific fitting as an improvement because the local gate passed; rejected because all paired point differences versus R02 were negative. Promote true-category Version A operationally; rejected because true type is unavailable before inference. Add a fallback for the excluded `no_tumor` route or tune category models after seeing the degradation; rejected as unregistered post-hoc variants.
- **Who approved:** User explicitly approved R04 execution after verifying registration commit `5e94b42`; Codex applied the registered point-estimate gate and fail-closed checks.


## 2026-09-27T21:00:07+05:30 — Register category-aware reliability model

- **Decision made:** Register Amendment 08 and R05 before fitting the category-aware estimator. Pause for user approval before execution.
- **Triggering artifact:** `audit/preregistration_amendment_08.md`, SHA256 `3864f8dab879d8bf7821b91551127380fd636d1b68a170193721ccba860223c0`; R05 config SHA256 `94afd0cec55e1bc11c973f5a81ce17b95fc5f3a622b3a9f605e5c2a72f71cabe`; hypothesis SHA256 `9c313b8cc28adfa74d37ea04bbc72c4e0803faf388bea0d5c0bce34ae014ef5d`; implementation-plan SHA256 `cb36bfe2d8572e0e5cb0f4f55f1cae0296a85d76f237904302b8b0f9b3b6655f`; R02 result SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`; R04 result SHA256 `e47d380024953f9e77be1d1916f4eeb0f103a72a12070ecc3a5708d43fc39046`.
- **Rationale:** R05 tests one model with the passing R02 ten-feature vector plus three frozen-classifier-predicted-category indicators. True type and classifier probabilities are prohibited inputs. The single predicted `no_tumor` case is excluded from both fitting and evaluation, leaving 679 cases. The fixed gate is predicted-glioma AUROC at least `0.88`; overall and other subgroup metrics are descriptive. Published and identical-cohort R02/R04 comparators are both locked so the exclusion cannot silently distort comparisons.
- **Alternatives considered:** Fit separate category models again; rejected because R04 showed consistent degradation. Use true-category one-hot indicators; rejected as inference-time leakage. Encode the excluded `no_tumor` case as all zeros or add a fourth indicator; rejected because the user explicitly required exclusion. Drop one tumor indicator as a reference category; rejected in favor of the explicitly registered full three-column one-hot representation with L2 regularization.
- **Who approved:** User accepted R04 and explicitly authorized R05 registration and planning; execution remains pending approval.


## 2026-09-27T21:09:49+05:30 — Reject category-aware reliability model

- **Decision made:** Reject R05 as `FAIL` under its unchanged predicted-glioma gate. Preserve the complete negative result, retain R02 as the current operational reliability choice, and pause before R06 without trying category-encoding variants.
- **Triggering artifact:** `experiments/r05_category_aware/artifacts/experiment_manifest.json`, SHA256 `237d8dbfeef592fa136ed513458161058acb6a98b32a6f79cc518dc34c6d8415`; result SHA256 `3f08ad6dde731209c53d84c4b01c6709bcec64dca536b801cd4e78d94abf2d8d`; 679 included internal-validation cases, 583 source similarity groups, 2,000 group-bootstrap resamples, and zero sealed-test images.
- **Rationale:** Classifier-predicted-glioma AUROC was `0.871096` (95% group-bootstrap CI `0.817977–0.920612`), below the fixed `0.88` minimum, so R05 failed. Its change versus R02 predicted glioma was only `+0.000117`, with paired CI `-0.003680–0.004092`. Overall AUROC was `0.954808`, essentially matching but slightly below matched-cohort R02 (`-0.000206`, paired CI `-0.003486–0.002516`). Predicted meningioma was `0.993740`, and predicted pituitary was `0.965014`; neither changes the failed glioma gate. True-category AUROCs were glioma `0.875493`, meningioma `0.992424`, and pituitary `0.958509`. Category indicators therefore did not solve the registered glioma weakness.
- **Verification:** Every pinned hash matched; the R02 per-case OOF vector reproduced with maximum difference `0`; every R02 and R04 overall/subgroup comparator reproduced exactly; exactly 22,891 C03-changed pixels across 679 images reproduced; the registered no-tumor case alone was excluded; all 679 included one-hot rows summed to one; all training partitions contained both outcomes; true category, reference-mask values, and classifier probabilities did not enter the model. Independent recomputation verified all seven AUROCs, 30,000 bootstrap rows, 13 coefficients, and every manifest hash. 123 tests passed and Ruff passed.
- **Alternatives considered:** Lower the `0.88` gate because the confidence interval includes it; rejected because the registered point estimate controls. Drop one category indicator, rescale indicators separately, or retain the no-tumor case after seeing the result; rejected as unregistered post-hoc variants. Prefer R05 because it substantially improves over R04; rejected because R02 is the governing aggregate comparator and R05 did not improve it or pass its own gate.
- **Who approved:** User explicitly approved R05 execution after verifying registration commit `bbd1288`; Codex applied the unchanged registered gate and fail-closed checks.


## 2026-09-27T21:17:51+05:30 — Register nonlinear reliability comparison

- **Decision made:** Register Amendment 09 and R06 before installing XGBoost or fitting any nonlinear estimator. Pause for user approval before environment mutation or execution.
- **Triggering artifact:** `audit/preregistration_amendment_09.md`, SHA256 `84305f4e04404d238407cc8650317d111aa871c72b67d62844cdd7ece921cdb7`; R06 config SHA256 `7bb7b3de352755b9bf229a94b8ff447c0f5a811fb4316bd8f90cff5e6fe50f43`; hypothesis SHA256 `db419877de241c371cedab63f05d0b84871dbe1f32b32f20b37bb1c6c991ced0`; implementation-plan SHA256 `d156ce4f5dfd1b551ad38845ec8e5897092de3500ac86308e42fb910c6af0157`; R02 result SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`; R05 result SHA256 `3f08ad6dde731209c53d84c4b01c6709bcec64dca536b801cd4e78d94abf2d8d`.
- **Rationale:** R06 fixes XGBoost 3.1.2 with 200 shallow trees, a fold-scaled 2×32 MLP, and a 100-tree random forest, all on the exact R02 ten-feature contract and five similarity-aware folds. The user-specified controlling gate is absolute AUROC at least `0.9750`; exact change from the reproduced 680-case R02 AUROC `0.9555020801` will also be reported. Native tree importance and 30-repeat held-out permutation importance are fixed before outcomes. If no nonlinear model passes, R02 remains the operational choice for interpretability.
- **Alternatives considered:** Substitute scikit-learn gradient boosting because XGBoost was absent; rejected because the requested models name XGBoost or LightGBM. A dry-run verified `xgboost==3.1.2` resolves, but installation is deferred until approval. Tune hyperparameters or use early stopping; rejected to avoid validation-driven search. Shift the absolute gate to `0.975502` using the exact 680-case comparator; rejected because the user explicitly fixed `0.9750` as the minimum, while the exact delta remains a mandatory report.
- **Who approved:** User accepted R05 and explicitly authorized R06 registration and planning; dependency installation and execution remain pending approval.


## 2026-09-27T23:41:43+05:30 — Reject nonlinear reliability models

- **Decision made:** Reject R06 as `FAIL` because no nonlinear model reached the unchanged `0.9750` gate. Retain R02 logistic regression as the operational reliability model and pause before R07 without tuning or retrying nonlinear variants.
- **Triggering artifact:** `experiments/r06_nonlinear_comparison/artifacts/experiment_manifest.json`, SHA256 `bafd8a2076564ea0bb563e47e6814660ddc95098ce976b154b2889c828aed634`; result SHA256 `aa09898ee9e46f4d3edeecae34ec87d90d7107416b95376c2a7dee7a77eefb73`; 680 internal-validation cases, 583 similarity groups, 2,000 paired group-bootstrap resamples, 900 held-out permutation runs, and zero sealed-test images.
- **Rationale:** R02 reproduced at AUROC `0.955502`. XGBoost achieved `0.940268` (95% CI `0.918127–0.959850`), a `-0.015234` change whose paired CI was `-0.027130–-0.004290`. Random forest achieved `0.935252` (CI `0.909057–0.958358`), a `-0.020250` change with paired CI `-0.037057–-0.005736`. The MLP achieved `0.906181` (CI `0.869755–0.939232`), a `-0.049321` change with paired CI `-0.077640–-0.023558`. All three were statistically worse than R02 under the paired development bootstrap and captured fewer zero-Dice cases (`17/19`, `17/19`, and `15/19` versus R02's `18/19`). Boundary entropy was the strongest permutation feature for all three models. Nonlinearity did not justify loss of coefficient-level interpretability.
- **Verification:** XGBoost 3.1.2 was locked in environment commit `38ee068` before fitting. Every pinned hash matched; R02 per-case OOF scores and AUROC reproduced with difference `0`; exactly 22,891 C03-changed pixels across 679 images reproduced; all folds and groups were valid; all probabilities were finite and in `[0,1]`; all ten features were prediction-derived. Independent recomputation verified 28 overall/subgroup AUROCs, 14,000 bootstrap rows, 100 native-importance rows, 900 permutation rows, and every manifest hash. The MLP reached 1,000 iterations with convergence warnings in folds 1 and 2; these were recorded prospectively and all probabilities remained valid. 126 tests passed and Ruff passed.
- **Alternatives considered:** Tune depth, learning rate, tree count, forest leaf size, MLP regularization, or early stopping after seeing the result; rejected as prohibited post-hoc search. Retain XGBoost because it was the best nonlinear model; rejected because it was significantly worse than R02 and failed its absolute gate. Claim native importance as causal explanation; rejected because gain and impurity can be biased and permutation importance remains associational.
- **Who approved:** User explicitly approved R06 execution after verifying registration commit `b1a23c1`; Codex applied the unchanged registered gate and fail-closed checks.

## 2026-09-27T23:50:02+05:30 — Register coefficient-bootstrap stability analysis

- **Decision made:** Register Amendment 10 and R07 before any bootstrap coefficient refit. Pause for user approval before execution.
- **Triggering artifact:** `audit/preregistration_amendment_10.md`, SHA256 `2f96675d8616605cb9794928e4b7dfa1fd9a76361f09d9ddcd802fa7f86ed4cc`; R07 config SHA256 `72509d20a2995451dc2efe1a3bae2569cbd64797b130931a52bc970985216767`; hypothesis SHA256 `376cefcb56bc33ee846c84d214e5141d8a5ea0e8e430cb656b9718273f967226`; implementation-plan SHA256 `3870fafd71eb743b84a39a2b8ac46926a8b184e1aedebe2a9144ee30592e6ab4`; R02 result SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`; R06 result SHA256 `aa09898ee9e46f4d3edeecae34ec87d90d7107416b95376c2a7dee7a77eefb73`.
- **Rationale:** R07 fixes exactly 2,000 seed-42 similarity-group resamples of the accepted R02 ten-feature standardized logistic model. Every draw refits both scaler and estimator; invalid draws stop rather than being replaced. The analysis reports percentile coefficient intervals, five-fold and bootstrap sign agreement, ten coefficient distributions, and a shared-feature comparison with R01. R07 is descriptive, has no performance gate, cannot change model selection, and does not give causal meaning to coefficients.
- **Alternatives considered:** Bootstrap individual images; rejected because known similarity groups are the independence unit. Hold the scaler fixed; rejected because deployment of the registered pipeline refits preprocessing with the estimator. Use Wald intervals; rejected in favor of the user-requested empirical group-bootstrap percentiles. Use out-of-fold predictions to infer coefficients; rejected because coefficients must come from direct registered model fits.
- **Who approved:** User accepted R06 and explicitly authorized R07 registration and planning; bootstrap execution remains pending approval.

## 2026-09-27T23:59:21+05:30 — Complete coefficient-bootstrap stability analysis

- **Decision made:** Accept R07 as `DESCRIPTIVE_COMPLETE`. Preserve all coefficient intervals and sign-stability results; retain R02 unchanged as the operational reliability model and continue to R08.
- **Triggering artifact:** `experiments/r07_coefficient_bootstrap/artifacts/experiment_manifest.json`, SHA256 `afddd3023655ec9fc539611a50eef3335d36923894e49aa99e831f9743e5dd10`; result SHA256 `fc8b1bd80826913ff990ea993bf1c0c8bdf028b614101f967afc4282c01cb948`; audit report SHA256 `3e4ebe68b528134f7cd8b1adb55a887d5734859d38fce6b9f993e803f751069e`; 680 internal-validation cases, 583 similarity groups, exactly 2,000 valid group-bootstrap refits, and zero sealed-test images.
- **Rationale:** Boundary entropy remained the dominant standardized coefficient (bootstrap mean `-2.1424`, 95% interval `[-2.8973,-1.3756]`) with 2,000/2,000 sign agreement. Predicted area, uncertain-pixel fraction, and normalized perimeter also had intervals excluding zero. Whole-probability mean, empty prediction, whole-probability SD, high-confidence foreground fraction, predicted-region entropy, and bounding-box fill ratio were less stable because their percentile intervals touched or crossed zero. Every shared feature retained its mean sign from R01 to R02, and all ten had 5/5 fold sign agreement. This is descriptive associational evidence, not a causal claim or a new model-selection gate.
- **Verification:** Every pinned hash matched; exactly 22,891 C03-changed pixels across 679 images reproduced; the committed R02 fold coefficients reproduced with maximum difference `0`; all 20,000 stored bootstrap coefficients independently reproduced their means, SDs, and percentile limits; every manifest hash matched; all features were prediction-derived; the ten-panel figure was visually checked. 128 tests passed and Ruff passed.
- **Alternatives considered:** Treat every 5/5 fold sign as fully stable; rejected because six group-bootstrap intervals include zero. Remove unstable features or refit R02; rejected because R07 is descriptive and cannot change selection. Interpret negative coefficients causally; rejected because predictors are correlated and the analysis is observational.
- **Who approved:** User verified registration commit `31725be` and explicitly approved R07 execution; Codex applied the registered descriptive completion rule.

## 2026-09-28T00:02:09+05:30 — Register prospective reliability fold evaluation

- **Decision made:** Register Amendment 11 and R08 before computing held-out-fold metrics. Proceed under the user's standing R07–R10 execution approval.
- **Triggering artifact:** `audit/preregistration_amendment_11.md`, SHA256 `79e3e5d49230b4ec0409f98842b2e40d836981aa82e34e6269a7589f58d9ad7a`; R08 config SHA256 `bdfc113ee0197f9cae26dadd32bbd72d96a004e10e98ef8eab366e214a3883b1`; hypothesis SHA256 `3f1d23fc1754df044b5e960d09cf770a16b9c30791ec60086260d8583953093e`; implementation-plan SHA256 `f75d1c9ae211bc08223d2aff5e5943efa2b5a40e49e6b653a0bb722a460b21ac`; R02 result SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`; R07 result SHA256 `fc8b1bd80826913ff990ea993bf1c0c8bdf028b614101f967afc4282c01cb948`.
- **Rationale:** R08 fixes five train-four/evaluate-one similarity-aware folds using the unchanged R02 pipeline. The controlling H7 gate is the unweighted mean of the five held-out-fold failure-detection AUROCs at least `0.941`; aggregate OOF AUROC is supporting evidence rather than a substitute. Fold ECE, Brier skill, a 2,000-draw group-bootstrap interval, true-category AUROCs, and zero-Dice capture are mandatory.
- **Alternatives considered:** Use aggregate OOF AUROC as the gate; rejected because the user explicitly requested mean fold AUROC. Weight fold means by case count; rejected in favor of the specified arithmetic mean. Fit one model on all 680 before evaluation; rejected because reported metrics must remain out-of-fold.
- **Who approved:** User explicitly approved the complete R07–R10 sequence, with a stop after R08 only if failure changes later registrations.

## 2026-09-28T00:06:54+05:30 — Accept prospective reliability fold evaluation

- **Decision made:** Accept R08 as `PASS` under the unchanged H7 mean-fold gate. Continue to R09 because the defined R08 failure stop condition did not occur.
- **Triggering artifact:** `experiments/r08_prospective_fold_evaluation/artifacts/experiment_manifest.json`, SHA256 `f96f296e07f8573898d5ff5840dafa78b8f26a63815bf4c35d2e2174156fb1b9`; result SHA256 `8e6ea05a18937949aa55aa4d285e535047e3c1dba3a5a98bc12b7b3e523f34a4`; audit report SHA256 `d9479d6b45e6138d6a92753cc18995acfc2cb41720f53af0e793554fce5f71e0`; 680 internal-validation cases, 583 similarity groups, 2,000 group-bootstrap resamples, and zero sealed-test images.
- **Rationale:** The five held-out-fold failure-detection AUROCs were `0.9831`, `0.9256`, `0.9771`, `0.9574`, and `0.9590`. Their unweighted mean was `0.9604` (SD `0.0225`), exceeding the registered `0.941` minimum. Aggregate OOF AUROC reproduced R02 at `0.9555` (95% group-bootstrap CI `0.9387–0.9704`), with 18/19 zero-Dice cases captured in the highest-risk 20%. True-category AUROC remained weakest for glioma at `0.8768`; fold 2 also remained a visible weak partition.
- **Verification:** Every pinned hash matched; the full 680-case R02 OOF vector and aggregate AUROC reproduced with difference `0`; group isolation held; exactly 22,891 C03-changed pixels across 679 images reproduced; independent recomputation matched every fold AUROC and its mean/SD; every manifest hash matched. 130 tests passed and Ruff passed.
- **Alternatives considered:** Fail R08 because fold 2 was below 0.941; rejected because the preregistered controlling statistic is the unweighted five-fold mean. Substitute the aggregate AUROC for the gate; rejected because it is supporting evidence only. Hide fold dispersion after the mean passed; rejected, so fold 2 and the SD remain explicit.
- **Who approved:** User authorized R08 execution under the standing R07–R10 sequence; Codex applied the fixed gate without relaxation.

## 2026-09-28T00:09:32+05:30 — Register alternative-fold reliability sensitivity

- **Decision made:** Register Amendment 12 and R09 before generating any alternative folds. Proceed under standing approval after committing this registration.
- **Triggering artifact:** `audit/preregistration_amendment_12.md`, SHA256 `5902598fae407dcd56bb0084c01098974e0cc78e20fb406d8d733dbf7015ec7a`; R09 config SHA256 `3ec7b2ff7122e42d7db1ec5dd343787475ab6c86edbe2070b86d9591fb07ef43`; hypothesis SHA256 `0b8ee2debe1865800f6158d7ad7e7439bcaa58ba77d2e71aa0ef6168ced8f3a1`; implementation-plan SHA256 `6076e757b34c2bdd60635d372542bb4b0f16e284a31df853fe272be97532dd9d`; R08 result SHA256 `8e6ea05a18937949aa55aa4d285e535047e3c1dba3a5a98bc12b7b3e523f34a4`.
- **Rationale:** R09 fixes five-fold `StratifiedGroupKFold` assignments at seeds 17, 42, and 93, using the locked reliability target for stratification and similarity groups for isolation. Each alternative produces complete OOF metrics and true-category AUROCs. Paired 2,000-draw group bootstraps compare each AUROC with original R02. A preregistered `NARROW/MODERATE/MATERIAL` dispersion label supports a clear descriptive conclusion but cannot change selection.
- **Alternatives considered:** Use arbitrary new seeds; rejected because the Stage 4 parent already fixed 17, 42, and 93. Stratify by tumor type; rejected because the registered split class is the reliability target. Select the best alternative split; rejected because R09 measures sensitivity and is not model selection.
- **Who approved:** User authorized the complete R07–R10 sequence and R08 passed its fixed gate, so no stop condition applies.

## 2026-09-28T00:13:28+05:30 — Complete alternative-fold sensitivity analysis

- **Decision made:** Accept R09 as `DESCRIPTIVE_COMPLETE` with preregistered stability label `MODERATE`. Preserve R02 selection and continue to R10.
- **Triggering artifact:** `experiments/r09_fold_sensitivity/artifacts/experiment_manifest.json`, SHA256 `fcfa94aefaefab4f1e6abcfa12534da52206baed63e616790d7a2c0e34f13423`; result SHA256 `29311b740b0ca9ae2806c7eb54ac0c86a93255d8706e324fb52f3262d61274bc`; audit report SHA256 `69350abcbc742923eba25e96bf174f7760478631b91b6e5fbd819c079b0a0618`; three distinct five-fold assignments, 680 cases per assignment, 583 isolated groups, 2,000 paired bootstrap draws per comparison, and zero sealed-test images.
- **Rationale:** Alternative AUROCs were `0.9325` (seed 17), `0.9553` (seed 42), and `0.9560` (seed 93), with mean `0.9479` and sample SD `0.0134`. Maximum absolute change from original R02 was `0.0230`, meeting the preregistered `MODERATE` rather than `NARROW` rule. Seed 17's paired change was `-0.0230` with CI `[-0.0604,+0.0025]`; its meningioma subgroup fell to `0.8565` while only seven failures make that estimate unstable. R09 remains descriptive and cannot alter selection.
- **Verification:** All source hashes matched; all three assignments were distinct from each other and original, covered every case once, preserved group isolation, and retained both outcomes in every partition. Original R02 OOF scores and AUROC reproduced with difference `0`. Independent recomputation matched all three AUROCs, their mean/SD, and every manifest hash. 132 tests passed and Ruff passed.
- **Alternatives considered:** Choose seed 93 as the preferred fold split; rejected because R09 measures sensitivity rather than selecting a favorable partition. Call stability narrow because two seeds closely match original; rejected because the registered rule uses all three. Treat seed 17 as proof of failure; rejected because R09 has no gate and its paired CI includes zero.
- **Who approved:** User authorized R09 under the standing sequence; Codex applied the registered descriptive rules without changing folds after results.

## 2026-09-28T06:52:59+05:30 — Register estimator-seed stability analysis

- **Decision made:** Register R10 under the unchanged Stage 4 parent definition with estimator seeds 7, 21, 42, 84, and 168. Do not create a seed-replacement amendment. Proceed under standing approval after committing registration.
- **Triggering artifact:** `experiments/r10_seed_stability/config.json`, SHA256 `23dfa7800e02bac8ddc3ff3b937add4b758186dbb7d4356645d054db43a5723b`; hypothesis SHA256 `2c59fa61075df7b463fb9fb2915e224cfc140c6db4f9394d7b2d224d64c429be`; implementation-plan SHA256 `47853c1863c47efa73f7bd395cd18a25d6054dace434c986c0fce8c45e80d6d6`; Stage 4 parent amendment SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`; R09 result SHA256 `29311b740b0ca9ae2806c7eb54ac0c86a93255d8706e324fb52f3262d61274bc`.
- **Rationale:** R10 varies only the logistic estimator random seed over the five parent-preregistered values while preserving the original folds, features, target, scaling, solver, and all other parameters. It reports per-seed OOF AUROC, ECE, Brier skill, paired bootstrap comparisons, and probability-vector differences. `EXACT/NUMERICALLY_STABLE/SEED_SENSITIVE` is a preregistered descriptive label, not a gate.
- **Alternatives considered:** Replace seeds with 42–46; rejected at the user's correction because it contradicted the existing registration. Amend the parent registration; explicitly rejected by the user. Vary fold seeds simultaneously; rejected because R09 already isolated fold sensitivity and R10 isolates estimator randomness.
- **Who approved:** User explicitly corrected the seed list and authorized R10 with seeds 7, 21, 42, 84, and 168.

## 2026-09-28T06:56:58+05:30 — Complete estimator-seed stability analysis

- **Decision made:** Accept R10 as `DESCRIPTIVE_COMPLETE` with preregistered stability label `EXACT`. Preserve R02 selection and proceed to final Stage 4 model selection.
- **Triggering artifact:** `experiments/r10_seed_stability/artifacts/experiment_manifest.json`, SHA256 `7bf8172e7fe5d75d3543e1ca72b189f3cae40a3d5488bbe753547ea45359ed5d`; result SHA256 `59245caf1f2d8aba0221503a08466b990a6d70318afdaa71c2d87de52e12c888`; audit report SHA256 `960cf06155169c227f2389b53c1ec9d2726c92a66dc7986eb6d5ded365b77182`; five preregistered seeds, 680 OOF cases per seed, 2,000 paired bootstrap draws per seed, and zero sealed-test images.
- **Rationale:** Seeds 7, 21, 42, 84, and 168 each produced AUROC `0.955502`, ECE `0.017133`, and Brier skill `0.503348`. All five OOF vectors were identical to the seed-42 vector, AUROC sample SD was `0`, and every paired bootstrap difference was exactly `0`. The seed-42 reconstruction differed from the serialized R02 probabilities by at most `3.09e-14`, below the fixed `1e-12` tolerance. This exactness is consistent with lbfgs determinism here and does not generalize to other estimator families.
- **Verification:** The parent seed definition was verified directly; every source hash matched; the original folds, feature contract, population, groups, C03 transform counts, and outcome support passed. Independent recomputation verified all five AUROCs, exact cross-seed vector equality, and every manifest hash. 134 tests passed and Ruff passed.
- **Alternatives considered:** Amend to seeds 42–46; explicitly rejected by the user. Treat exactness as proof all logistic training is deterministic; rejected because it is limited to this solver, data, and environment. Tune or change solver to create randomness; rejected as an unregistered variant.
- **Who approved:** User explicitly confirmed the parent-preregistered seeds and authorized R10 execution without amendment.

## 2026-09-28T07:05:21+05:30 — Select operational reliability model and close Stage 4

- **Decision made:** Select R02's ten-feature aggregate logistic reliability model for the C03-calibrated E08 pathway, mark H7 `PASS`, complete the conditional E08 selection record, and declare Stage 4 complete and ready for Stage 5 review.
- **Triggering artifact:** `experiments/e08_final_selection/artifacts/experiment_manifest.json`, SHA256 `cc1be855c11b98378bf708504921e330030fc854d0ff5fc4dbdd772cfaf092b0`; selection result SHA256 `dc2bb7f305d97830b499dc2599050e52c7119c4a3b1ae4997c240cc73edb11b9`; model-selection report SHA256 `a69851f923b5de9b5f06ac03e23b61bd4df22cef4fd35d68f62e241dd10ace61`; Stage 4 completion report SHA256 `7b619032bf9271a6eef01889c96bf7cc04a154f4e92f3bf7f2fd3f488188132f`; source evidence commit `1eb5c4f2ae60f8da45392ffc448a39a4bf7a2265`.
- **Rationale:** Among selectable R02–R06 candidates that passed their own gates, R02 achieved aggregate OOF AUROC `0.955502` versus R04 operational routing at `0.930860`; R02 also remained higher on R04's matched 679-case cohort (`0.955015`). AUROC therefore decided the registered ranking before ECE, Brier, or experiment-ID tie-breakers. R08's unweighted mean fold AUROC was `0.960424`, above the H7 minimum `0.941`. Relative to v1, selected E08 reliability changed AUROC by `+0.014512`, ECE by `-0.004038`, and Brier skill by `+0.027296`. Overlapping AUROC intervals prevent a superiority claim.
- **Verification:** All C03 and R01–R10 result and manifest hashes matched. The selection table independently reproduced R02 as the highest-AUROC eligible model; H7 arithmetic and E08-v1 differences reproduced exactly; every final-selection manifest hash matched. 136 tests passed and Ruff passed. Sealed-test accesses, threshold changes, and classifier/neural-segmenter retraining events were all zero.
- **Alternatives considered:** Select R04 because it passed six category gates; rejected because the registered final rule ranks aggregate OOF AUROC and R04 is lower. Select R05 because its aggregate AUROC nearly matches R02; rejected because R05 failed its predicted-glioma gate. Claim E08 statistically superior to v1; rejected because bootstrap intervals overlap and this is internal development evidence.
- **Who approved:** User authorized final selection and Stage 4 consolidation after R10. Codex applied the existing rule without relaxation and pauses before Stage 5.

## 2026-09-28T07:19:23+05:30 — Fix the Stage 5 reliability-policy cost model

- **Decision made:** Accept P01 as `DESCRIPTIVE_COMPLETE` and fix the preregistered binary review costs at `TP=1`, `FP=1`, `TN=0`, and `FN=10`. Do not select a reliability threshold in P01; pause before Stage 5.2.
- **Triggering artifact:** `experiments/p01_cost_model/artifacts/experiment_manifest.json`, SHA256 `ef52545bc645be0a453978046cdf094546408091d68c6ad8f15b5f99230bab50`; result SHA256 `d66042c2ce9ea848c6e98e8a06a26c9aace02a4f54b253de893b1ccafb531109`; cost-model document SHA256 `06d60c9cfd702df8b510be83aa3c4c0de792a159934c5b0aefa8edcce8b09045`.
- **Rationale:** One unit represents the resource cost of mandatory review for either a good or bad segmentation. Ten units for a silently accepted `Dice <0.50` mask encode the registered preference to spend review effort rather than accept a poor mask. The weights are relative engineering preferences and are not clinical utility, monetary, or patient-harm estimates.
- **Verification:** Eight governing source hashes matched. C03-calibrated E08 and R02 selection matched. The R02 table contained 680 unique OOF validation cases, 598 acceptable cases and 82 failures, five held-out folds, finite bounded reliability scores, and exact outcome agreement with the fixed `0.50` Dice cutoff. No sealed-test path was present. Threshold changes, neural retraining, and reliability-estimator refitting were all zero.
- **Alternatives considered:** Estimate monetary or clinical costs; rejected because no validated clinical workflow or utility elicitation exists. Use tumor-specific costs; deferred because the registered initial policy uses one matrix and subgroup-specific values require stakeholder evidence. Optimize the threshold during P01; rejected because costs must be fixed before Stage 5.2 observes the cost curve.
- **Who approved:** User approved the verified P01 plan and explicitly authorized registration, artifact production, review logging, and commit while requiring a pause before Stage 5.2.

## 2026-09-28T07:36:03+05:30 — Select internal-development reliability-policy thresholds

- **Decision made:** Accept P02 as `PASS`. Select global binary review threshold `0.9490250173235835`. Record the exhaustive three-state optimum as `theta_low=theta_high=0.9490250173235835`, which validly collapses the Warning band to zero cases under the fixed costs. Retain the global threshold as the policy candidate and keep true-category optima descriptive.
- **Triggering artifact:** `experiments/p02_threshold_selection/artifacts/experiment_manifest.json`, SHA256 `80e1b9f3f655d43c451a194b7a9b5ddd6b6cc3de930cbd961855f880fe8196ef`; result SHA256 `ab86f12bb3116429bbfeb633cdef0ea0f636623757202d821e3f5b52a2fac05c`; threshold-selection report SHA256 `49a5d515aff3d32b3ebebc224ce8d30ef168ea0d0bc103e26a11ac720f0d3067`.
- **Rationale:** The selected binary operating point minimizes the registered cost at `205` total or `0.301471` per case, flags `195/680`, catches `81/82` failures, and misses one. The conventional `0.50` reference costs `406`, flag-all costs `680`, and flag-none costs `820`. All 230,860 valid three-state pairs were evaluated with exact quarter-cost arithmetic; the optimum contains 195 Review Required, zero Warning, and 485 Standard cases. A nonempty Warning state would cost more under the preregistered weights and therefore was not forced.
- **Subgroup finding:** Glioma's descriptive optimum `0.950106439123928` changes zero assignments and produces the same cost as the global threshold, so it does not materially differ operationally. Meningioma's `0.5556869041978794` and pituitary's `0.9470618500658499` reduce subgroup cost descriptively, but sparse failures and use of true labels prohibit operational routing.
- **Verification:** Ten upstream hashes matched before result generation. The sweep independently reproduced 679 binary candidates and 230,860 three-state pairs. The population remained 680 unique OOF cases, 598 acceptable and 82 bad, across five folds. Both cost vectors remained fixed; the segmentation threshold stayed `0.50`; sealed-test access and model retraining/refitting were zero. Independent artifact and tie-break recomputation passed. All three figures passed visual inspection. 139 tests passed, Ruff passed, and the experiment registry validated.
- **Alternatives considered:** Force a nonempty Warning interval; rejected because it violates the registered cost minimization. Deploy category-specific thresholds; rejected because this analysis groups by true type, subgroup failure counts are sparse, and the global policy was registered as the operating candidate. Treat `0.50` as an existing default; rejected because the repository had no deployed reliability-policy threshold.
- **Who approved:** User approved the registered P02 plan and authorized the complete threshold sweep, figures, governance artifacts, commit, and push while requiring a pause before Stage 5.3.

## 2026-09-28T09:42:19+05:30 — Formalize the two-state reliability decision rule

- **Decision made:** Accept P03 as `PASS` and adopt the two-state policy contract: reliability below `0.9490250173235835` is `REVIEW_REQUIRED`; reliability at or above it is `STANDARD`. Missing or invalid reliability fails closed to `REVIEW_REQUIRED`. Do not expose a Warning state.
- **Triggering artifact:** `experiments/p03_decision_rule/artifacts/experiment_manifest.json`, SHA256 `2a2bbc1198ae08e4b206ee91c0bdf9c0ac7d45c4d80ffff7eaf982a769038377`; result SHA256 `5b7573987e92b1a82461e92e37efd68ba46bdfc0bc0cc426c1128218e1944011`; decision-rule document SHA256 `43290ba09ec991819dbbc21346b447cd70c0cfcda1e75b0d21e99b502fd744c5`.
- **Rationale:** P02's exhaustive cost minimization selected equal lower and upper thresholds, so the Warning interval is empty. P03 preserves that finding instead of inventing a nonoptimal band. The exact boundary assigns equality to Standard. Missing, nonnumeric, non-finite, Boolean, negative, and greater-than-one inputs fail closed with explicit unavailable or invalid reason codes.
- **Observed behavior:** Across the same 680 OOF cases, the formal rule produced 195 Review Required and 485 Standard, caught 81 of 82 failures, missed one, reviewed 114 acceptable masks, and reproduced total cost 205. Only `STANDARD` and `REVIEW_REQUIRED` were emitted.
- **Glioma finding:** The descriptive glioma optimum `0.950106439123928` differs numerically from the global threshold but changes zero assignments and zero cost, so no separate glioma policy is introduced. Glioma remains the weakest true-category reliability subgroup at AUROC `0.876843` (approximately `0.877`).
- **Verification:** Six governing hashes matched; the threshold and cost vector were unchanged; all 680 unique OOF cases and five folds matched; the segmentation threshold remained `0.50`; test access and model retraining/refitting were zero. Thirteen valid, boundary, unavailable, and invalid input examples passed. The JSON Schema structure and examples were checked through the policy contract tests without adding an unavailable optional dependency. 144 tests passed, Ruff passed, and registry validation passed.
- **Alternatives considered:** Add a Warning state for presentation flexibility; rejected because it contradicts P02's registered optimum. Route glioma separately; rejected because assignments and cost are identical and true type is not a policy feature. Accept missing reliability as Standard; rejected because the registered contract is fail closed.
- **Who approved:** User approved the verified P03 plan and explicitly authorized registration, contract artifacts, tests, review logging, commit, and push while requiring a pause before Stage 5.4.

## 2026-09-28T10:23:55+05:30 — Adopt the fixed user-facing reliability message catalogue

- **Decision made:** Accept P04 as `PASS` and adopt the fixed eight-message catalogue. Cover general and classifier-predicted-glioma variants for `STANDARD / AT_OR_ABOVE_THRESHOLD`, `REVIEW_REQUIRED / BELOW_THRESHOLD`, `REVIEW_REQUIRED / RELIABILITY_UNAVAILABLE`, and `REVIEW_REQUIRED / RELIABILITY_INVALID`. Do not expose a Warning policy state.
- **Triggering artifact:** `experiments/p04_warning_copy/artifacts/experiment_manifest.json`, SHA256 `655e4b2abaa9769c33295980bdcdb8e041ea2abb0fedf01198ba08160ff02d8a`; result SHA256 `a03b36297132aee490a74d03cf5cfa787495243a738ea2cef1f78ed1ee3f3fdb`; warning-copy document SHA256 `ef2744da08948cf682c191b89d6299b2121cf5f599ef654a306f5705eedb1d99`; catalogue SHA256 `8e9a8c185221fac7f1bad5a4dedfdf8661e8d59b889ae83b0dd7b731f1004c2f`.
- **Rationale:** Complete messages avoid unchecked runtime fragment assembly. Every entry states the exact threshold `0.9490250173235835`, includes `Research prototype — not for clinical use`, and avoids clinical, diagnostic, correctness, safety, or accuracy promises. Glioma copy explicitly identifies a model-predicted category and describes weaker internal reliability ranking without changing policy.
- **Fail-closed event:** The first pre-result gate detected that the registered disclaimer did not decode to the exact U+2014 em dash string in the PowerShell/Python path. No result existed. The registration was corrected to use the JSON Unicode escape, committed, and the complete gate was rerun successfully before catalogue generation.
- **Verification:** Three upstream hashes matched; the P03 threshold, two-state contract, and no-Warning finding were unchanged. All eight combinations and unique IDs verified. Every full message contained the exact threshold and disclaimer; all fallback variants required review; glioma language was prediction-qualified; prohibited and percentage claims were absent; plain-language limits passed. The segmentation threshold stayed `0.50`; test access, retraining/refitting, and gate/policy relaxation were zero. 149 tests passed, Ruff passed, and registry validation passed.
- **Alternatives considered:** Assemble base copy and glioma fragments dynamically; rejected because complete catalogue entries are easier to validate and audit. Use reassuring accuracy language for Standard; rejected because threshold passage is not a correctness or clinical-validity claim. Omit the prediction for Review Required; rejected because manual inspection needs the output visible.
- **Who approved:** User approved the verified P04 plan and explicitly authorized the fixed catalogue, documentation, validation tests, review logging, commit, and push while requiring a pause before Stage 5.5.

## 2026-09-28T10:50:22+05:30 — Measure reliability-threshold sensitivity

- **Decision made:** Accept P05 as `PASS` with sensitivity classification `FRAGILE` and locally-flat status `NO`. Keep the selected threshold `0.9490250173235835` unchanged.
- **Triggering artifact:** `experiments/p05_threshold_sensitivity/artifacts/experiment_manifest.json`, SHA256 `9f5573652f2623cb3ae8a9098d3ab7f589f54ea4d4a7bafe0e91c3e7033f1a8c`; result SHA256 `fa99058527d645d2d85d2b2bb02f63be97916a968bf9a777be0b9914c31e3711`; sensitivity curve CSV SHA256 `36b8d0fc2aba5ae4d0b032f84356b1fd28d7efd8707317a0142280cf110c7236`.
- **Rationale:** The selected cost is 205. Both applied ±0.02 endpoints cost 245, a 19.51% increase that exceeds the preregistered 10% locally-flat limit. The worst ±0.05 endpoint costs 628, exceeding the preregistered 25% robustness limit; the worst ±0.10 endpoint costs 680, with the positive endpoint clipped to threshold 1.0. The required conclusion is: Small perturbations produce large cost changes. The threshold requires careful calibration.
- **Historical comparison:** Threshold 0.50 costs 406 (+201), flag nothing at 0.00 costs 820 (+615), and flag everything at 1.00 costs 680 (+475), compared with 205 at the selected threshold.
- **Verification:** Nine governing hashes matched. The 680 unique OOF cases, 598 acceptable cases, 82 failures, five folds, cost vector, and target definition matched. All 551 curve rows were independently recomputed; the selected row reproduced P02 exactly. The segmentation threshold stayed `0.50`; sealed-test access, training, refitting, and gate relaxation were zero. The figure passed visual inspection.
- **Alternatives considered:** Relabel the threshold robust because it remains cheaper than historical baselines; rejected because it fails both preregistered robustness conditions. Change the selected threshold after sensitivity inspection; rejected because P05 is descriptive and cannot alter P02.
- **Who approved:** User approved the registered P05 plan and authorized sensitivity analysis, governance artifacts, commit, push, and the Stage 5 completion report while requiring a pause before Phase 6.

## 2026-09-28T13:01:49+05:30 — Register external cross-dataset transport evaluation

- **Decision made:** Register Amendment 13, hypothesis H8, and X01–X05 before external image access. Preserve execution order Cheng, OpenNeuro pituitary, UPENN-GBM, Meningioma-SEG-CLASS, then synthesis.
- **Triggering artifact:** `audit/preregistration_amendment_13.md`, SHA256 `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5`; registration verification SHA256 `ae3a767d2d3442e60f34ba6966e569adacd89adb7bff39e750f0d60b2cf678ad`.
- **Rationale:** The external cohorts require prospective identity, overlap, patient, label, geometry, access, and metric-availability audits. The model stack, thresholds, bootstrap, domain-gap analysis, failure taxonomy, controlled-access rule, disk lifecycle, and GBM-to-glioma caveat are fixed before image access.
- **Fail-closed finding:** C03 has a complete frozen all-validation mapping, but R02 has no committed deployable estimator containing scaler state and logistic intercept. External R02 scoring is registered as `BLOCKED_ARTIFACT_MISSING`; fitting, reconstructing, or averaging models is prohibited without a separately approved amendment.
- **Disk and access:** Drive E had 350,710,870,016 free bytes (`326.62495 GiB`). Sequential processing becomes mandatory if the projected simultaneous footprint exceeds 250 GB. Meningioma-SEG-CLASS remains `CONTROLLED_ACCESS_PENDING_HUMAN_REQUEST`; no request or grant is claimed because human identity and attestations are required.
- **Verification:** Six frozen artifacts matched their registered SHA256 values; the experiment registry validated with X01–X05; external image access and BRISC sealed-test access were zero; segmentation threshold remained `0.50`; training/refitting was zero.
- **Alternatives considered:** Silently refit R02 on all validation cases; rejected as a violation of the no-refit constraint. Transfer mean fold coefficients without scaler/intercept; rejected because it is not the registered model. Treat Cheng as independent by name alone; rejected pending source-overlap audit.
- **Who approved:** User approved Amendment 13, H8, the fixed dataset order, disk policy, controlled-access handling, mandatory GBM caveat, and a pause after Stage 6.0.

## 2026-09-28T13:34:00+05:30 — Authorize one deterministic R02 operational fit

- **Decision made:** Register Amendment 14 and X00 before fitting. Authorize exactly one deterministic full-population fit of the selected R02 `StandardScaler` plus L2 logistic-regression specification for operational serialization. Preserve R02's OOF scores as the only reported reliability evaluation.
- **Triggering artifact:** `audit/preregistration_amendment_14.md`, SHA256 `8abe7d0fb56c116f2c3f3f667e0263564dcb89b5021b8a3dc61959487849b7d5`; user authorization following Amendment 13's `BLOCKED_ARTIFACT_MISSING` result.
- **Rationale:** External reliability transfer requires a complete estimator with feature order, scaler state, intercept, coefficients, and runtime provenance. R02 selected the specification but did not serialize it. A fixed all-development fit resolves deployment without changing the estimator or evaluating it on its fitting cases.
- **Verification rule:** Reproduce R07's recorded full-population standardized coefficients and independently reproduce scaler state, intercept, and all 680 probabilities within `1e-12`; then reproduce the probabilities after serialization and reload within `1e-12`. R07 did not save a historical full-population probability vector, so no nonexistent comparison is claimed.
- **Alternatives considered:** Average fold estimators or reconstruct a model from mean coefficients; rejected because neither is the registered R02 estimator. Treat OOF predictions as a transferable estimator; rejected because they are case-specific evaluation outputs. Fit during external evaluation; rejected because it would mix fitting and transport evaluation.
- **Who approved:** User explicitly authorized Amendment 14, its exact estimator, verification, serialization, constraints, deliverables, commit, and push.

## 2026-09-28T16:37:15+05:30 — Freeze the selected R02 reliability estimator

- **Decision made:** Accept X00 as `PASS` and freeze the deterministic full-population R02 estimator for operational inference. Preserve R02 OOF results as the reported performance evidence.
- **Triggering artifact:** `models/r02_reliability_manifest.json`, SHA256 `0d56aab5d562a34325003cee116c4f9d5f8fbe3fcc5278717338706d58e5303f`; X00 result SHA256 `e97ec7c46a317b1e2aef5671e29a11d58ec4c2b6f36f1b85ab089fdfe124d655`.
- **Rationale:** R07 coefficients reproduced with maximum absolute error `0`. Independent feature reconstruction and fitting reproduced probabilities with error `0`; serialization reload error was `0`, all within `1e-12`.
- **Verification:** Frozen source hashes, 680 cases, 598 successes, 82 failures, 583 groups, locked ten-feature order, feature contract, 22,891 C03 changed pixels across 679 images, and threshold `0.50` matched. External and sealed-test image access, neural retraining, and threshold changes were zero.
- **Alternatives considered:** Fold-model averaging and coefficient-only reconstruction remained rejected. The exact fixed specification was fitted once for deployment under Amendment 14.
- **Who approved:** User explicitly authorized Amendment 14 and the fit-and-freeze operation. Codex applied the predeclared deterministic checks without relaxation.

## 2026-09-28T14:08:00+05:30 — Freeze the Stage 6 shared adapter contract

- **Decision made:** Accept Stage 6.1 as `COMPLETE_SYNTHETIC_VERIFICATION_ONLY`. Freeze the shared uint8 bridge, segmenter and classifier letterboxing, mask transform, source-coordinate restoration, NIfTI canonicalization, DICOM geometry ordering, C03 operational mapping, R02 feature order, and full-precision policy threshold before external image access.
- **Triggering artifact:** `audit/stage6_shared_adapter_contract.md`, SHA256 `9e22bc6a90a7b2c6f508200e0beba2c69892cfbfbddcc09ee7d94e95ddc156f1`; adapter SHA256 `f510219315c8ea8c31b7da7c4b4def6710a9f6fd17351ac8b602a04212d571ad`; adapter-test SHA256 `0d98c480122b78f5842f0c6343c5fc86c554c9b518619ed2fda2d48cddd73b73`.
- **Rationale:** All external sources must cross one deterministic, outcome-independent boundary. Native uint8 is preserved; non-8-bit MRI receives the preregistered nonzero 0.5/99.5 percentile display bridge before the exact development transforms. Geometry and physical metadata remain attached so masks and measurements cannot silently drift.
- **Verification:** Ten focused synthetic tests and 165 repository tests passed; Ruff passed. Frozen C03 and R02 hashes matched. Segmenter threshold remained `0.50`, reliability-policy threshold remained exactly `0.9490250173235835`, and external/BRISC sealed-test image access and retraining were zero. Drive E retained 350,710,026,240 free bytes (`326.624165 GiB`).
- **Alternatives considered:** Let each dataset invent preprocessing; rejected because outcome-dependent adapters would invalidate comparison. Reuse BRISC path-based E07 routing; rejected because external routing must use the frozen classifier prediction. Infer physical measurements without spatial metadata; rejected fail closed.
- **Who approved:** User approved the Stage 6 plan and authorized Stage 6.1 after Amendment 14 passed.


## 2026-09-28T17:14:07+05:30 — Audit Cheng/Figshare v8 before inference

- **Decision made:** Classify X01 as `ELIGIBLE_OVERLAP_CONTROL` and authorize one frozen-stack evaluation under that label. Do not treat it as independent external validation.
- **Triggering artifact:** `experiments/x01_cheng_overlap_control/artifacts/dataset_audit.json`, SHA256 `376475408891a18e2eb8491eedbe84f673b346ac460f180de24d0c5522eefaff`; official Figshare metadata SHA256 `b02780e0c61b9150c8eac7ee090411f33f1b57ecf3088c0bb5e33cb68752d252`.
- **Rationale:** All 3,064 images and masks, 233 patient identifiers, five patient-separated official folds, release hashes, archive membership, mask encoding, and tumor-border alignment passed. The official BRISC publication documents Cheng as an upstream source of BRISC's Nickparvar aggregate, establishing source overlap regardless of re-encoding differences.
- **Verification:** All source files matched byte, MD5, and SHA256 checks. BRISC overlap comparison used only the existing frozen inventory and opened zero BRISC pixels. Inference, performance evaluation, neural retraining, threshold changes, and BRISC test-pixel access were zero.
- **Alternatives considered:** Declare Cheng independent because filenames and encodings differ; rejected because provenance overlap is documented. Stop X01 as ineligible; rejected because the image/mask contract is technically valid for an explicitly labeled overlap control.
- **Who approved:** User approved Amendment 13 and authorized progression through eligible audits and evaluations unless a registered stop condition occurs.


## 2026-09-28T17:36:04+05:30 — Reject X01 transport gates and retain overlap-control findings

- **Decision made:** Complete X01 as `FAIL_REGISTERED_TRANSPORT_GATES_OVERLAP_CONTROL_ONLY`. Accept the technically valid frozen run and visual review, but reject the registered transport gate because glioma Dice and R02 ECE failed.
- **Triggering artifact:** `experiments/x01_cheng_overlap_control/artifacts/evaluation_result.json`, SHA256 `c5cd7564f09537ff4433d6b606d3de67a5d3928693591c778370f6d26be94b1f`; deterministic visual panel SHA256 `efa0fbc53438e63c2296a82cb72c86b71bc3772fbbaba753715275392f234c56`.
- **Rationale:** Glioma subject-macro Dice was `0.414165` against `0.55`. Reliability AUROC remained `0.938719`, but ECE was `0.061323` against `0.05`. Meningioma and pituitary Dice and all classifier gates passed. Visual inspection found aligned references/predictions and no common transform defect.
- **Verification:** Independent recomputation over 3,064 rows had maximum metric difference `2.220446049250313e-16`. Frozen hashes matched; 2,000 patient-cluster bootstraps used seed 42. No BRISC sealed-test file was opened. Neural retraining, calibrator/reliability refit, segmentation-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Relax the glioma or ECE gates because ranking AUROC was strong; rejected. Present X01 as independent external validation; rejected because Cheng is an upstream BRISC source. Tune the C03 mapping or reliability threshold on Cheng; rejected as forbidden external-data fitting.
- **Who approved:** User approved Amendment 13, the frozen X01 overlap-control evaluation, and unchanged fail-closed gates.


## 2026-09-28T19:08:23+05:30 — Audit OpenNeuro ds006248 v1.0.0 before inference

- **Decision made:** Classify X02 as `ELIGIBLE_INDEPENDENT` and authorize one frozen-stack evaluation.
- **Triggering artifact:** `experiments/x02_openneuro_pituitary/artifacts/dataset_audit.json`, SHA256 `999ff3a09f2918024924c9a16151b9ef5b60a6527752bdec36a0fc0b3b3737d4`; snapshot commit `1b8ebd1f1171049c7f4258ea41933e7b8ee6c06e`.
- **Rationale:** All 50 subjects and 100 required annex payloads passed exact hash, size, modality, finite-value, image/mask geometry, integer-label, tumor-presence, and subject-uniqueness checks. Exact display-pHash candidates against frozen BRISC/X01 pituitary metadata: 0/0.
- **Licence handling:** The frozen release conflicts internally (`CC0` in `dataset_description.json`; `CC BY-NC 4.0` in `README.md`). The project applies the more restrictive `CC BY-NC 4.0` statement unless the publisher clarifies it.
- **Verification:** BRISC comparison opened zero BRISC pixels. Inference, performance evaluation, neural retraining, refitting, threshold changes, and BRISC test-pixel access were zero.
- **Alternatives considered:** Download the entire multimodal dataset; rejected because the registered input contract requires only CE-cor T1 and expert ground truth. Treat the licence conflict as permission to use CC0; rejected as insufficiently conservative.
- **Who approved:** User approved Amendment 13 and continuation through eligible audits/evaluations unless a registered stop condition occurs.


## 2026-09-28T19:30:30+05:30 — Register X02 target-representability handling and one restart

- **Decision made:** Register Amendment 15 before restarting X02. Preserve all 425 source-positive slices, assign Dice/IoU/boundary F1 zero to the five targets erased by frozen 128×128 nearest-neighbor preprocessing, use the strict all-425 subject-macro Dice for the unchanged gate, and report representable-only, source-coordinate, and exploratory whole-volume views separately.
- **Triggering artifact:** `audit/preregistration_amendment_15.md`, SHA256 `4fea798e9a50ef12d5ea2d9cc627cfd8594207860cba10f3bb82e2b7ec782d9e`; X02 audit result SHA256 `999ff3a09f2918024924c9a16151b9ef5b60a6527752bdec36a0fc0b3b3737d4`.
- **Rationale:** Five of 425 native source-positive edge slices contain only 2–10 tumor pixels and become empty after the already frozen nearest-neighbor resize. Excluding them from the primary gate would favor the model after observing a transport limitation; changing interpolation or adding mask dilation would change the frozen adapter. Conservative zero scores preserve membership and make the limitation visible.
- **Aborted attempt:** The first launch stopped at `sub-008:coronal:003`. Five complete batches (80/624 slices) were inferred before the next data-loader batch failed. No X02 performance output was saved or inspected. This is one aborted inference attempt, not a completed evaluation.
- **Verification:** Read-only preprocessing verification identified exactly five cases. BRISC sealed-test access, neural retraining, calibrator/reliability refitting, segmentation-threshold changes, policy-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Drop the five slices; rejected as post-discovery attrition. Preserve small masks by dilation or different interpolation; rejected because the adapter is frozen. Stop X02 permanently; rejected because native source-coordinate and strict sensitivity views permit a transparent evaluation without altering inputs or targets.
- **Who approved:** User explicitly approved Amendment 15, the strict all-425 primary metric, the required five-case artifact, source-coordinate and exploratory whole-volume reporting, and one clean restart.


## 2026-09-28T19:46:18+05:30 — Reject X02 transport gates and retain independent external findings

- **Decision made:** Complete X02 as `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`. Reject H8 for this cohort without tuning on X02.
- **Triggering artifact:** `experiments/x02_openneuro_pituitary/artifacts/evaluation_result.json`, SHA256 `a19dc76f447d7151a85df805d9ec9deafdaf251bf8e235ec7c18630ec0375a59`; deterministic prediction panel SHA256 `cdd17034ea4daa58ecea45056844ffbf01f3df50325c91934805fcd974a16b84`.
- **Rationale:** Pituitary recall passed at `0.940000`, but classifier ECE was `0.076187`. Strict all-425 subject-macro Dice was `0.127242` against `0.75`. Reliability AUROC passed at `0.871890`, while ECE was `0.255096` and top-20% captured `85/357` failures. Complete misses accounted for `309/357` failures.
- **Target representability:** The five Amendment-15 cases remain in the strict gate with zero scores. Representable-only Dice `0.129501` and native source-coordinate Dice `0.127327` show that these five cases do not explain the broad failure.
- **Verification:** Independent recomputation covered `45` decision-driving values with maximum absolute difference `0.0`. Visual review found no systematic flip, transpose, offset, or source-mask alignment defect. No BRISC sealed-test image was opened. Neural retraining, calibration/reliability refit, segmentation-threshold changes, policy-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Drop the five nonrepresentable targets or report only the 420-slice sensitivity; rejected by Amendment 15. Tune preprocessing/calibration/reliability on X02; rejected because external data are evaluation-only. Relax the top-20% gate because failure prevalence makes it infeasible; rejected because gates remain fixed after results.
- **Who approved:** User approved Amendment 15 and one restart with fail-closed handling.

## 2026-09-28 — Register Amendment 16 for the UPENN-GBM curated derivative

- **Decision made:** Authorize the complete 147-scan expert-labelled subset from GitHub `data-nih/tcia` tag `upenn-gbm` at commit `a8140bcfa512241fde2475cc3c4640b5cb97dc2b` as the frozen X03 derivative source, with whole tumor defined as derivative labels `{1,2,3}`.
- **Triggering artifact:** Local source-file manifest SHA256 `fe488d87a48bfcce1c4c735793ccccb28a9ba68a7d7400241692d7e45a68dfae`; 294 release assets matched published byte counts and SHA256 digests. The first fail-closed audit stopped because the derivative uses enhancing-tumor value 3 instead of native BraTS value 4.
- **Rationale:** Release metadata attributes the derivative to TCIA UPENN-GBM v2. A source-equivalent subject-002 mask comparison proved exact spatial equality after only `4 -> 3`, establishing label 3 as enhancing tumor while preserving labels 1 and 2.
- **Alternatives considered:** Silently accept label 3; rejected. Treat the derivative as byte-identical to the original Aspera package; rejected. Download the full 69 GB source archive; unnecessary for the fixed 147 expert CE-T1/mask pairs after prospective derivative registration and exact per-file verification.
- **Constraints:** No X03 inference or performance result existed before registration. Frozen models, adapter, thresholds, gates, and the coarse `glioblastoma -> glioma` caveat remain unchanged.
- **Who approved:** User explicitly approved Amendment 16 and sequential Stage 6 completion.


## 2026-09-28T20:36:39+05:30 — Audit UPENN-GBM v2 before inference

- **Decision made:** Classify X03 as `ELIGIBLE_INDEPENDENT` and authorize one frozen-stack evaluation.
- **Triggering artifact:** `experiments/x03_upenn_gbm/artifacts/dataset_audit.json`, SHA256 `44d4f030ff6830ab9e58d0ccb856a3fbb03df774352d054251352a87a2a3190a`.
- **Rationale:** All 147 expert masks and paired CE-Gd T1 volumes passed published SHA256, size, modality, geometry, label, tumor-presence, duplicate, and overlap checks. Exact cross-dataset pHash candidates: 0.
- **Mapping caveat:** Glioblastoma is a glioma subtype, but this GBM-only cohort and whole-tumor target are not directly comparable to heterogeneous BRISC glioma.
- **Verification:** No inference or metric computation occurred. BRISC sealed-test pixels, retraining, refitting, and threshold changes were zero.
- **Who approved:** User approved Amendment 13 and continuation through eligible audits/evaluations.

## 2026-09-29 — Register X03 target-representability handling and one restart

- **Decision made:** Register Amendment 17 before restarting X03. Preserve all 9,464 source-positive slices, assign zero overlap metrics to the 82 targets erased by frozen 128×128 nearest-neighbor preprocessing, evaluate the unchanged gate on this strict cohort, and report representable-only and native-coordinate views separately.
- **Triggering artifact:** `audit/preregistration_amendment_17.md`; hash registered in `experiments/registry.json`. X03 audit result SHA256 `44d4f030ff6830ab9e58d0ccb856a3fbb03df774352d054251352a87a2a3190a`.
- **Rationale:** The 82 masks contain only 1–15 native pixels. Dropping them would create post-discovery attrition, while changing resize or dilating masks would alter the frozen adapter. Conservative zero scores preserve membership and expose the limitation.
- **Aborted attempt:** The pre-inference representability check stopped before model loading. No probability, metric, panel, decision, or performance result was produced. The complete affected set contains 82 slices from 54 patients.
- **Verification:** BRISC sealed-test access, neural retraining, calibration/reliability refitting, segmentation-threshold changes, policy-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Exclude the 82 slices; rejected. Change interpolation or dilate targets; rejected. End X03 permanently; rejected because strict, representable-only, and native-coordinate views permit transparent evaluation without modifying the frozen stack.
- **Who approved:** User explicitly approved Amendment 17, a renewed audit, and exactly one frozen evaluation restart.

## 2026-09-29 — Register native-datatype loader correction and X03 attempt three

- **Decision made:** Register Amendment 18 before correcting the shared NIfTI loader. Preserve canonical native array datatypes so UPENN `uint8` volumes use the already registered `native_uint8` adapter path, then renew the X03 audit and authorize exactly one third evaluation attempt.
- **Triggering artifact:** `audit/preregistration_amendment_18.md`; hash registered in `experiments/registry.json`.
- **Rationale:** The loader unconditionally converted native `uint8` data to `float32`. This contradicted the adapter contract and caused 18 constant or effectively constant reference-empty slices from 14 patients to enter percentile windowing and be rejected.
- **Aborted attempts:** Attempt 1 stopped before inference on 82 target-representability losses. Attempt 2 stopped after nine completed batches and 144 inferred slices on the datatype defect. Neither attempt saved probabilities, performance metrics, panels, or a result.
- **Verification:** The 18 slices include 14 all-zero and four one-nonzero-pixel slices; none is reference-positive. BRISC sealed-test access, retraining, refitting, threshold changes, resize changes, mask dilation, and gate relaxation were zero.
- **Alternatives considered:** Treat black slices as a data failure; rejected because native `uint8` is explicitly supported. Add arbitrary intensity variation; rejected. Exclude the 18 slices; rejected because the registered whole-volume view includes all 22,785 slices.
- **Who approved:** User explicitly approved Amendment 18, the regression-tested loader correction, a renewed audit, and exactly one third X03 attempt.

## 2026-09-29 — Register X03 domain-bootstrap membership correction and attempt four

- **Decision made:** Register Amendment 19 before correcting X03 domain-gap bootstrap membership. Bind mask-derived features to 9,382 `representable_rows` and input, encoder, reliability, and policy analyses to 9,464 `primary_rows`, with runtime length assertions.
- **Triggering artifact:** `audit/preregistration_amendment_19.md`; hash registered in `experiments/registry.json`.
- **Rationale:** Attempt 3 completed all frozen inference, then a 9,382-element representable-only feature array was indexed with the 9,464-row membership. The error is downstream bookkeeping and occurred before any result was written or inspected.
- **Attempt history:** Attempt 1 stopped on 82 target losses; Attempt 2 stopped on the NIfTI datatype defect after 144 inferred slices; Attempt 3 inferred all 22,785 slices and stopped on bootstrap membership. All stops were fail-closed and none produced an X03 performance result.
- **Verification:** BRISC sealed-test access, neural retraining, calibration/reliability refitting, segmentation-threshold changes, policy-threshold changes, resize changes, mask dilation, and gate relaxation were zero.
- **Alternatives considered:** Disable patient bootstrap; rejected. Bootstrap mask features using full rows with placeholder values; rejected. Exclude target losses from every analysis; rejected because Amendment 17 requires strict reliability evaluation on all 9,464 cases.
- **Who approved:** User explicitly approved Amendment 19, the regression-tested membership correction, a renewed audit, and exactly one fourth X03 attempt.


## 2026-09-29T08:31:25+05:30 — Audit UPENN-GBM v2 before inference

- **Decision made:** Classify X03 as `ELIGIBLE_INDEPENDENT` and authorize one frozen-stack evaluation.
- **Triggering artifact:** `experiments/x03_upenn_gbm/artifacts/dataset_audit.json`, SHA256 `bb72fe15bd9ce4cc19ce92d5a68f595207e29c94b2dbc65e6071c4849cbeab56`.
- **Rationale:** All 147 expert masks and paired CE-Gd T1 volumes passed published SHA256, size, modality, geometry, label, tumor-presence, duplicate, and overlap checks. Exact cross-dataset pHash candidates: 0.
- **Mapping caveat:** Glioblastoma is a glioma subtype, but this GBM-only cohort and whole-tumor target are not directly comparable to heterogeneous BRISC glioma.
- **Verification:** No inference or metric computation occurred. BRISC sealed-test pixels, retraining, refitting, and threshold changes were zero.
- **Who approved:** User approved Amendment 13 and continuation through eligible audits/evaluations.


## 2026-09-29T08:52:59+05:30 — Audit UPENN-GBM v2 before inference

- **Decision made:** Classify X03 as `ELIGIBLE_INDEPENDENT` and authorize one frozen-stack evaluation.
- **Triggering artifact:** `experiments/x03_upenn_gbm/artifacts/dataset_audit.json`, SHA256 `463e03fb07733bbbbe6fc2f50d197bdbfaa6e4a1becb446f0662bbdd88cdd8aa`.
- **Rationale:** All 147 expert masks and paired CE-Gd T1 volumes passed published SHA256, size, modality, geometry, label, tumor-presence, duplicate, and overlap checks. Exact cross-dataset pHash candidates: 0.
- **Mapping caveat:** Glioblastoma is a glioma subtype, but this GBM-only cohort and whole-tumor target are not directly comparable to heterogeneous BRISC glioma.
- **Verification:** No inference or metric computation occurred. BRISC sealed-test pixels, retraining, refitting, and threshold changes were zero.
- **Who approved:** User approved Amendment 13 and continuation through eligible audits/evaluations.


## 2026-09-29T09:22:40+05:30 — Audit UPENN-GBM v2 before inference

- **Decision made:** Classify X03 as `ELIGIBLE_INDEPENDENT` and authorize one frozen-stack evaluation.
- **Triggering artifact:** `experiments/x03_upenn_gbm/artifacts/dataset_audit.json`, SHA256 `f870a89cd5165e41187220c5c2326b9597c181427ed16dc643678da9fb9bc30a`.
- **Rationale:** All 147 expert masks and paired CE-Gd T1 volumes passed published SHA256, size, modality, geometry, label, tumor-presence, duplicate, and overlap checks. Exact cross-dataset pHash candidates: 0.
- **Mapping caveat:** Glioblastoma is a glioma subtype, but this GBM-only cohort and whole-tumor target are not directly comparable to heterogeneous BRISC glioma.
- **Verification:** No inference or metric computation occurred. BRISC sealed-test pixels, retraining, refitting, and threshold changes were zero.
- **Who approved:** User approved Amendment 13 and continuation through eligible audits/evaluations.


## 2026-09-29T09:45:31+05:30 — Reject X03 transport gates and retain independent external findings

- **Decision made:** Complete X03 as `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`. Reject H8 for this cohort without tuning on X03.
- **Triggering artifact:** `experiments/x03_upenn_gbm/artifacts/evaluation_result.json`, SHA256 `7b6f7b674688b3310c585b9a75a3a97b1a316c241ab8145e2359d1fd484019a6`; deterministic prediction panel SHA256 `1b487ef761376a9df33eb511f57b7433deb1247db6245b80773baa8c98f36dfb`.
- **Rationale:** Glioma recall failed at `0.421769`, and classifier ECE was `0.245390`. Strict all-9,464 subject-macro Dice was `0.220774` against `0.55`. Reliability AUROC passed at `0.831586`, while ECE was `0.339100` and top-20% captured `1886/7600` failures. Complete misses accounted for `4263/7600` failures.
- **Target representability:** The 82 Amendment-17 cases remain in the strict gate with zero scores. Representable-only Dice `0.222563` and native source-coordinate Dice `0.220473` show that the 82 cases do not explain the broad failure.
- **Verification:** Independent recomputation covered `45` decision-driving values with maximum absolute difference `0.0`. Visual review found no systematic flip, transpose, offset, or source-mask alignment defect. No BRISC sealed-test image was opened. Neural retraining, calibration/reliability refit, segmentation-threshold changes, policy-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Drop the 82 nonrepresentable targets or report only the 9,382-slice sensitivity; rejected by Amendment 17. Modify native dtype handling instead of honoring the registered adapter contract; rejected by Amendment 18. Reuse primary membership for mask-domain bootstraps; rejected by Amendment 19. Tune on X03 or relax the top-20% gate; rejected because external data are evaluation-only and gates remain fixed.
- **Who approved:** User approved Amendments 17–19 and exactly one fourth attempt after the Amendment-19 correction.


## 2026-09-29T14:57:12+05:30 — Close X04 as controlled-access blocked

- **Decision made:** Record X04 as `ACCESS_BLOCKED` after X01-X03 completion and omit performance metrics from X05.
- **Triggering artifact:** `experiments/x04_meningioma_seg_class/artifacts/access_audit.json`, SHA256 `77350372c96977093a53e022878112c4efe7186f7a0c42b4e3a59e8a48f74722`.
- **Rationale:** No verified human-account access grant and no local Meningioma-SEG-CLASS payload existed. Amendment 13 explicitly predeclares this outcome and permits synthesis to continue.
- **Verification:** Frozen C03/R02 hashes passed. Dataset files and images opened, BRISC sealed-test access, retraining/refitting, threshold changes, and gate relaxations were all zero.
- **Alternatives considered:** Claim or simulate access; rejected. Impute meningioma metrics; rejected. Delay synthesis indefinitely; rejected by the registered access-blocked rule.
- **Who approved:** User approved Amendment 13 and instructed X04 to be marked `ACCESS_BLOCKED` if access remained unavailable after X01-X03.


## 2026-09-29T15:03:37+05:30 — Reject H8 and complete Stage 6 synthesis

- **Decision made:** Reject H8 for the evaluated independent cohorts and complete Stage 6. Preserve X01 as overlap-control evidence and X04 as access-blocked.
- **Triggering artifacts:** X01 result `c5cd7564f09537ff4433d6b606d3de67a5d3928693591c778370f6d26be94b1f`; X02 result `a19dc76f447d7151a85df805d9ec9deafdaf251bf8e235ec7c18630ec0375a59`; X03 result `7b6f7b674688b3310c585b9a75a3a97b1a316c241ab8145e2359d1fd484019a6`; X04 access result `77350372c96977093a53e022878112c4efe7186f7a0c42b4e3a59e8a48f74722`.
- **Rationale:** X02 and X03 both failed registered independent transport gates. Reliability ranking remained useful, but calibration and top-risk capture did not transfer. X01 is not independent and X04 has no performance data.
- **Verification:** Source order and terminal statuses matched Amendment 13. Frozen artifact hashes passed. BRISC sealed-test access, retraining/refitting, threshold changes, gate relaxations, and pooled cross-dataset metrics were zero.
- **Alternatives considered:** Pool heterogeneous cohorts, treat X01 as independent, impute X04, or tune from external results; all rejected by the registered protocol.
- **Who approved:** User approved Amendment 13 and the sequential Stage 6 completion protocol.
