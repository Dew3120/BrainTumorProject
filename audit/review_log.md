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
