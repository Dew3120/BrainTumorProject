# Pre-registration amendment 13 — external cross-dataset transport

**Status:** REGISTERED BEFORE EXTERNAL IMAGE ACCESS

**Registered:** 2026-09-28

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Selected stack:** `experiments/e08_final_selection/result.json`, SHA256 `dc2bb7f305d97830b499dc2599050e52c7119c4a3b1ae4997c240cc73edb11b9`

This amendment prospectively registers external transport evaluation. It does not modify H2–H4, open the sealed BRISC test partition, relax any earlier gate, change the segmentation threshold, fit a model, or authorize clinical claims.

## H8 — external transport

The frozen classifier, E08 segmentation ensemble, C03 calibration mapping, and R02 reliability policy retain useful category discrimination, mask overlap, probability calibration, and segmentation-failure ranking on eligible external MRI cohorts without retraining, refitting, threshold adjustment, or external-data-driven model selection.

H8 is evaluated separately for every eligible cohort and then synthesized. A cohort cannot pass a metric that is mathematically unidentifiable. Missing metrics are `NOT_ESTIMABLE`, never silently pooled across datasets or imputed. Passing H8 is a research transport result only; it does not establish prospective performance, clinical validity, diagnostic safety, or suitability for patient care.

## Registered cohorts and fixed order

| ID | Frozen source identity | Intended role | Initial access state |
|---|---|---|---|
| X01 | Cheng/Figshare `brain_tumor_dataset`, article `1512427`, audit candidate version 8 | Provenance-overlap control; independent only if the audit proves no BRISC source overlap | Public; not accessed before this amendment |
| X02 | OpenNeuro `ds006248`, version `1.0.0`, DOI `10.18112/openneuro.ds006248.v1.0.0` | Independent pituitary evaluation | Public; not accessed before this amendment |
| X03 | TCIA `UPENN-GBM`, version 2, DOI `10.7937/TCIA.709X-DN49` | Independent glioblastoma evaluation using aligned NIfTI | Public; not accessed before this amendment |
| X04 | TCIA `Meningioma-SEG-CLASS`, version 1, DOI `10.7937/0TKV-1A36` | Independent meningioma evaluation | Controlled access pending; no image access |
| X05 | The eligible completed cohorts above | Cross-dataset synthesis | Not started |

The fixed completion order is X01 Cheng, X02 OpenNeuro, X03 UPENN-GBM, X04 Meningioma-SEG-CLASS, then X05 synthesis. Registration and audit may establish that a source is `ELIGIBLE_INDEPENDENT`, `ELIGIBLE_OVERLAP_CONTROL`, `PARTIALLY_EVALUABLE`, `INELIGIBLE_STOP`, or `ACCESS_BLOCKED`.

If X04 access is not granted by the time X01–X03 are complete, X04 is recorded as `ACCESS_BLOCKED` and the synthesis proceeds without its performance metrics. The access request requires the responsible human account holder's controlled-access credentials and attestations; repository registration does not claim that such a request has been submitted or granted.

## Frozen inference stack

- Classifier checkpoint SHA256: `3bc044027b83b2de225ebde3fb817be8827d2b427c599824697b13ff7a727535`.
- E08 v1 fallback checkpoint SHA256: `16dc8745993eef959d0f2924c9147a45320673f6fc0b465ab08f8ad7b2f723de`.
- E08 E05 component checkpoint SHA256: `e17e7f5afec1c56e9faf12e12db263eff4a3ebc184cfcd7bdfcfff0c1a977c6a`.
- E08 E06 component checkpoint SHA256: `b98afabc8118471e01f560ff6a727ce25c08001574bf12005d77d1be773f43ab`.
- E08 E07 specialist checkpoint SHA256: `bb68ce4f0a9203d74a5ec9dea2e83468d3d833ac285a13ebbdb1d793a0d3bbe0`.
- E08 result SHA256: `580846aa620234749ab15ac21a3f12df7398a96edd393564c3abfa1b9f1bf654`.
- E08 final-selection manifest SHA256: `cc1be855c11b98378bf708504921e330030fc854d0ff5fc4dbdd772cfaf092b0`.
- C03 result SHA256: `3af002eacbc05e14134abba67135515a58e060b0c4f255bbccd8f10b79c49cb9`.
- C03 frozen all-validation mapping SHA256: `fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa`.
- R02 result SHA256: `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`.
- R02 manifest SHA256: `1dc386a3cd342f14fcb827960c1ec89f749b2413962c768aec1c22bdf74cb819`.
- Segmentation hard-mask threshold: exactly `0.50`.
- Reliability-policy threshold: exactly `0.9490250173235835`; storage, comparison, schema validation, reports, and UI payloads must preserve this full decimal representation.

### R02 deployable-artifact gate

At registration, R02 has committed OOF scores, five fold coefficient vectors, and R07 has a descriptive full-population standardized coefficient vector. No committed artifact contains the complete deployable R02 transform: feature order, fitted scaler means/scales, logistic intercept, coefficients, solver/version metadata, and a reference prediction checksum together.

Therefore external R02 reliability inference is `BLOCKED_ARTIFACT_MISSING` at registration. Re-running `.fit()`, reconstructing fold models, averaging fold coefficients, inventing scaler values, or treating OOF scores as a transferable model would violate the frozen-stack and no-refit rules. Before any external reliability score is produced, a separately approved prospective amendment must either identify and hash an already-existing complete estimator artifact or authorize one explicit fit-and-freeze operation. Until then, segmentation and classification adapters may be built and tested without external images, but reliability transfer metrics remain blocked.

## Disk and data-lifecycle policy

Registration-time drive E state is 8,578,424,832 bytes used and 350,710,894,592 bytes free (`326.624973 GiB` free). Each dataset audit records free bytes before download, downloaded bytes, extracted bytes, derived-artifact bytes, and free bytes after completion.

Known source sizes at registration are approximately 0.84 GB for Cheng, 69 GB for the aligned UPENN-GBM NIfTI release, and 9.39 GB for Meningioma-SEG-CLASS; the exact OpenNeuro snapshot size is obtained from its versioned manifest before download. If the projected simultaneous raw-plus-extracted-plus-working footprint exceeds 250 GB, datasets are processed sequentially. Intermediate archives or reproducible extracted copies may be deleted only after the dataset evaluation, content manifest, result, and commit are complete and pushed. Original checksums and retrieval instructions remain tracked.

## Dataset audit gate

Every cohort is audited and committed before inference. The audit verifies exact accession/version/licence, archive and file hashes, subject identifiers, subject uniqueness, image/mask completeness, modalities, orientation, geometry, spacing, label semantics, corrupt or blank files, mask alignment, within-dataset duplicates, cross-external duplicates, and overlap with BRISC's already frozen hashes. BRISC test pixels are never opened; only existing inventory hashes may be compared.

The audit outcomes are:

- `ELIGIBLE_INDEPENDENT`: compatible target and no detected source overlap.
- `ELIGIBLE_OVERLAP_CONTROL`: technically evaluable but not independent.
- `PARTIALLY_EVALUABLE`: only named metrics are valid.
- `INELIGIBLE_STOP`: target, identity, integrity, or alignment prevents defensible evaluation.
- `ACCESS_BLOCKED`: required access is unavailable.

An `INELIGIBLE_STOP`, unexpected hash, changed version, mask mismatch, non-finite value, or prior-definition conflict stops the active dataset before performance results. X04 `ACCESS_BLOCKED` is recorded and omitted from synthesis metrics under the explicit exception above.

## Patient independence and bootstrap

External data are never used to fit or select the model. Every subject belongs to one frozen evaluation cohort; all slices from a subject remain together. Official partitions are preserved when present. Primary aggregation gives each subject equal weight. Confidence intervals use 2,000 subject-cluster bootstrap resamples with NumPy generator seed 42 and percentile endpoints 2.5 and 97.5. Single-class draws for AUROC are discarded and deterministically redrawn, with at most 10,000 attempts for 2,000 valid draws.

## Input and target contract

The model receives the same grayscale, aspect-preserving bilinear resize, centered black 128×128 canvas, and `[0,1]` normalization used in development. Masks use nearest-neighbor resize. No external outcome statistic controls preprocessing. Predictions are restored to source coordinates for valid volume-level evaluation.

- Cheng uses its supplied contrast-enhanced T1 slices and masks.
- OpenNeuro uses coronal CE-T1w and tumor mask label 1.
- UPENN-GBM uses registered T1-Gd/T1-CE NIfTI and the documented union of tumor subregion labels as whole tumor.
- Meningioma-SEG-CLASS uses T1-CE with its associated RTSTRUCT tumor contour.

For UPENN-GBM, glioblastoma is a subtype of glioma. Mapping `glioblastoma -> glioma` is coarse. Every UPENN result and synthesis comparison must state that its GBM-only population and target are not directly comparable to heterogeneous BRISC glioma without this caveat.

Primary segmentation is the BRISC-comparable tumor-bearing 2D slice view. Whole-volume analysis including reference-empty slices is secondary. Metrics must never switch between these views without naming the view.

## Registered metrics and reference gates

Report classification category recall, confusion counts, top-label ECE, Brier and NLL, plus one-versus-rest AUROC only when both classes exist. Tumor-only datasets cannot estimate no-tumor specificity or binary tumor/no-tumor AUROC.

Report subject-macro and slice-macro Dice/IoU, zero-Dice and empty-prediction frequency, valid whole-volume Dice, boundary F1, spacing-qualified Hausdorff distance, and raw-versus-C03 NLL/Brier/ECE/MCE for lesion-centred, background, and whole-image regions.

Where the R02 artifact gate is resolved prospectively, report reliability failure AUROC for `Dice <0.50`, ECE/MCE, Brier and Brier skill, top-20% failure and zero-Dice capture, full-precision policy assignments, review rate, failures caught/missed, acceptable cases flagged, and P01 cost. Otherwise report these as `BLOCKED_ARTIFACT_MISSING`.

Applicable existing reference gates are not relaxed: class recall `>=0.85`, classifier top-label ECE `<=0.05`, glioma Dice `>=0.55`, meningioma and pituitary Dice `>=0.75`, reliability AUROC `>=0.75` with lower CI `>=0.70`, reliability ECE `<=0.05`, Brier skill and lower CI strictly above zero, and top-20% failure capture `>=0.50`. A cohort with an unidentifiable required metric is `PARTIALLY_EVALUABLE`, not passed.

## Domain gap and failure taxonomy

Domain-gap analysis compares external cohorts only with BRISC development validation, never sealed BRISC test pixels. Report modality/contrast, orientation, thickness, spacing, dimensions, intensity distance, foreground area, centroid/location, component count, fragmentation, boundary gradient, frozen encoder representation shift, reliability-score shift when available, and policy-review-rate shift when available. Continuous distances use Wasserstein distance and fixed histograms use Jensen-Shannon divergence.

Reuse the Phase 4 failure definitions without alteration: empty prediction, nonempty zero overlap, under-prediction, severe over-prediction, fragmented prediction, disconnected false positive, and edge-touching reference/prediction. Report by dataset, category, orientation, and tumor-size group.

## Completion and stopping rules

1. Execute X01, X02, X03, X04, X05 in that order.
2. Register and commit before data access; audit and commit before inference; evaluate once with the frozen adapter; independently recompute metrics; visually inspect deterministic panels; log the decision; test; commit; and push.
3. No BRISC sealed-test pixel access, neural retraining, calibrator refit, reliability refit, threshold adjustment, external-data-driven model selection, or gate relaxation is permitted.
4. The synthesis distinguishes independent, overlap-control, partial, blocked, and ineligible cohorts.
5. `audit/stage6_cross_dataset.md` must state what the evaluation does and does not establish.
