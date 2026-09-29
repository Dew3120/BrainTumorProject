# Pre-registration amendment 17 — X03 target representability and one restart

**Status:** REGISTERED AFTER A PRE-PERFORMANCE VERIFICATION STOP AND BEFORE ANY X03 PERFORMANCE RESULT

**Registered:** 2026-09-29

**Parent registration:** `audit/preregistration_amendment_13.md`, SHA256 `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5`

**Derivative registration:** `audit/preregistration_amendment_16.md`, SHA256 `802ade9daecfdbc11fcd9344c7aee28e9199562f46fc1990624c940ad8b6bf34`

**X03 audit:** `experiments/x03_upenn_gbm/artifacts/dataset_audit.json`, SHA256 `44d4f030ff6830ab9e58d0ccb856a3fbb03df774352d054251352a87a2a3190a`

This amendment resolves a target-representability condition discovered by a fail-closed assertion before X03 model inference. It does not change a model, image transform, mask interpolation method, calibration mapping, reliability estimator, segmentation threshold, policy threshold, dataset membership, gate, or hypothesis.

## Triggering verification stop

The audited cohort contains 9,464 native whole-tumor-positive axial slices from 147 patients. Frozen 240×240 to 128×128 nearest-neighbor mask preprocessing removes every target pixel from 82 slices belonging to 54 patients. Their native whole-tumor masks contain 1–15 pixels; every resulting model-space mask contains zero pixels.

The first evaluation attempt stopped before model loading or inference and before any probability, metric, panel, decision, or performance result was created. Read-only preprocessing enumeration identified the complete affected membership. This is one aborted pre-inference attempt, not a completed evaluation.

## Frozen metric structure

The 9,464-slice native-positive membership remains fixed. The 82 cases are never relabeled, silently removed, or credited as correct empty predictions.

1. **Primary strict model-space view.** All 9,464 source-positive slices remain in the primary cohort. Each target erased at 128×128 receives Dice `0`, IoU `0`, boundary F1 `0`, and Hausdorff `NOT_ESTIMABLE`. The unchanged X03 subject-macro glioma Dice gate `>=0.55` is evaluated on this strict view.
2. **Representable-only model-space view.** The 9,382 slices retaining target foreground at 128×128 are reported separately as a secondary sensitivity analysis. They do not replace the strict primary view.
3. **Native source-coordinate view.** Frozen hard predictions are restored to 240×240 and compared against native whole-tumor masks for all 9,464 source-positive slices. Subject-macro and slice-macro Dice and IoU are descriptive.
4. **Whole-volume view.** Frozen predictions are restored for all 22,785 axial slices, including reference-empty slices. Whole-volume subject-macro and cohort-global Dice and IoU are exploratory.
5. **Pixel calibration.** Lesion-centred, background, and whole-image summaries use the 9,382 representable cases because a lesion-centred model-space region is undefined for an empty reference. Exact exclusions and denominators are mandatory.
6. **Reliability.** R02 inputs remain prediction-derived. Reliability evaluation uses the strict all-9,464 Dice-success target; all 82 erased targets are failures and are never excluded or used as model inputs.
7. **Failure taxonomy.** Representable cases use the frozen Phase 4 definitions. Each erased target receives `target_not_representable_at_128` as its primary failure mode.
8. **Domain-gap mask features.** Reference-mask model-space comparisons use the 9,382 representable cases and disclose the denominator. Input, encoder, reliability-score, and policy-rate comparisons use all 9,464 primary cases.

The required artifact `experiments/x03_upenn_gbm/artifacts/target_not_representable.csv` lists all 82 affected cases with patient identifier, axial slice index, native target-pixel count, zero post-resize target-pixel count, volume-edge status, and axial orientation.

## Audit restart and evaluation authorization

After this amendment is committed, X03 must restart its derivative-release audit from the beginning and commit the renewed audit decision. Exactly one frozen evaluation restart is authorized after the renewed audit. The result records one aborted pre-inference attempt and one completed attempt. Any further fail-closed condition stops X03 again.

## Preserved constraints

- BRISC sealed-test pixels remain closed.
- Frozen 240×240 to 128×128 preprocessing remains unchanged.
- Mask interpolation remains nearest-neighbor. Dilation, closing, component preservation, or another rescue transform is prohibited.
- Neural retraining, calibrator refitting, and reliability refitting remain prohibited.
- Segmentation threshold remains exactly `0.50`.
- Reliability-policy threshold remains exactly `0.9490250173235835`.
- The X03 glioma subject-macro Dice gate remains `>=0.55`.
- Glioblastoma is a glioma subtype, but this GBM-only cohort and whole-tumor target are not directly comparable to heterogeneous BRISC glioma.
- X03 remains retrospective research transport evidence and cannot establish clinical validity.

## Decision authority

The user explicitly approved Amendment 17, the strict all-9,464 primary metric, the 9,382-slice sensitivity, the all-9,464 source-coordinate view, the required 82-case artifact, a complete audit restart, and exactly one evaluation restart.
