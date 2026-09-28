# Pre-registration amendment 15 — X02 target representability and restart

**Status:** REGISTERED AFTER A PRE-PERFORMANCE VERIFICATION STOP AND BEFORE ANY X02 PERFORMANCE RESULT

**Registered:** 2026-09-28

**Parent registration:** `audit/preregistration_amendment_13.md`, SHA256 `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5`

**X02 audit:** `experiments/x02_openneuro_pituitary/artifacts/dataset_audit.json`, SHA256 `999ff3a09f2918024924c9a16151b9ef5b60a6527752bdec36a0fc0b3b3737d4`

This amendment resolves a target-representability condition discovered by a fail-closed assertion during the first X02 launch. It does not change a model, image transform, mask interpolation method, calibration mapping, reliability estimator, segmentation threshold, policy threshold, dataset membership, gate value, or hypothesis.

## Triggering verification stop

The audited cohort contains 425 source-positive coronal slices. The registered nearest-neighbor resize to the frozen 128×128 model space removes all label-1 pixels from five source-positive edge slices:

| Case | Native tumor pixels | Native dimensions | Model-space tumor pixels |
|---|---:|---:|---:|
| `sub-008:coronal:003` | 10 | 512×512 | 0 |
| `sub-015:coronal:013` | 2 | 512×512 | 0 |
| `sub-017:coronal:000` | 9 | 512×512 | 0 |
| `sub-025:coronal:008` | 9 | 512×512 | 0 |
| `sub-028:coronal:013` | 7 | 1024×1024 | 0 |

The condition affects 5/425 source-positive slices (1.1765%). The first launch stopped at the registered assertion before result generation. Five complete batches, containing 80/624 external slices, had passed through the frozen models. No probabilities, metrics, panels, decisions, or performance result artifacts were saved or inspected. This is recorded as one aborted inference attempt, not a completed evaluation.

## Frozen handling

The 425-slice source-positive membership remains fixed. The five cases are never relabeled as negative, silently removed, or credited as correct empty predictions.

1. **Registered gate view — strict all-425 model-space view.** All 425 source-positive slices remain in the primary cohort. Each source-positive slice whose reference becomes empty at 128×128 receives Dice `0`, IoU `0`, boundary F1 `0`, and Hausdorff `NOT_ESTIMABLE`. The registered pituitary subject-macro Dice gate remains `>=0.75` and is evaluated on this strict view.
2. **Representable-only model-space view.** The 420 cases retaining foreground at 128×128 are reported separately as descriptive sensitivity analysis. They do not replace the strict gate view.
3. **Native source-coordinate slice view.** Frozen hard predictions are restored to source coordinates and compared with native label-1 masks for all 425 source-positive slices. Subject-macro and slice-macro Dice/IoU are reported descriptively.
4. **Whole-volume view.** Frozen predictions are restored for all 624 coronal slices, including reference-empty slices, and whole-volume Dice/IoU are reported as registered.
5. **Calibration regions.** Lesion-centred, background, and whole-image pixel-calibration summaries use the 420 model-space-representable cases because a lesion-centred region is undefined for an empty model-space reference. The exact five-case exclusion and denominator are mandatory in every calibration result.
6. **Reliability.** R02 scores remain prediction-only. Reliability evaluation uses the strict all-425 Dice-success target, so every target-representability loss is a failure (`Dice <0.50`) rather than an exclusion.
7. **Failure taxonomy.** Existing Phase 4 definitions remain unchanged for representable cases. The five cases additionally receive the explicit primary mode `target_not_representable_at_128`.
8. **Domain-gap mask features.** Model-space reference-mask geometry comparisons use the 420 representable cases and disclose that denominator. Non-mask input, encoder, reliability-score, and policy-rate comparisons use all 425 primary cases.

The whole-volume view is exploratory. It is not the primary gate because the segmenter was trained on annotated 2D slices rather than complete volumes.

The five nonrepresentable cases are written to `experiments/x02_openneuro_pituitary/artifacts/target_not_representable.csv` with subject identifier, coronal slice index, native tumor-pixel count, and zero post-resize pixels.

The completion report presents five named analytical views: strict all-425 model-space segmentation, representable-only 420-slice model-space segmentation, native source-coordinate segmentation on all 425 positive slices, exploratory whole-volume segmentation on all 624 slices, and confidence transport (pixel calibration on 420 representable slices plus case reliability on the strict 425-slice outcome). It reports classification separately.

Reliability interpretation must state that external pituitary transfer may be affected if target-representability loss is associated with prediction-derived R02 inputs. The five cases remain failures in reliability evaluation, and their reliability scores are never recalculated from a reference mask.

## Restart authorization

One clean X02 restart is authorized after code and tests enforce every rule above. The restart must be launched from a committed and pushed clean working tree. The result records one aborted attempt plus one completed attempt. Another verification failure stops X02 again.

## Constraints retained

- BRISC sealed-test pixels remain closed.
- Neural retraining, calibrator refitting, and reliability refitting remain prohibited.
- Image preprocessing remains the registered nonzero 0.5/99.5 percentile display bridge followed by the frozen resize and normalization.
- Mask preprocessing remains nearest-neighbor resize; no dilation, closing, minimum-component preservation, or other rescue transform is permitted.
- Segmentation threshold remains exactly `0.50`.
- Reliability-policy threshold remains exactly `0.9490250173235835`.
- The pituitary Dice gate remains `>=0.75`; no gate is relaxed.
- X02 remains research-only external transport evidence and cannot establish clinical validity.

## Decision authority

The user approved the proposed amendment and one clean restart by directing Codex to continue after receiving the fail-closed report. This amendment is registered before the restarted run and before any X02 performance result exists.
