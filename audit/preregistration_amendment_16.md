# Pre-registration amendment 16 — UPENN-GBM curated derivative identity and label encoding

**Status:** REGISTERED BEFORE X03 PERFORMANCE INFERENCE

**Registered:** 2026-09-28

**Parent registration:** `audit/preregistration_amendment_13.md`, SHA256 `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5`

**X03 experiment:** `experiments/x03_upenn_gbm/`

This amendment resolves a source-encoding mismatch discovered by the fail-closed X03 dataset audit. The audit stopped on the first mask before model inference, model outputs, or performance metrics were produced. This amendment does not change H8, select subjects using outcomes, modify the frozen adapter, retrain or refit any model, change either threshold, relax any gate, or authorize clinical claims.

## Derivative release identity

- Repository: `https://github.com/data-nih/tcia`
- Release: `https://github.com/data-nih/tcia/releases/tag/upenn-gbm`
- Git tag: `upenn-gbm`
- Tag commit: `a8140bcfa512241fde2475cc3c4640b5cb97dc2b`
- GitHub release ID: `222317008`
- Published: `2025-06-02T02:42:11Z`
- Licence declared by the derivative release: `CC BY 4.0`
- Source collection: TCIA `UPENN-GBM`, version 2, DOI `10.7937/TCIA.709X-DN49`

The release describes itself as a redistribution prepared by the Pittsburgh Fiber Data Hub from the TCIA collection. Its release metadata states that the NIfTI volumes are skull-stripped and co-registered and that the tumor labels are automated and manually corrected expert segmentations from the UPENN-GBM collection. The project treats this release as a curated derivative, not as byte-identical proof of the original 69 GB TCIA Aspera archive.

## Fixed evaluation payload

The payload is the complete expert-labelled subset exposed by this derivative release:

- 147 expert segmentation volumes matching `sub-*_seg.nii.gz`.
- 147 paired contrast-enhanced T1 volumes matching `sub-*_ce-gd_T1w.nii.gz`.
- 294 files total; 427,673,840 bytes.
- Source-file manifest SHA256: `fe488d87a48bfcce1c4c735793ccccb28a9ba68a7d7400241692d7e45a68dfae`.
- Every file has already matched the byte count and SHA256 digest published by GitHub for release ID `222317008`.

The X03 audit must independently recheck all 294 local files against this frozen manifest before inference. Unexpected membership, size, or hash changes stop X03.

## Observed derivative label encoding

The derivative expert masks use the following sequential encoding:

| Value | Semantic region |
|---:|---|
| 0 | Background |
| 1 | Necrotic and non-enhancing tumor core (NCR/NET) |
| 2 | Peritumoral edematous or infiltrated tissue (ED) |
| 3 | Gadolinium-enhancing tumor (ET) |

The native BraTS/TCIA convention represents enhancing tumor as value `4`. The derivative recodes that value to `3` without changing its spatial membership. Before registration, this was verified on the source-equivalent expert mask for subject 002:

- Derivative `sub-002_seg.nii.gz` SHA256: `322c019da6f6bab864f80636e8663e7425581fe0a5bc78e139356bd532595093`.
- Source-equivalent `UPENN-GBM-00002_11_segm.nii.gz` SHA256: `27c2ec785c9d483065502b870a0cc7aeedff4e2b0bba89974186b0a52cba4b4c`.
- Both arrays have canonical shape `240 × 240 × 155`.
- The derivative array is bitwise identical to the source-equivalent array after the single mapping `4 -> 3`; labels 0, 1, and 2 are unchanged.
- Voxel counts match exactly: label 1 = 15,366; label 2 = 168,592; derivative label 3/source label 4 = 27,793.

This check establishes the semantic mapping. It is a provenance and target-contract check, not a performance result.

## Whole-tumor target

The fixed X03 reference target is the union of all three derivative tumor labels:

`whole_tumor = mask in {1, 2, 3}`

This is semantically the same whole-tumor region as native BraTS/TCIA `{1, 2, 4}`. No mask dilation, erosion, relabeling based on predictions, or outcome-dependent preprocessing is allowed.

## Category mapping and comparability

The registered mapping remains `glioblastoma -> glioma`. Glioblastoma is a subtype of glioma, so the mapping is biologically coherent but coarse. X03 is a GBM-only, atlas-aligned, expert-whole-tumor cohort and is not directly comparable to the heterogeneous BRISC glioma population without this caveat. Every X03 report and the X05 synthesis must state this limitation.

## Prospective confirmation and constraints

- No X03 classifier, segmenter, calibration, reliability, policy, Dice, AUROC, ECE, or other performance result was produced or inspected before this amendment.
- The stopped audit performed no model inference and wrote no audit decision or performance artifact.
- The complete 147-scan cohort remains fixed; no case was selected or excluded using outcomes.
- The classifier, E08 ensemble, C03 mapping, R02 estimator, shared adapter, and registered gates remain frozen.
- Segmentation threshold remains exactly `0.50`.
- Reliability-policy threshold remains exactly `0.9490250173235835`.
- BRISC sealed-test access, neural retraining, calibration refitting, reliability refitting, threshold changes, and gate relaxation remain prohibited.
- The X03 audit restarts from the beginning. It must be committed before the first performance-inference run.
