# X03 UPENN-GBM dataset audit

**Outcome:** `ELIGIBLE_INDEPENDENT`  
**Inference performed:** No

## Release and integrity

- Frozen source: UPENN-GBM version 2, DOI `10.7937/TCIA.709X-DN49`, CC BY 4.0.
- The minimal registered payload contains all 147 expert segmentation volumes and their paired CE-Gd T1w volumes: 294 files, 0.398 GiB.
- Every file matches the release asset's published byte count and SHA256 digest.
- Cohort membership is 147 scans from 147 patient identifiers. Repeated scan suffixes remain grouped by base patient identifier for all future bootstrap calculations.

## Target and geometry

- The reference target is whole tumor: union of derivative labels 1 (necrotic/non-enhancing core), 2 (edema), and 3 (enhancing tumor). Amendment 16 verifies that derivative label 3 is an exact recoding of native label 4.
- Primary evaluation is the 9464 whole-tumor-bearing axial slices after canonical RAS orientation. Whole-volume analysis is secondary.
- Shapes, affines, finite values, nonconstant images, integer labels, image/mask pairing, tumor presence, and exact file integrity pass for every scan.
- Exact duplicate image volumes: 0; exact duplicate masks: 0.
- Source image dtypes: `{'uint8': 147}`; source mask dtypes: `{'uint8': 147}`.
- Whole-tumor 26-connected 3D component-count distribution: `{1: 41, 2: 27, 3: 19, 4: 9, 5: 12, 6: 13, 7: 5, 8: 4, 9: 1, 10: 3, 11: 4, 12: 2, 15: 3, 17: 1, 19: 1, 20: 1, 31: 1}`.

## Independence audit

- Exact display-pHash candidates against frozen BRISC glioma, X01 glioma, and X02 pituitary metadata: 0.
- Distance ≤2 candidates: 0; distance ≤4 candidates: 7; minimum distance: 4.
- The audit opened zero BRISC pixels. A deterministic 12-scan alignment panel shows no recurring flip, shift, transpose, or detached reference-mask pattern.
- X04 comparison is pending controlled access; no X04 pixels or hashes are available. If X04 access is granted, its audit must perform the reciprocal X03 comparison before X04 inference.

## Required mapping caveat

Glioblastoma is a subtype of glioma. The mapping `glioblastoma -> glioma` is coarse. This GBM-only, atlas-aligned, whole-tumor cohort is not directly comparable to heterogeneous BRISC glioma, and every X03 result must repeat that limitation.

## Disk and decision

- Drive E free after audit: 324.238 GiB. The 250 GB sequential-processing trigger is not reached.
- The frozen stack may proceed to one X03 evaluation under `ELIGIBLE_INDEPENDENT`. No model, calibration mapping, reliability estimator, mask threshold, policy threshold, or gate is changed.
