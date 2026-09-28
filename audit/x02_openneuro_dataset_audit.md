# X02 OpenNeuro pituitary dataset audit

**Outcome:** `ELIGIBLE_INDEPENDENT`  
**Independent external evidence:** Yes  
**Inference performed:** No

## Release and integrity

- Official snapshot: OpenNeuro `ds006248` version `1.0.0`, DOI `10.18112/openneuro.ds006248.v1.0.0`.
- Snapshot commit/tag: `1b8ebd1f1171049c7f4258ea41933e7b8ee6c06e` / `1.0.0`.
- Exactly 100 required payloads were retrieved: one coronal CE-T1w image and one ground-truth segmentation for each of 50 subjects.
- Every payload matches the byte count and SHA256 encoded in its git-annex pointer. Required payload size is 0.270 GiB.

## Licence discrepancy

The same frozen release contains conflicting licence statements: BIDS `dataset_description.json` says `CC0`, while `README.md` says `CC BY-NC 4.0`. The audit does not silently choose the less restrictive declaration. Project handling follows `CC BY-NC 4.0` unless the publisher resolves the discrepancy. This governance issue does not alter technical eligibility, but it is binding for redistribution and downstream use.

## Cohort and target

- 50 unique subjects; no patient-level train/evaluation split is constructed because the entire independent cohort is frozen evaluation data.
- Source sequence: coronal contrast-enhanced T1, 3T GE 750w (`COR T1+C`).
- Expert ground truth is a multi-label mask. Only label `1` (tumor) is the registered target; labels 2–5 are excluded from the tumor target.
- Canonical RAS geometry is used. Primary evaluation uses the 425 tumor-bearing coronal slices along canonical axis 1. Whole-volume analysis including empty-reference slices is secondary.
- Image and mask shapes match for every subject; maximum affine discrepancy is within `1e-4`. All arrays are finite, all images are nonconstant, all masks use integer labels 0–5, and every subject contains tumor label 1.
- Tumor-bearing slices per subject: minimum 5, median 8.0, maximum 12.

## Independence and duplicate audit

- Exact duplicate image volumes across subjects: 0.
- Exact duplicate mask volumes across subjects: 0.
- Repeated tumor-bearing slice arrays beyond first occurrence: 0.
- BRISC frozen-metadata pHash candidates: exact 0, distance ≤2 0, distance ≤4 0; minimum distance 12.
- X01 Cheng pHash candidates: exact 0, distance ≤2 0, distance ≤4 0; minimum distance 12.
- No BRISC pixels were opened. The comparison uses only the existing frozen metadata inventory.
- A deterministic panel of 12 evenly spaced subjects was reviewed at each subject's maximum-tumor-area coronal slice. Label 1 remained aligned with visible sellar anatomy; no recurring flip, shift, transpose, or detached-mask pattern was found.

## Disk

- Required image/mask payload: 0.270 GiB.
- Drive E free after audit: 324.645 GiB.
- The projected Stage 6 footprint remains below the 250 GB sequential-processing trigger at this step.

## Decision

The frozen stack may proceed to one X02 evaluation under `ELIGIBLE_INDEPENDENT`. Existing model, calibration, segmentation-threshold, and reliability-policy values remain frozen. Results are research transport evidence only and cannot establish clinical validity.
