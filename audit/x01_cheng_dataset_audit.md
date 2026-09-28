# X01 Cheng/Figshare v8 dataset audit

**Outcome:** `ELIGIBLE_OVERLAP_CONTROL`  
**Independent external evidence:** No  
**Inference performed:** No

## Release and integrity

- Official article: Figshare `1512427`, version `8`, DOI `10.6084/m9.figshare.1512427.v8`.
- Licence: CC BY 4.0.
- All six official source files match their registered byte counts and MD5/SHA256 values.
- All four ZIP archives pass CRC testing and contain exactly `1.mat` through `3064.mat` once.
- All 3,064 extracted MATLAB v7.3/HDF5 files are present and readable.

## Cohort

- 3,064 contrast-enhanced T1 slices from 233 patient identifiers.
- Category slices: glioma 1,426; meningioma 708; pituitary 930.
- Category patients: {"glioma": 89, "meningioma": 82, "pituitary": 62}.
- The official `cvind` file has five folds. Each patient identifier belongs to exactly one fold; no patient crosses folds and no patient has multiple labels.

## Image, mask, and geometry checks

- Every image and mask pair has matching dimensions, is finite, and is nonblank; every mask is binary and nonempty. Dimensions are {"256x256": 15, "512x512": 3049}.
- MATLAB v7.3/HDF5 reverses the stored matrix dimensions. Arrays are transposed before use. This was checked against the supplied tumor-border coordinates for every case.
- Minimum tumor-border alignment within a two-pixel tolerance: 1.0000; median: 1.0000.
- Per-slice orientation is not encoded. The README describes axial, coronal, and sagittal images globally but does not identify each slice's orientation.
- The README documents 0.49×0.49 mm in-plane spacing, 6 mm thickness, and 1 mm gap. These values are cohort documentation, not per-file geometry. Volume and skull-depth calculations are prohibited because slice order and orientation are unavailable.

## BRISC overlap decision

The official BRISC paper states that BRISC is derived from the Nickparvar aggregate, which includes the Cheng dataset. That provenance is sufficient to classify this cohort as an overlap control. It cannot be reported as independent external validation.

The audit additionally compared Cheng display pHashes with the already frozen BRISC inventory. It did not open any BRISC pixels, including sealed test pixels. Source slices with a same-label BRISC candidate were: exact pHash 1444, distance ≤2 1660, and distance ≤4 1735. Re-encoding and source conversion can change hashes; these candidates support review but do not replace the documented provenance conclusion.

## Duplicate and storage observations

- Repeated image arrays beyond first occurrence: 0.
- Repeated mask arrays beyond first occurrence: 0.
- Raw source footprint: 0.819 GiB.
- Extracted footprint: 0.859 GiB.
- Drive E free after audit: 324.927 GiB.

## Decision

The frozen stack may be evaluated once on this cohort as X01 overlap-control reproduction. Results must be labeled overlap control and cannot support cross-dataset independence or clinical-generalization claims. Patient identifiers must be the bootstrap unit. All existing thresholds and models remain frozen.
