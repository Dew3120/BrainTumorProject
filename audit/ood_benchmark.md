# OOD benchmark audit

O01 is complete with **2,680 images**: 1,300 detector-fitting images and 1,380 held-out evaluation images. The held-out ID cohort is the registered 680 positive BRISC development-validation slices. Each of seven OOD sources contributes 100 fitting and 100 evaluation images.

## Integrity

- SHA256 and a frozen 128×128 standardized-pixel SHA256 are recorded for every image.
- Fitting and evaluation groups are disjoint; near-OOD cohorts are patient-disjoint.
- No standardized-pixel duplicate crosses a benchmark class or split.
- No reference mask, diagnostic label, or Stage-6 performance value was used.
- BRISC sealed-test access: **0**.

## Provenance

Far OOD uses MedMNIST OrganAMNIST from Zenodo record 10519652, CIFAR-10 from the University of Toronto, and deterministic NumPy-generated random and uniform images. Archive MD5 and SHA256 values are stored in `result.json`. Near OOD uses the locally audited Cheng/Figshare, OpenNeuro ds006248 v1.0.0, and UPENN-GBM v2 releases.

## Scope limitation

The near-OOD cohorts were processed previously during Stage 6. They are held out from OOD-detector fitting/evaluation partitions, but are not historically unseen by the project. This benchmark tests registered retrospective separation, not clinical safety or universal OOD detection.
