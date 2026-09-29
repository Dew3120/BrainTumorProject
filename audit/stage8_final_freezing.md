# Stage 8 final freezing completion report

**Status:** COMPLETE

**Freeze commit:** `c2566214bb921c7185fc8949b3e0376492e7113e`

**Documentation commit:** this report's containing commit

**Frozen manifest:** `runs/final_stack/manifest.json`

**Manifest SHA256:** `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`

**Frozen package size:** 52,100,071 bytes

## Final selection

| Role | Frozen selection | Operational definition |
|---|---|---|
| Classifier | v1 ResNet-18 | Four classes in the fixed order glioma, meningioma, pituitary, no_tumor; raw softmax probabilities remain in force. |
| Segmenter | E08 ensemble | Arithmetic mean of E05, E06, and one classifier-routed component. Predicted glioma routes to E07; every other predicted class routes to v1. |
| Pixel calibrator | C03 isotonic | The registered `final_all_validation_mapping` with 501 knots. |
| Case reliability | R02 reduced logistic | Frozen StandardScaler plus L2 logistic regression using the registered 10 prediction-only features in fixed order. |
| OOD detector | O05 secondary logistic | Logistic ensemble of empirical-CDF-normalized O02, O03, and two O04 scores. |
| OOD policy | O06 | `ACCEPT` below the review threshold and `REVIEW` at or above it. REFUSE remains disabled. |

H7 is **satisfied** on both registered conditions: C03 calibration passed its registered gate, and selected R02 reliability AUROC `0.9555020801044132` exceeded the `0.941` comparator. H9 is **satisfied**: selected O05 held-out AUROC was `0.9957184873949579`, above the registered `0.92` gate.

## Frozen thresholds

The manifest stores policy values as full-precision strings so application parsing cannot silently round them.

| Threshold | Frozen value |
|---|---:|
| Segmentation hard mask | `0.50` |
| Case reliability | `0.9490250173235835` |
| OOD review | `0.36854275476100085` |
| OOD refuse | disabled / `null` |

No registered threshold changed during Stage 8.

## Package contents

The package contains 31 allowlisted artifacts plus its manifest:

- Five neural checkpoints: classifier v1, E05, E06, E07, and v1 segmenter fallback.
- Model and selection configs for the classifier, four segmenter components, E08, C03, and O05.
- The complete C03 isotonic mapping.
- The R02 scaler, logistic estimator, feature order, scaler state, coefficients, intercept, and provenance manifest.
- O02 logistic state, reconstructed O03 Ledoit-Wolf state, O05 empirical-CDF references and logistic state, and O06 policy/message catalogue.
- A self-contained inference runtime, locked Python project files, three synthetic fixtures, and expected reconstruction outputs.

Every non-manifest file is listed by relative path, SHA256, byte size, purpose, and source provenance where applicable. Runtime loading rejects missing and unexpected files.

## Verification

All fail-closed checks passed:

1. All five source checkpoint hashes matched their registered SHA256 values before copying and after copying.
2. The C03 and R02 source artifacts matched their registered SHA256 values.
3. O03 was deterministically recovered from its registered 600 BRISC fitting embeddings using Ledoit-Wolf. Its squared Mahalanobis distances reproduced the recorded vector with maximum absolute error `0.0`, passing the `1e-12` gate.
4. O02 probabilities reconstructed from the frozen scaler, coefficients, and intercept within `1e-12`.
5. O05 selected probabilities reconstructed from its 600-case fit-ID empirical-CDF references and secondary-logistic state with maximum absolute error `0.0`.
6. The manifest threshold strings exactly match the registered values.
7. Three synthetic fixtures reconstructed the classifier, E08 routing and probability maps, C03 mapping, hard mask, R02 score, O02/O03/O04 component scores, O05 score, and O06 policy twice within `1e-12`. Hard-mask and probability-map hashes matched exactly.
8. The strict allowlist found no unexpected file in the frozen package.

The first staging attempt stopped before promotion because importing the copied runtime generated an unlisted `__pycache__` file. No model, threshold, hash, or numerical reproduction mismatch occurred. The loader was corrected to prohibit bytecode writes inside the package; the empty staging build was then repeated and passed. This preserves the fail-closed audit trail.

Verification commands:

```powershell
uv run python scripts/verify_final_stack.py
uv run ruff check .
$env:PYTHONPATH=((Resolve-Path '.').Path + ';' + (Resolve-Path 'scripts').Path)
uv run pytest -q
```

Final results:

- Frozen-stack verifier: **PASS**
- Tests: **207 passed**
- Ruff: **PASS**
- BRISC test-image access: **0**
- Neural retraining: **0**
- Segmentation-threshold changes: **0**
- Reliability- or OOD-threshold changes: **0**
- Gate relaxations: **0**

## Excluded alternatives

- E01-E04 are excluded entirely.
- E05-E07 are excluded as standalone final segmenters. Their checkpoints remain only because they are required components of selected E08.
- C01, C02, and C04-C07 are excluded; only C03 is operational.
- R01 and R03-R10 are excluded from operation; only R02 is operational.
- O02-O04 are excluded as standalone policy choices. Their frozen states remain only because the selected O05 ensemble requires their component scores.
- The O05 primary equal-weight average is excluded; the secondary logistic ensemble is selected.
- O06 REFUSE is excluded because its registered enablement rule failed; only ACCEPT and REVIEW are active.

## Scope and limitations

The freeze proves that the registered research stack can be reconstructed deterministically from a complete, hash-verified package. It does not establish clinical validity, safety, prospective performance, patient-independent BRISC evaluation, or generalization to a new hospital. Stage 6 rejected H8 on the evaluated independent cohorts, and that external-domain limitation remains binding. Glioma remains the weakest reliability subgroup. The OOD benchmark is retrospective and does not cover every invalid or shifted input.

**Stage 8 complete. Ready for Stage 9.**
