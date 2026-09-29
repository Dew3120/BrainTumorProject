# Pre-registration amendment 18 — preserve native NIfTI datatype and authorize X03 attempt three

**Status:** REGISTERED AFTER THE SECOND PRE-PERFORMANCE X03 VERIFICATION STOP AND BEFORE ANY X03 PERFORMANCE RESULT

**Registered:** 2026-09-29

**Parent registration:** `audit/preregistration_amendment_13.md`, SHA256 `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5`

**Target-representability registration:** `audit/preregistration_amendment_17.md`, SHA256 `3c7941e388f8c095884d7bd62bc1886fb3aa8a1b5003520f76cdcbffd6ee12b7`

This amendment corrects a loader defect discovered by a fail-closed assertion before any X03 result was saved. The registered adapter already specifies that native `uint8` inputs use the `native_uint8` path. The shared NIfTI loader incorrectly converted every canonical volume to `float32`, hiding the native datatype and sending UPENN-GBM through the nonzero-percentile path. Correcting that procedural defect restores the registered adapter contract; it does not change a gate, model, threshold, resize, interpolation method, target, or hypothesis.

## Triggering verification stop

The second X03 attempt reached 18 reference-empty axial slices from 14 patients that had no usable percentile range after the erroneous `uint8` to `float32` conversion. Fourteen slices were all zero and four contained one nonzero pixel; 12 were volume-edge slices. Nine complete batches, containing 144 of 22,785 slices, passed through the frozen models before the next loader batch stopped. No probabilities, metrics, panels, decisions, or performance result were saved.

## Loader correction

1. `load_canonical_nifti()` must preserve the canonical volume's native array datatype rather than unconditionally calling `get_fdata(dtype=float32)`.
2. Native `uint8` UPENN-GBM volumes remain `uint8` through slice extraction and enter the registered `native_uint8` display path.
3. Geometry, RAS canonicalization, finite-value checks, and affine checks remain unchanged.
4. A regression test must create and load a constant-valued `uint8` NIfTI volume, confirm the loaded array remains `uint8`, and confirm both registered preprocessors accept its slice without rejection.
5. Existing non-`uint8` images continue to use the registered nonzero 0.5/99.5 percentile bridge.

## Audit and attempt-three authorization

After this amendment is committed, X03 must rerun its audit from the beginning against the same registered derivative release: the same 294 files, published SHA256 values, labels `{0,1,2,3}`, 147 subjects, identifiers, and decision criteria. The renewed audit decision must be committed before inference.

One third X03 attempt is authorized after the renewed audit:

- Attempt 1 stopped before inference after detecting the 82 Amendment-17 target losses.
- Attempt 2 stopped after 144 inferred slices when the loader datatype defect rejected 18 constant or effectively constant reference-empty slices.
- Attempt 3 must use the corrected native-datatype loader and all Amendment-17 metric views.
- Any further fail-closed condition stops X03 again; no automatic restart is authorized.

## Constraints retained

- BRISC sealed-test pixels remain closed.
- The registered image preprocessing contract is unchanged; only correct datatype routing is restored.
- The 240×240 to 128×128 adapter and interpolation methods remain unchanged.
- Target preprocessing remains nearest-neighbor with no dilation, closing, or component preservation.
- Neural retraining, calibrator refitting, and reliability refitting remain prohibited.
- Segmentation threshold remains exactly `0.50`.
- Reliability-policy threshold remains exactly `0.9490250173235835`.
- Gates and Amendment-17 denominators remain unchanged.

## Decision authority

The user explicitly approved Amendment 18, the native-datatype loader correction, the regression test, a complete renewed X03 audit, and exactly one third X03 attempt.
