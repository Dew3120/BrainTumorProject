# X03 decision

**Audit decision:** `ELIGIBLE_INDEPENDENT`

All 147 expert-labelled CE-T1/mask pairs passed release-integrity, target, geometry, and overlap checks. One frozen-stack evaluation may proceed. Glioblastoma is mapped coarsely to glioma and is not directly comparable to heterogeneous BRISC glioma.

## Amendment 17 restart record

The original evaluation attempt stopped before model loading because 82 of 9,464 source-positive axial slices from 54 patients became empty after the frozen 240×240 to 128×128 nearest-neighbor target transform. Native targets contained 1–15 pixels. No probabilities, metrics, panels, or performance result were produced.

Amendment 17 authorizes exactly one restart. The strict primary view retains all 9,464 slices and assigns zero overlap scores to the 82 erased targets. The 9,382 representable slices and all 9,464 source-coordinate slices are reported separately. Preprocessing, interpolation, models, gates, and thresholds remain frozen.

## Amendment 18 loader-defect stop

The second attempt stopped when the shared NIfTI loader's unconditional `float32` conversion routed native `uint8` UPENN volumes through percentile windowing. Eighteen reference-empty slices from 14 patients had no usable percentile range; 14 were black and four contained one nonzero pixel. Nine complete batches, containing 144 slices, had passed through the frozen models. No probability, metric, panel, decision, or performance result was saved.

Amendment 18 classifies this as a loader defect because the registered adapter already requires native `uint8` data to use its `native_uint8` path. It authorizes a datatype-preserving loader correction, complete audit rerun, and exactly one third X03 attempt. Models, preprocessing operations, resize, interpolation, targets, gates, and thresholds remain unchanged.
