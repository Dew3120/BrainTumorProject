# X03 decision

**Audit decision:** `ELIGIBLE_INDEPENDENT`

All 147 expert-labelled CE-T1/mask pairs passed release-integrity, target, geometry, and overlap checks. One frozen-stack evaluation may proceed. Glioblastoma is mapped coarsely to glioma and is not directly comparable to heterogeneous BRISC glioma.

## Amendment 17 restart record

The original evaluation attempt stopped before model loading because 82 of 9,464 source-positive axial slices from 54 patients became empty after the frozen 240×240 to 128×128 nearest-neighbor target transform. Native targets contained 1–15 pixels. No probabilities, metrics, panels, or performance result were produced.

Amendment 17 authorizes exactly one restart. The strict primary view retains all 9,464 slices and assigns zero overlap scores to the 82 erased targets. The 9,382 representable slices and all 9,464 source-coordinate slices are reported separately. Preprocessing, interpolation, models, gates, and thresholds remain frozen.
