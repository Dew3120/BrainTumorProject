# X03 decision

**Audit decision:** `ELIGIBLE_INDEPENDENT`

All 147 expert-labelled CE-T1/mask pairs passed release-integrity, target, geometry, and overlap checks. One frozen-stack evaluation may proceed. Glioblastoma is mapped coarsely to glioma and is not directly comparable to heterogeneous BRISC glioma.

## Aborted attempts and Amendment 18 restart

Attempt 1 stopped before model loading because 82 of 9,464 source-positive axial slices from 54 patients became empty after the frozen 240×240 to 128×128 nearest-neighbor target transform. Amendment 17 retains them in the strict primary view with zero overlap scores and reports 9,382 representable slices separately.

Attempt 2 stopped after nine complete batches and 144 inferred slices because the shared NIfTI loader converted native `uint8` volumes to `float32`. Eighteen reference-empty slices from 14 patients then entered percentile windowing and were rejected for having no usable range. No probabilities, metrics, panels, decisions, or performance result were saved from either attempt.

Amendment 18 corrected the loader to preserve canonical native datatypes. A constant-`uint8` regression test confirms both registered preprocessors now use `native_uint8`. The complete renewed audit passed on the same 294 release files, labels, patients, geometry, and decision criteria. Exactly one third X03 evaluation attempt is authorized.
