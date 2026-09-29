# X03 decision

**Audit decision:** `ELIGIBLE_INDEPENDENT`

All 147 expert-labelled CE-T1/mask pairs passed release-integrity, target, geometry, and overlap checks. One frozen-stack evaluation may proceed. Glioblastoma is mapped coarsely to glioma and is not directly comparable to heterogeneous BRISC glioma.

## Aborted first evaluation attempt

Before model loading or inference, the registered target-representability assertion found that 82 of 9,464 source-positive axial slices from 54 patients became empty after the frozen 240×240 to 128×128 nearest-neighbor mask transform. Native targets contained 1–15 pixels. The run stopped under the registered fail-closed instruction. No model output, performance metric, panel, or evaluation result was produced.

The user authorized Amendment 17, a complete audit restart, and exactly one frozen evaluation restart. The amendment keeps all 9,464 cases in the strict primary metric with zero overlap scores for the 82 erased targets, reports the 9,382 representable cases separately, and reports all 9,464 cases at native source resolution. Preprocessing, interpolation, models, gates, and thresholds remain unchanged.
