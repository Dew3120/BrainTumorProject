# Fixed three-system probability ensemble

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E08

An equal-weight arithmetic mean of the frozen E05 weighted-model, E06 boundary-model, and E07 classifier-routed specialist probability maps will improve glioma image-mean Dice by at least `0.02` over the best category-harm-eligible E05–E07 single system. Meningioma and pituitary Dice may each decline by no more than `0.01` relative to frozen v1.

Eligibility follows amendment 01's final-selection filter: exclude a candidate only when it violates a category-harm constraint registered for that candidate. E05 is excluded because it violated both registered harm controls; E06 had no category-harm constraint; E07 passed both harm controls. E07 is the fixed single-system comparator because its routed glioma Dice (`0.5802066652`) exceeds E06 (`0.5793657813`). The ensemble therefore requires glioma Dice of at least `0.6002066652`.

Every component is frozen. E05 and E06 contribute their raw sigmoid probability maps. E07 contributes its already frozen routed probability map: specialist only when the frozen classifier predicts `glioma`, otherwise v1. The three maps receive weight `1/3`, are averaged at 128×128, and are thresholded once at the unchanged `0.50`. No component, weight, route, threshold, or checkpoint will be selected after viewing ensemble results.

Passing is a local BRISC internal-validation result. It does not establish patient-independent, cross-dataset, or clinical generalization, and it does not by itself replace frozen v1.
