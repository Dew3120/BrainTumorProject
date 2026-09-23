# Glioma failure decomposition findings

## Result

The frozen v1 model's weak glioma performance has one strong evidenced predictor and several secondary model-output patterns.

**Small reference masks are the clearest predictor of failure.** Of 32 small glioma masks under 100 model-space pixels, 21 fail (65.6%) and mean Dice is 0.393. The failure-rate difference against the other size groups is +0.430, with a similarity-group bootstrap 95% interval of +0.258 to +0.592. Medium masks fail in 27 of 135 cases (20.0%); large masks fail in 12 of 37 (32.4%).

**Failed predictions are commonly fragmented and spatially wrong.** Among the 60 Dice-below-0.50 gliomas, 50 (83.3%) have more than one raw predicted component and 44 (73.3%) retain more than one component after excluding components under four pixels. Twelve are non-empty predictions with zero reference overlap; none is an empty prediction. Median failed-case centroid error is 16.4 pixels in the 128×128 model space. Sixteen failures with some overlap have a prediction/reference area ratio below 0.5, while 15 exceed 2.0. The average largest-component fraction is 0.759, so many failures contain substantial disconnected foreground outside the dominant region.

**The predeclared boundary, focality, and location splits do not independently clear the association rule.** Diffuse cases fail more often than sharp cases (55/173, 31.8%, versus 5/31, 16.1%), and multifocal references fail more often than single-focal references (6/10, 60.0%, versus 54/194, 27.8%). Their group-bootstrap risk-difference intervals include zero, narrowly in both comparisons. Peripheral references also have a higher raw failure rate than central references (36.8% versus 28.6%), but its interval includes zero. These patterns are hypotheses for future data, not confirmed predictors here.

**A shared preprocessing defect was not found.** All 680 validation predictions reproduced the stored metrics exactly; the maximum difference was 0.0. All six 10-case visual panels preserve image/reference alignment, and no reference or prediction touches the 128×128 model edge. The errors vary across axial, coronal, and sagittal slices and appear as model localization, fragmentation, under-coverage, and over-coverage rather than a repeated flip, shift, stretch, or file-pairing problem.

## Interpretation

The strongest supported explanation is a combination of **resolution-sensitive small targets** and **unstable spatial/topological predictions**. Weak boundary contrast and multifocality remain plausible contributors, but this internal sample does not estimate them precisely enough to call them confirmed predictors. This evidence justifies the registered component filtering, morphology, multi-scale inference, TTA, glioma weighting, and boundary-loss experiments. It does not predict that any experiment will pass its gate.

Edema, enhancement pattern, and ventricle proximity are recorded as not assessable or indeterminate. BRISC provides unverified single JPEG slices without the full MRI sequences and clinical acquisition context required for reliable assessment of those properties.

## Limits

- All evidence is from the reused BRISC internal validation partition.
- Patient independence is unverified.
- The size thresholds are model-space pixels, not physical units.
- Associations are univariable and descriptive; they are not causal effects.
- A failed Dice score and a component pattern do not establish tumor biology.
- No sealed test image was accessed.
