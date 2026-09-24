# Reliability-weighted pixel calibration decision

**Decision:** REJECTED

The predeclared three-check conjunctive gate was not satisfied and was not relaxed. Reported metrics use only out-of-fold probabilities. The final all-validation temperature is an operational proposal only.

The fitting weights came from frozen v1 out-of-fold prediction-only reliability scores. Dice and reference-mask geometry did not enter the weights. The fixed threshold remained `0.50`.

No model was retrained and no sealed-test image was accessed.
