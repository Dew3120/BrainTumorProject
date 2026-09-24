# Classifier-routed per-category pixel calibration decision

**Decision:** REJECTED

The predeclared nine-check conjunctive gate was not satisfied and was not relaxed. Reported metrics use only out-of-fold probabilities. Routing used the frozen classifier prediction, never true tumor type.

The single no-tumor prediction used the preregistered global fallback and did not control the gate. The fixed threshold remained `0.50`.

No model was retrained and no sealed-test image was accessed.
