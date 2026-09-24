# Isotonic pixel calibration decision

**Decision:** ACCEPTED

The predeclared six-check conjunctive gate was satisfied and was not relaxed. Reported metrics use only out-of-fold probabilities. The unchanged threshold is `0.50`; hard-mask changes are reported descriptively and are not part of the C03 gate.

Reference masks were used only as supervised fitting targets and to define evaluation regions. The fitted mapping receives only a raw segmenter probability at inference.

No model was retrained, no threshold was changed, and no sealed-test image was accessed.
