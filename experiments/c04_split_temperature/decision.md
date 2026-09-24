# Separate-region temperature calibration decision

**Decision:** ACCEPTED_LOCALLY_ORACLE_ONLY

The predeclared four-check conjunctive gate was satisfied and was not relaxed. Reported metrics use only out-of-fold probabilities. The hard threshold remained `0.50`.

C04 uses the reference mask to route pixels between temperatures. It is an oracle diagnostic, is unavailable during real inference, and cannot satisfy H7 regardless of its local result.

No model was retrained and no sealed-test image was accessed.
