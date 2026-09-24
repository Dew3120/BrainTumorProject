# Boundary-aware retraining decision

**Decision:** REJECTED

The selected epoch-40 candidate changed glioma Dice by `-0.0296` and glioma boundary F1 by `-0.0470`. The fixed gate required both changes to be at least `+0.0500`. The conjunctive gate failed and was not relaxed.

All required follow-up checks completed. Visual review found no systematic technical defect. There were 88 cases below Dice 0.50. Pixel calibration failed its locked gate; case reliability passed its locked gate.

No sealed-test image was accessed, and this internal BRISC result is not clinical validation.
