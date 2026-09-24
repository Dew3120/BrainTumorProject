# Test-time augmentation decision

**Decision:** REJECTED

Eight-view TTA failed the preregistered conjunctive gate and is closed without relaxing it.

Glioma Dice delta: `-0.1603` (required `+0.0300`).
Meningioma Dice delta: `+0.0075` (minimum `-0.0100`).
Pituitary Dice delta: `-0.2580` (minimum `-0.0100`).

No model weight or probability threshold changed, and no sealed-test image was accessed.
