# Largest connected component filtering

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E01

Keeping only the largest 8-connected component of each frozen v1 hard mask will reduce disconnected false-positive regions enough to improve glioma image-mean Dice by at least `0.02`, while meningioma and pituitary image-mean Dice each worsen by no more than `0.01`.

This is deterministic post-processing on the fixed validation predictions. It does not retrain the model or change the 0.50 threshold. Passing is a local development result and does not establish patient-independent or clinical generalization.
