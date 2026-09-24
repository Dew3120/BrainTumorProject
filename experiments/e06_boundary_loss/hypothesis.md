# Boundary-aware segmenter retraining

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E06

Fresh training with the v1 balanced BCE and foreground Dice losses plus Kervadec-style signed-distance boundary loss will improve glioma image-mean Dice by at least `0.05` and glioma boundary F1 by at least `0.05` over frozen v1.

The architecture, preprocessing, train/validation membership, optimizer, learning rate, weight decay, batch size, epoch budget, early stopping, checkpoint ranking, precision, seed, and absence of augmentation remain identical to v1. The three top-level loss terms have fixed coefficient `1.0`. Boundary F1 uses one-pixel predicted/reference boundaries and two-pixel matching tolerance. Passing is a local BRISC development result and does not establish patient-independent or clinical generalization.
