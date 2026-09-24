# Glioma-weighted segmenter retraining

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E05

Fresh training with each glioma case contributing three times the loss weight of each meningioma or pituitary case will improve glioma image-mean Dice by at least `0.05` over frozen v1, while meningioma and pituitary image-mean Dice each worsen by no more than `0.01`.

The architecture, preprocessing, train/validation membership, optimizer, learning rate, weight decay, batch size, epoch budget, early stopping, checkpoint ranking, precision, seed, and absence of augmentation remain identical to v1. Only the normalized per-case training loss weights change. Validation loss is unweighted and checkpoint selection remains validation macro-type hard Dice. Passing is a local BRISC development result and does not establish patient-independent or clinical generalization.
