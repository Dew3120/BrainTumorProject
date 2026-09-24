# Two-scale inference

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E04

Averaging the frozen segmenter's 128×128 probability map with its 256×256 probability map resized to 128×128 will improve glioma image-mean Dice by at least `0.02`, while meningioma and pituitary image-mean Dice each worsen by no more than `0.01`.

The 256 probability map is resized with bilinear interpolation using `align_corners=False`. The arithmetic mean is thresholded at the unchanged `0.50` probability threshold. This experiment does not retrain the model. Passing is a local BRISC development result and does not establish patient-independent or clinical generalization.
