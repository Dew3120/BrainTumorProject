# Eight-view test-time augmentation

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E03

Averaging inverse-mapped probability maps from the fixed eight-view transform set will improve frozen-v1 glioma image-mean Dice by at least `0.03`, while meningioma and pituitary image-mean Dice each worsen by no more than `0.01`.

The fixed views are identity, rotations by 90/180/270 degrees, horizontal flip, vertical flip, main-diagonal transpose, and anti-diagonal transpose. Each probability map is returned to the original coordinate system before arithmetic averaging and thresholding at the unchanged `0.50` probability threshold. This experiment does not retrain the model. Passing is a local BRISC development result and does not establish patient-independent or clinical generalization.
