# Glioma-specialist routed segmentation

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E07

A fresh segmenter trained only on the 817 glioma training pairs will improve routed glioma image-mean Dice by at least `0.08` over frozen v1 when routing is performed by the frozen classifier's predicted four-class label. The routed system must also achieve classifier routing accuracy of at least `0.95`, while meningioma and pituitary Dice may each decline by no more than `0.01` relative to frozen v1.

The specialist uses the v1 architecture, preprocessing, balanced BCE plus foreground Dice loss, optimizer, learning rate, weight decay, batch size, 75-epoch budget, patience, precision, seed, threshold, and no-augmentation policy. Its checkpoints are ranked only by image-mean hard Dice on the 204 glioma validation pairs. At evaluation, a case uses the specialist only when the frozen classifier argmax is `glioma`; every other predicted class, including `no_tumor`, uses the frozen v1 segmenter. Routing accuracy is exact classifier-predicted class agreement with the dataset tumor type across all 680 positive-mask validation cases.

Passing is a local BRISC internal-validation result. It does not establish patient-independent, cross-dataset, or clinical generalization.
