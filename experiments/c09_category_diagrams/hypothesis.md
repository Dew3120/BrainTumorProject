# Per-category reliability diagrams

**Status:** REGISTERED — NOT STARTED

**Experiment:** Calibration study C09

C09 produces six descriptive 15-bin equal-frequency reliability diagrams: lesion-centred pixel calibration and case-level segmentation reliability, each for glioma, meningioma, and pituitary.

The pixel diagrams use C03 isotonic calibration because it is the operational C01–C07 method that passed its registered gate and has the lowest whole-image out-of-fold ECE under amendment 02's fixed ranking. C04 and C07 are excluded despite local passes because reference-mask routing makes them non-operational. Pixel calibration is reconstructed from C03's saved fold mappings and evaluated only on each case's held-out fold. Reference masks define the displayed lesion-centred evaluation region but never select the isotonic transform.

The case-level diagrams use the accepted v1 reliability model's frozen OOF scores and observed `Dice >=0.50` outcomes. C09 fits nothing, changes no threshold, revises no prior decision, retrains no model, and accesses no sealed-test image.
