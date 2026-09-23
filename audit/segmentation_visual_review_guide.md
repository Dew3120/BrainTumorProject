# Segmentation visual review guide

Review the exact 128×128 model input, the reference mask in red, and the prediction in cyan. A bad cyan prediction is a model failure. Treat it as a pipeline defect only when the image, reference mask, pairing, or preprocessing geometry is wrong.

| Technical defect indicating a pipeline bug | Genuine segmentation difficulty |
|---|---|
| Reference mask is shifted away from the visible lesion or anatomy. | Tumor boundary is diffuse, irregular, or clinically ambiguous. |
| Image or mask appears horizontally or vertically flipped. | Tumor contrast against surrounding tissue is low. |
| Reference mask clearly belongs to a different MRI image. | Lesion occupies very few pixels. |
| Mask is rotated relative to the MRI anatomy. | Tumor crosses tissue with similar intensity. |
| Anatomy is stretched because aspect ratio was lost. | Tumor intensity is heterogeneous. |
| Letterbox padding enters the anatomy or prediction unexpectedly. | Anatomy, location, or tumor appearance is unusual. |
| Reference mask is clipped systematically at an image edge. | Partial-volume effects make the boundary uncertain. |
| Boundaries have a consistent one-pixel displacement in the same direction. | Prediction finds the correct area but under-segments an uncertain margin. |
| Mask edges are blurred, gray, or non-binary after resizing. | Prediction includes a plausible nearby structure with similar appearance. |
| Filename, category, image, and mask do not describe the same case. | Reference is aligned but the model misses, over-predicts, or fragments the lesion. |

## Review sequence

1. Confirm the input anatomy is readable and preserves its aspect ratio.
2. Confirm the red reference lies on the visible lesion and shares the same orientation as the MRI.
3. Check whether the cyan prediction overlaps the red reference or appears elsewhere.
4. Look for systematic shifts, flips, edge clipping, or padding-related predictions across multiple cases.
5. Record isolated poor predictions as task/model failures when the input and red reference are technically correct.
6. Compare the worst and best panels for repeated technical patterns rather than judging a single image alone.

The panels are diagnostic displays, not medical annotations. Boundary quality must not be interpreted as clinical validity.
