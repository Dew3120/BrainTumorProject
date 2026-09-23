# Segmentation uncertainty analysis

This analysis uses the accepted classifier and segmenter on the held-out BRISC validation partition. It compares all 17 zero-Dice cases with the 20 highest-Dice cases. No model was retrained, thresholds remained at 0.50, and no test image was accessed.

## Verification

- Shared split SHA256: `d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28`
- Classifier checkpoint SHA256: `3bc044027b83b2de225ebde3fb817be8827d2b427c599824697b13ff7a727535`
- Segmenter checkpoint SHA256: `16DC8745993EEF959D0F2924C9147A45320673F6FC0B465AB08F8AD7B2F723DE`
- Classifier maximum metric/probability difference: 0.0 / 0.0
- Segmenter maximum metric/per-image difference: 0.0 / 0.0
- Test images accessed: 0

## Group comparison

| Signal (median) | 17 zero-Dice | 20 highest-Dice |
|---|---:|---:|
| Classifier predicted-class confidence | 0.9964 | 0.9999 |
| Classifier true-class probability | 0.9939 | 0.9999 |
| Segmenter mean probability inside reference | 0.0005 | 0.9795 |
| Segmenter max probability inside reference | 0.0207 | 1.0000 |
| Segmenter max probability outside reference | 1.0000 | 0.9994 |
| Segmenter whole-image mean probability | 0.0096 | 0.0347 |

Zero-Dice category counts: {'glioma': 12, 'meningioma': 2, 'pituitary': 3}. Highest-Dice category counts: {'glioma': 1, 'meningioma': 15, 'pituitary': 4}.

The groups have very different tumor-type composition: zero-Dice cases are mostly glioma, whereas the highest-Dice cases are mostly meningioma. Their descriptive confidence differences therefore combine failure severity with tumor-type mix and must not be interpreted as a controlled calibration comparison.

## Findings

### 1. Low confidence or confident-wrong?

The zero-Dice segmentations are spatially wrong rather than empty. Their median mean probability inside the true reference is 0.0005, while their median maximum probability outside the reference is 1.0000. Every zero-Dice case has a non-empty hard prediction outside the reference. The classifier is correct in 16 of 17 cases; the remaining case predicts no tumor with 0.999985 confidence and assigns approximately 0.00000025 probability to the true meningioma class. The pixel-level pattern combines extreme under-confidence on the true lesion with extreme over-confidence elsewhere.

### 2. Confidence in highest-Dice versus zero-Dice cases

Median classifier predicted-class confidence is 0.9964 for zero-Dice cases and 0.9999 for the highest-Dice cases; the distributions are similar. In contrast, median segmenter probability inside the reference rises from 0.0005 to 0.9795, so the segmenter's spatial signal separates the groups much more clearly than classifier confidence. Maximum probability outside the reference is near 1.0 in both groups, showing that a single outside-mask maximum is saturated and is not a useful standalone failure detector.

### 3. Classifier/segmenter decoupling

Across all 680 paired validation cases, Spearman correlation between classifier predicted-class confidence and segmenter Dice is 0.3252; correlation with segmenter mean probability inside the reference is 0.1660. These are weak-to-moderate and weak relationships, respectively. By comparison, segmenter mean inside-reference probability correlates with its own Dice at 0.5428. classifier/segmenter agreement audit also found that 78 of 83 segmentation failures retained the correct classifier type, and 16 of 17 zero-Dice cases retained it. Classifier confidence cannot be used as a proxy for segmentation reliability.

### 4. Calibration hand-off

The observed problem is miscalibration and over-confidence, not global under-confidence. calibration and uncertainty work should calibrate the classifier and segmenter separately. The classifier needs class-probability calibration and an abstention/disagreement rule because its no-tumor false negative is confidently wrong. The segmenter needs pixel-level reliability analysis inside and outside reference regions, calibration curves, and case-level uncertainty or abstention signals that expose misplaced foreground. Calibration can make confidence more honest; it cannot repair incorrect localization or replace retraining and external validation.

## Manual review result

The user reviewed the single classifier-no-tumor/segmenter-non-empty case and four deterministically sampled zero-Dice gliomas. The meningioma reference covers a visible lesion; the scattered prediction is therefore treated as a segmentation localization failure rather than an incomplete reference. The four sampled gliomas were anatomically heterogeneous. A separate all-17 position check did not confirm a consistent skull-base bias.

- Panel: `E:\BrainTumorProject\runs\segmentation\20260923-072439-397263-full\uncertainty_manual_review.png`
- Selection CSV: `E:\BrainTumorProject\runs\segmentation\20260923-072439-397263-full\uncertainty_manual_review_selections.csv`

## Scope

These findings describe selected extremes and correlations on BRISC's internal validation partition. They do not establish clinical validity, patient-independent generalization, or sealed-test performance.
