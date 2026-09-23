# Classifier/segmenter agreement audit

This verification joins the accepted classifier and accepted segmenter on the same 680 held-out BRISC validation images that have tumor masks. No test image was accessed. Four-class classifier correctness is evaluated against the tumor type. Segmenter success means hard-mask Dice at or above 0.50.

## Verification

- Shared split SHA256: `d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28`
- Classifier checkpoint SHA256: `3bc044027b83b2de225ebde3fb817be8827d2b427c599824697b13ff7a727535`
- Segmenter checkpoint SHA256: `16DC8745993EEF959D0F2924C9147A45320673F6FC0B465AB08F8AD7B2F723DE`
- Classifier validation images recomputed: 893
- Classifier maximum metric difference: 0.0
- Classifier maximum stored-probability difference: 0.0
- Segmenter validation images recomputed: 680
- Segmenter maximum metric difference: 0.0
- Segmenter maximum per-image metric difference: 0.0
- Test images accessed: 0

## Agreement table

| Classifier result | Segmenter Dice >= 0.50 | Segmenter Dice < 0.50 | Total |
|---|---:|---:|---:|
| Correct tumor type | 583 (85.7%) | 78 (11.5%) | 661 |
| Incorrect tumor type / no tumor | 14 (2.1%) | 5 (0.7%) | 19 |
| Total | 597 | 83 | 680 |

Percentages in the table are shares of all 680 joined cases.

## Interpretation

- The classifier predicted the correct tumor type in 661 of 680 cases (97.2%).
- The segmenter reached Dice >= 0.50 in 597 of 680 cases (87.8%).
- The classifier remained correct in 78 of 83 segmentation failures (94.0%). Correct type recognition therefore does not guarantee correct localization.
- The segmenter succeeded in 14 of 19 classifier errors (73.7%).
- Both components failed their respective criteria in 5 cases.
- Of the 17 zero-Dice cases, 16 still had the correct classifier type.

The binary special-case flags below use the classifier's unchanged tumor-score threshold of 0.50. The four-class prediction is retained separately in every row of the CSV.

## Special cases

### Classifier says no tumor; segmenter predicts a non-empty region

Count: **1**

| Path | True type | Classifier class | Binary decision | Tumor score | Segmenter Dice | Predicted pixels |
|---|---|---|---|---:|---:|---:|
| data/raw/brisc/brisc2025/classification_task/train/meningioma/brisc2025_train_01380_me_ax_t1.jpg | meningioma | no_tumor | no_tumor | 0.000015 | 0.000000 | 179 |

### Classifier says tumor; segmenter predicts an empty region

Count: **0**

No cases.

### Classifier has the correct tumor type; segmenter Dice is zero

Count: **16**

| Path | True type | Classifier class | Binary decision | Tumor score | Segmenter Dice | Predicted pixels |
|---|---|---|---|---:|---:|---:|
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00050_gl_ax_t1.jpg | glioma | glioma | tumor | 0.999574 | 0.000000 | 171 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00279_gl_ax_t1.jpg | glioma | glioma | tumor | 0.993552 | 0.000000 | 106 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00417_gl_co_t1.jpg | glioma | glioma | tumor | 0.999960 | 0.000000 | 84 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00559_gl_co_t1.jpg | glioma | glioma | tumor | 0.999932 | 0.000000 | 49 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00607_gl_co_t1.jpg | glioma | glioma | tumor | 0.999840 | 0.000000 | 87 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00701_gl_co_t1.jpg | glioma | glioma | tumor | 0.998312 | 0.000000 | 261 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00744_gl_co_t1.jpg | glioma | glioma | tumor | 0.999776 | 0.000000 | 31 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00789_gl_co_t1.jpg | glioma | glioma | tumor | 0.999276 | 0.000000 | 241 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00871_gl_sa_t1.jpg | glioma | glioma | tumor | 0.999657 | 0.000000 | 112 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00874_gl_sa_t1.jpg | glioma | glioma | tumor | 0.998596 | 0.000000 | 42 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00950_gl_sa_t1.jpg | glioma | glioma | tumor | 0.999925 | 0.000000 | 179 |
| data/raw/brisc/brisc2025/classification_task/train/glioma/brisc2025_train_00972_gl_sa_t1.jpg | glioma | glioma | tumor | 0.998710 | 0.000000 | 376 |
| data/raw/brisc/brisc2025/classification_task/train/meningioma/brisc2025_train_01959_me_co_t1.jpg | meningioma | meningioma | tumor | 0.998079 | 0.000000 | 24 |
| data/raw/brisc/brisc2025/classification_task/train/pituitary/brisc2025_train_03839_pi_ax_t1.jpg | pituitary | pituitary | tumor | 0.999058 | 0.000000 | 151 |
| data/raw/brisc/brisc2025/classification_task/train/pituitary/brisc2025_train_03894_pi_ax_t1.jpg | pituitary | pituitary | tumor | 0.996364 | 0.000000 | 150 |
| data/raw/brisc/brisc2025/classification_task/train/pituitary/brisc2025_train_04439_pi_co_t1.jpg | pituitary | pituitary | tumor | 0.999990 | 0.000000 | 183 |

## Scope

These are paired behaviors on BRISC's internal validation partition. They do not measure clinical validity, patient-independent generalization, or performance on the sealed test set.
