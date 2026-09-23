# Classifier confidence calibration

Scalar temperature scaling was evaluated with five-fold, similarity-aware cross-fitting on 893 BRISC internal-validation images. No model weights were trained, no test image or test prediction was accessed, and the operational classifier decisions were not changed.

## Fitted temperatures

- Final temperature fitted on all 893 validation cases: `1.34551903`
- Out-of-fold temperatures: `1.39860998`, `1.29599208`, `1.37496586`, `1.31474824`, `1.33682145`
- Similarity groups: `784` total, `79` with multiple validation images, largest group `6` images

## Calibration metrics

| Metric | Raw | Cross-fitted calibrated | Change |
|---|---:|---:|---:|
| Multiclass NLL | 0.084694 | 0.078479 | -0.006215 |
| Multiclass Brier | 0.036498 | 0.035318 | -0.001181 |
| Top-label ECE | 0.008392 | 0.005648 | -0.002743 |
| Top-label MCE | 0.054167 | 0.013059 | -0.041108 |
| Accuracy | 0.977604 | 0.977604 | +0.000000 |

### Classwise ECE

| Class | Raw | Cross-fitted calibrated |
|---|---:|---:|
| glioma | 0.004627 | 0.005841 |
| meningioma | 0.007414 | 0.009307 |
| pituitary | 0.009888 | 0.010917 |
| no_tumor | 0.002088 | 0.002673 |

### Tumor versus no-tumor probability calibration

| Metric | Raw | Cross-fitted calibrated | Change |
|---|---:|---:|---:|
| Binary NLL | 0.019244 | 0.017941 | -0.001303 |
| Binary Brier | 0.002246 | 0.002306 | +0.000060 |
| Binary ECE | 0.002068 | 0.002585 | +0.000517 |
| Binary MCE | 0.011096 | 0.010851 | -0.000245 |

## Gate result

**FAIL**

- `multiclass_nll_improved`: **PASS**
- `multiclass_brier_improved`: **PASS**
- `top_label_ece_at_most_0_05`: **PASS**
- `every_classwise_ece_at_most_0_08`: **PASS**
- `brier_delta_bootstrap_ci95_upper_below_zero`: **FAIL**
- `four_class_predictions_identical`: **PASS**

The primary fitting objective is multiclass NLL, but acceptance is conjunctive: an ECE above 0.05 is a failure even when NLL improves.

The aggregate point estimates improved, but the paired group-bootstrap 95% interval for calibrated-minus-raw Brier score crosses zero. The conservative gate requires the upper bound to be below zero, so this fitted temperature is not accepted for application use.

- Brier change 95% interval: `[-0.002919, 0.000404]`
- Reliability diagram: `E:\BrainTumorProject\runs\classifier\20260921-170607-243133-baseline\classifier_reliability_diagram.png`

## Limits

These are post-hoc internal-validation results. The checkpoint was previously selected with this validation partition, known groups are similarity groups rather than patient identifiers, and patient independence is unverified. This does not establish clinical or cross-dataset calibration.
