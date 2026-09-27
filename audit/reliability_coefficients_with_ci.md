# Reliability coefficients with bootstrap confidence intervals

**Status:** DESCRIPTIVE_COMPLETE

R07 refit the accepted R02 standardized logistic reliability model on 2,000 similarity-group bootstrap resamples. This is a development-set stability analysis, not external or clinical validation. Coefficients are conditional associations among correlated prediction-derived features and are not causal effects.

## Coefficient stability

| Rank | Feature | Full fit | Bootstrap mean | SD | 95% CI | Fold sign | Bootstrap sign |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `boundary_entropy_mean` | -2.0749 | -2.1424 | 0.3850 | [-2.8973, -1.3756] | 5/5 | 2000/2,000 |
| 2 | `predicted_area_fraction` | 0.9810 | 1.0311 | 0.3901 | [0.2389, 1.7612] | 5/5 | 1988/2,000 |
| 3 | `whole_probability_mean` | -0.6354 | -0.5991 | 0.4137 | [-1.3990, 0.2184] | 5/5 | 1852/2,000 |
| 4 | `empty_prediction` | -0.6320 | -0.5788 | 0.3028 | [-1.0996, 0.0000] | 5/5 | 1721/2,000 |
| 5 | `whole_probability_sd` | 0.5944 | 0.5676 | 0.2994 | [-0.0223, 1.1762] | 5/5 | 1935/2,000 |
| 6 | `uncertain_pixel_fraction` | 0.5079 | 0.5201 | 0.2288 | [0.0708, 0.9675] | 5/5 | 1978/2,000 |
| 7 | `high_confidence_foreground_fraction` | 0.4715 | 0.4598 | 0.3416 | [-0.2008, 1.1555] | 5/5 | 1824/2,000 |
| 8 | `normalized_perimeter` | -0.3887 | -0.3930 | 0.1702 | [-0.7249, -0.0638] | 5/5 | 1975/2,000 |
| 9 | `predicted_region_entropy_mean` | 0.2796 | 0.2602 | 0.3122 | [-0.3431, 0.8674] | 5/5 | 1604/2,000 |
| 10 | `bounding_box_fill_ratio` | 0.1596 | 0.1612 | 0.1633 | [-0.1445, 0.4864] | 5/5 | 1686/2,000 |

The ordering above is fixed by absolute bootstrap-mean magnitude. An interval excluding zero is evidence of coefficient-direction stability in this internal development population; it does not prove causal importance.

## R01 fourteen-feature comparison

| Feature | R01 mean | R02 mean | Signed change | Magnitude change | Mean sign changed |
|---|---:|---:|---:|---:|---|
| `empty_prediction` | -0.6262 | -0.6157 | +0.0104 | -0.0104 | no |
| `predicted_area_fraction` | 0.9664 | 0.9426 | -0.0238 | -0.0238 | no |
| `whole_probability_mean` | -0.6042 | -0.5916 | +0.0126 | -0.0126 | no |
| `whole_probability_sd` | 0.5496 | 0.5488 | -0.0008 | -0.0008 | no |
| `predicted_region_entropy_mean` | 0.1517 | 0.1764 | +0.0247 | +0.0247 | no |
| `boundary_entropy_mean` | -1.9862 | -1.9973 | -0.0111 | +0.0111 | no |
| `high_confidence_foreground_fraction` | 0.4546 | 0.4503 | -0.0044 | -0.0044 | no |
| `uncertain_pixel_fraction` | 0.5087 | 0.5130 | +0.0044 | +0.0044 | no |
| `bounding_box_fill_ratio` | 0.1965 | 0.1665 | -0.0300 | -0.0300 | no |
| `normalized_perimeter` | -0.3946 | -0.3881 | +0.0065 | -0.0065 | no |

## Integrity

- Cases: 680
- Similarity groups: 583
- Valid bootstrap refits: 2000
- Maximum R02 fold-coefficient reproduction difference: `0`
- C03 changed pixels/images reproduced: 22891/679
- All ten features are available at inference time and derive only from the calibrated prediction.
- Reference masks were used only to define the reliability target and evaluate it.
- Test images accessed: 0; threshold changes: 0; classifier/neural-segmenter retraining: 0.
