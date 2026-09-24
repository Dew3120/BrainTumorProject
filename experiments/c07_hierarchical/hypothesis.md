# Nested hierarchical pixel calibration

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** Calibration experiment C07

Nested selection among seven fixed combinations of reference-region splitting (`R`), frozen reliability weighting (`W`), and classifier-predicted category routing (`C`) will achieve lesion-centred aggregate ECE at most `0.03` and lesion-centred ECE at most `0.05` within each predicted glioma, meningioma, and pituitary route. These four checks are conjunctive.

The candidates are `R`, `W`, `C`, `RW`, `RC`, `WC`, and `RWC`. For each outer held-out fold, the other four original similarity-aware folds form the inner cross-validation population. Each candidate produces inner out-of-fold predictions on those four folds. Selection is lexicographic by lowest inner lesion-centred ECE, then NLL, then Brier, then candidate ID. The chosen candidate is refitted on the full outer-training population and evaluated once on the untouched outer fold. The same nested procedure is repeated for all five outer folds.

`C` and `W` use prediction-derived inputs. Any candidate containing `R` requires the reference-defined lesion region to route pixels and is oracle-only. C07 is operationally eligible only if every outer-fold selection and the final all-validation proposal omit `R`. The classifier router never uses true type. The no-tumor route uses the corresponding global fallback because only one validation case follows that route.

The final all-validation candidate is selected using five-fold inner OOF metrics across all 680 cases, then fitted on all cases as an operational proposal only. It does not supply reported metrics. Hard-mask changes, all regions, NLL, Brier, MCE, and true-label breakdowns are descriptive. The threshold remains `0.50`; no neural model is retrained and no sealed-test image is accessed.
