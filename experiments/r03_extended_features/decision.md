# Extended calibrated-E08 reliability model decision

**Decision:** FAIL

The fourteen-feature baseline reproduced at AUROC `0.954502814259`, with absolute aggregate difference `0` and maximum per-case OOF score difference `0` from committed R01.

The registered twenty-feature model achieved AUROC `0.951423444000`. The fixed minimum was `0.959502814259`; therefore R03 **failed** its superiority gate. Extended-minus-baseline AUROC was `-0.003079370259`, with paired 95% similarity-group bootstrap interval `[-0.014050581511, 0.005847072724]`.

All 680 source images matched their frozen SHA256 values. Five new features used only the calibrated prediction; the boundary-gradient feature used only the uploaded image after preprocessing that matched the segmenter pixel for pixel plus the predicted boundary. No reference-mask value, classifier probability, true type, identity, or metadata entered the reliability estimator. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
