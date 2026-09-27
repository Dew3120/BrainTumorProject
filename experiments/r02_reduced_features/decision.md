# Reduced calibrated-E08 reliability model decision

**Decision:** PASS

The fourteen-feature baseline reproduced at AUROC `0.954502814259`, with absolute difference `0` from committed R01 and maximum per-case OOF score difference `0`.

The registered ten-feature model achieved AUROC `0.955502080104`. The fixed minimum was `0.949502814259`; therefore R02 **passed** its noninferiority gate. Reduced-minus-full AUROC was `0.000999265846`, with paired 95% similarity-group bootstrap interval `[0.000066687465, 0.001998220021]`.

All ten features derive only from the C03 out-of-fold calibrated probability map and threshold-`0.50` prediction. No reference-mask value, classifier probability, raw image, path, identity, or metadata entered the reliability estimator. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
