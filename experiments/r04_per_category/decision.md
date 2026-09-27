# Per-category calibrated-E08 reliability decision

**Decision:** PASS

R04 required all six true-category and classifier-predicted-category models to achieve failure-detection AUROC at least `0.80`. The unchanged conjunctive gate **passed**.

Version A AUROCs were glioma `0.857368`, meningioma `0.845455`, and pituitary `0.925457`. Version B AUROCs were predicted-glioma `0.867716`, predicted-meningioma `0.902973`, and predicted-pituitary `0.957851`.

Glioma changed by `-0.019475` in Version A and `-0.003263` in Version B relative to the identical-case R02 aggregate model. Pituitary changed by `-0.031646` in Version A and `-0.006061` in Version B. These subgroup comparisons are internal descriptive findings.

Version B routed `679/680` cases to a tumor-specific reliability model. The single frozen-classifier `no_tumor` prediction was recorded and excluded without fallback. Four-class routing accuracy was `0.972059`; accuracy among the 679 tumor-routed cases was `0.973490`.

All ten model features derive only from the C03 out-of-fold calibrated probability map and threshold-`0.50` prediction. True category entered Version A grouping and routing evaluation only; it did not enter Version B. No reference-mask value or classifier probability entered a reliability model. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
