# O07 — OOD score and segmentation failure association

## Registered question H12

Within X02 and X03 separately, quantify whether higher frozen O05 OOD score is associated with lower calibrated segmentation Dice and enrichment of Dice-below-0.50 failures.

## Decision rule

Descriptive only. Report Spearman correlation, failure-detection AUROC, failure average precision, high-OOD-quintile enrichment, capture, and patient-bootstrap intervals. No numeric result passes or fails H12, and no result may tune O05 or O06.

## Interpretation

Association would show that O05 ranks some failures within the evaluated external sources. It would not establish causal protection, unseen-site generalization, external calibration, or clinical safety. Lack of association would narrow O05 to source detection on its registered benchmark.
