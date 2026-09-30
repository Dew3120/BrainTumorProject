# R11 — size-robust glioma metrics

## Registered hypothesis H10

On the fixed 204-case BRISC glioma validation subset, E08 has higher mean surface Dice at a two-model-pixel tolerance than v1, matching the positive direction already observed for ordinary Dice.

## Gate

`mean_surface_dice_E08 - mean_surface_dice_v1 > 0`.

Equality fails. HD95 and lesion-wise detection are secondary descriptive metrics. All definitions, empty-mask handling, component matching, bootstrap rules, and stopping conditions are fixed by Amendment 21.

## Interpretation

A pass shows only that E08's internal directional improvement persists under one complementary boundary metric. It does not remove the mathematical relationship between target size and overlap metrics, prove a causal small-target mechanism, establish patient independence, or establish external or clinical validity.
