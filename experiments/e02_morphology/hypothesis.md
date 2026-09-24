# Morphological closing and opening

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H1, local experiment E02

Binary closing followed by binary opening will fill small holes and remove small foreground noise while retaining the primary tumor region. The experiment tests square 3×3 and 5×5 kernels independently on frozen v1 hard masks.

A variant passes only if glioma image-mean Dice improves by at least `0.02` and neither meningioma nor pituitary worsens by more than `0.01`. Among passing variants, higher glioma Dice is selected; an exact tie selects 3×3. No result can change the fixed gate.
