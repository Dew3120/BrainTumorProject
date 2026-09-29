# OOD threshold selection

# Decision — O06 OOD policy

**Status:** ACCEPTED (descriptive)

The REVIEW threshold is `0.36854275476100085` from the registered BRISC-fitting 95th percentile. The REFUSE candidate is `0.86151672113516586` and is **not enabled** under the fixed utility rule. BRISC validation false-flag rate is `8.8235%`; held-out OOD detection rate is `99.8571%`.

Thresholds were computed only from BRISC detector-fitting scores; held-out labels did not choose the numeric thresholds. REFUSE enablement used the pre-result utility rule recorded in the implementation.
