# Unified ECE comparison

**Status:** REGISTERED — NOT STARTED

**Experiment:** Calibration study C08

C08 is a descriptive standardization and integrity study, not a fitted calibrator. It independently recomputes C01–C07 ECE from each stored 15-bin equal-frequency summary, verifies exact bin count and pixel coverage, and requires agreement with the stored ECE within `1e-12`. It also reports the frozen classifier's C01 four-class and binary 15-bin ECE and recomputes the accepted v1 case-reliability ECE using 15 equal-frequency case bins.

The result is verified only if all seven source-result hashes match, every compared pixel summary has exactly 15 non-empty bins whose counts equal the stored pixel count, every recomputed ECE matches its stored value within tolerance, and the reliability source contains exactly 680 unique finite scores. C08 does not revise C01–C07 gates or decisions, refit any method, change a threshold, retrain a model, or access the sealed test.
