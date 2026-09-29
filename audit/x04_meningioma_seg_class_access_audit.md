# X04 Meningioma-SEG-CLASS access audit

**Audit outcome:** `ACCESS_BLOCKED`

The registered TCIA Meningioma-SEG-CLASS version 1 cohort requires a verified human-account access grant. No grant was confirmed and no local payload was present after X01-X03 completed. Amendment 13 therefore requires this terminal access outcome.

## Verification

- Dataset files opened: 0
- Dataset images opened: 0
- Performance metrics: `NOT_ESTIMABLE`
- BRISC sealed-test images opened: 0
- Neural retraining, calibration refitting, reliability refitting: 0
- Segmentation threshold: `0.50`, unchanged
- Reliability policy threshold: `0.9490250173235835`, unchanged
- Frozen R02/C03 artifact hashes: verified
- Free disk at decision: 348,132,126,720 bytes

## Consequence

X05 may proceed and must list X04 as access-blocked. It must not treat missing X04 metrics as a failure, pass, zero, or independent meningioma result. Human-controlled access can support a future separately registered evaluation, but it cannot be backfilled into this completed Stage 6 run.
