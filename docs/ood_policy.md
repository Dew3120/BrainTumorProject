# OOD operating policy

The application must run the O05 OOD detector before tumor classification or segmentation.

## Decision rule

- **ACCEPT:** score below `0.36854275476100085`. Continue to the frozen tumor stack and then apply the separate reliability policy.
- **REVIEW:** score at or above `0.36854275476100085` and below the REFUSE boundary. Pause and show the registered warning.
- **REFUSE:** not enabled; the 99th-percentile candidate remains descriptive.
- Missing, non-finite, or out-of-range scores fail closed to **REVIEW**.

The segmentation threshold remains `0.50`. The existing reliability-policy threshold remains exactly `0.9490250173235835`.

## User-facing warning

> This input does not match the model's training distribution. Predictions may be unreliable.
>
> **Research prototype — not for clinical use**

## Coverage on the held-out benchmark

- BRISC validation flagged: **60/680 (8.82%)**.
- OOD evaluation images flagged: **699/700 (99.86%)**.

These rates describe the registered retrospective benchmark. They do not establish clinical safety or coverage of every possible OOD input.
