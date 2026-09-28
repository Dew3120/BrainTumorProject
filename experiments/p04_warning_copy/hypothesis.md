# P04 — User-facing reliability policy copy

## Purpose

P04 defines the complete user-facing messages for every supported P03 policy outcome and predicted-glioma variant.

## Catalogue contract

The catalogue contains exactly eight complete messages: general and predicted-glioma variants for `STANDARD / AT_OR_ABOVE_THRESHOLD`, `REVIEW_REQUIRED / BELOW_THRESHOLD`, `REVIEW_REQUIRED / RELIABILITY_UNAVAILABLE`, and `REVIEW_REQUIRED / RELIABILITY_INVALID`.

Every message must contain the exact policy threshold `0.9490250173235835` and the exact disclaimer `Research prototype — not for clinical use`. Messages must use plain language, make no clinical or accuracy claim, and never emit a Warning policy state.

## Completion rule

P04 completes only if the P03 hashes and policy contract match, all eight catalogue combinations exist exactly once, the copy checks pass before final artifact generation, and no test access, segmentation-threshold change, model fitting, gate relaxation, or policy change occurs.
