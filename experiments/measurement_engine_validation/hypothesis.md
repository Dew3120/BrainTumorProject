# Measurement-engine validation

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** H6

The measurement engine will reproduce analytically known pixel area, bounding-box dimensions, maximum diameter, and spacing-based unit conversions on at least 50 deterministic synthetic masks within every G6 tolerance. Empty masks must return an explicit not-measurable state.

Passing validates geometry code only. It does not validate model boundaries, physical BRISC measurements, or clinical size estimates.
