# P03 — Reliability decision-rule formalization

## Purpose

P03 formalizes the P02 internal-development optimum as a deterministic two-state policy contract for later application integration.

## Rule

For a finite reliability score `r` in `[0,1]`:

- `r < 0.9490250173235835` → `REVIEW_REQUIRED`
- `r >= 0.9490250173235835` → `STANDARD`

Missing, nonnumeric, non-finite, or out-of-range reliability values fail closed to `REVIEW_REQUIRED` with an explicit reason code.

## No Warning state

P02 found `theta_low == theta_high == 0.9490250173235835` after exhaustively evaluating 230,860 registered threshold pairs. The Warning interval is empty and P03 will not create an unregistered band.

## Completion rule

P03 is descriptive policy formalization. It completes only if all upstream hashes, the exact threshold and costs, the 680-case policy outcomes, boundary behavior, invalid-input fallbacks, and feature-independent state logic verify without test access, segmentation-threshold change, or model fitting.
