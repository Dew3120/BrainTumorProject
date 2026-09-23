# Experiment tracking contract

Every registered experiment has one directory listed in `registry.json` and contains:

- `hypothesis.md` — falsifiable claim, population, and fixed decision gate.
- `config.json` — immutable parameters, seeds, dataset hashes, and gate values.
- `result.json` — machine-readable results; `NOT_RUN` before execution.
- `decision.md` — `PENDING` before execution, then accepted, rejected, or invalid with rationale.
- `artifacts/` — figures and tables produced only by that experiment.

Create a new directory from `_template` before running a new experiment. Register it and log the decision in `audit/review_log.md` before inspecting results. Never reuse a directory for a changed hypothesis or configuration. Run `uv run python scripts/validate_experiment_tracking.py` to verify the contract.
