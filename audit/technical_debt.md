# Technical debt

## Phase 8 cleanup: repository-wide Ruff findings

Logged on 2026-09-23 while closing Phase 4 Section C. Running `uv run ruff check .` reports 10 findings in older Phase 1/environment scripts. These findings are outside the verification-only scope of Section C and remain intentionally unfixed until Phase 8.

| File | Ruff findings | Phase 8 action |
|---|---|---|
| `scripts/audit_brisc.py` | `I001`, `DTZ005`, two `BLE001` | Sort imports, use a timezone-aware timestamp, and replace broad exception handlers with the narrow exceptions the audit can safely record. |
| `scripts/review_brisc.py` | `I001`, `F401`, `DTZ005`, `F841` | Sort imports, remove unused `Counter` and `by_path`, and use a timezone-aware timestamp. |
| `scripts/verify_training_environment.py` | `I001`, `DTZ005` | Sort imports and use a timezone-aware timestamp. |

Phase 8 acceptance criteria:

1. `uv run ruff check .` exits successfully with no ignored findings added for these files.
2. The full test suite passes after cleanup.
3. No accepted model checkpoint, frozen split, or recorded Phase 3/4 result artifact is modified.
