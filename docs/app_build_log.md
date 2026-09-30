# Gyrus Application Build Log

This file is append-only. Each application stage records the implementation commit, files, verification evidence, constraints, and known gaps before the next stage begins. Corrections must be added as dated follow-up entries rather than rewriting an accepted stage.

## Phase 6, Stage 1 — Domain layer

- **Date:** 2026-09-30 (Asia/Colombo)
- **Status:** Complete; awaiting user review
- **Implementation commit:** `997ede92282fd9d0522957a14e87043f7c4c0ec6` — `Build typed Gyrus domain layer`
- **Scope:** Typed Python domain modules only. No Qt, PyTorch, network-client, model loading, training, refitting, or sealed-test access.
- **Frozen-stack manifest SHA256 before and after implementation:** `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`

### Stage summary

Stage 1 establishes Gyrus's framework-independent contract. It verifies the frozen package before use; validates and preprocesses user images; enforces OOD-first disclosure, no-tumor, and segmentation-reliability policies; validates immutable inference results; records non-destructive mask edits; indexes opt-in local history; and builds the data model for the fixed three-page PDF. Actual neural inference, inter-process transport, Qt screens, interactive editing, history bundle creation, and PDF rendering remain assigned to later stages.

### Files

#### `gyrus/__init__.py`

- **Purpose:** Defines the application package and version `0.1.0`.
- **Why:** Gives later worker, UI, history, and PDF components one canonical application identity.
- **Content:** Package docstring and immutable `__version__` export.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_boundaries.py`
- **Known edge cases:** Version synchronization with the future Windows executable metadata is deferred to packaging.

#### `gyrus/domain/__init__.py`

- **Purpose:** Exposes the supported public domain API from one import location.
- **Why:** Prevents later Qt and worker layers from depending on private implementation details.
- **Content:** Explicit exports for constants, exceptions, manifest verification, preprocessing, policy decisions, schemas, orchestration, edit provenance, history, and report models.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_boundaries.py`
- **Known edge cases:** New domain types must be deliberately added to the export list; wildcard discovery is intentionally avoided.

#### `gyrus/domain/constants.py`

- **Purpose:** Holds registered thresholds, states, reasons, categories, and the required disclaimer.
- **Why:** Centralizes policy values and preserves full-precision decimal strings instead of scattering rounded literals.
- **Content:** `Decimal` thresholds for segmentation `0.50`, reliability `0.9490250173235835`, OOD review `0.36854275476100085`; typed enums; application metadata; disclaimer.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_policy.py tests/test_gyrus_domain_boundaries.py`
- **Known edge cases:** Float conversion is allowed only at numerical boundaries; comparisons use validated finite values and registered decimal provenance.

#### `gyrus/domain/errors.py`

- **Purpose:** Defines stable fail-closed exception types and machine-readable error codes.
- **Why:** Lets the worker and UI distinguish integrity, input, schema, policy, history, and report failures without parsing prose.
- **Content:** A base `GyrusDomainError` plus focused subclasses carrying stable codes and safe user-facing messages.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_manifest.py tests/test_gyrus_domain_input_contract.py tests/test_gyrus_domain_persistence_models.py`
- **Known edge cases:** Stage 8 will map worker crashes and IPC timeouts into additional application-state errors.

#### `gyrus/domain/manifest.py`

- **Purpose:** Verifies `runs/final_stack/` as a strict, read-only, allowlisted package.
- **Why:** Gyrus must fail before inference if the manifest, artifact hashes, sizes, thresholds, paths, or package membership drift.
- **Content:** Frozen manifest hash verification, exact threshold-string checks, SHA256 and size verification for every artifact, traversal/symlink/duplicate rejection, and unexpected-file rejection.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_manifest.py`
- **Known edge cases:** Verification intentionally rejects harmless extra files and symlinks; any packaging metadata needed later must first be explicitly registered and allowlisted.

#### `gyrus/domain/input_contract.py`

- **Purpose:** Validates PNG/JPEG uploads and reproduces the frozen classifier and segmenter preprocessing contracts.
- **Why:** Input ambiguity or preprocessing drift would invalidate every downstream score.
- **Content:** File-size, format, extension, dimensions, pixel-count, animation, alpha, EXIF-orientation, and decompression checks; grayscale conversion; 128×128 aspect-preserving black-canvas segmentation input; 224×224 classifier tensor normalization; immutable arrays.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_input_contract.py`
- **Known edge cases:** DICOM and NIfTI are not accepted in Phase 6; animated files, transparency, orientation tags, malformed images, and decompression bombs fail closed.

#### `gyrus/domain/policy.py`

- **Purpose:** Converts OOD and reliability scores into the registered application decisions.
- **Why:** Policy behavior must be deterministic, full precision, independent of presentation code, and fail closed for missing or invalid scores.
- **Content:** O06 OOD review decision, P03 reliability boundary, Amendment 22 no-tumor path, joint status, reason codes, and score validation.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_policy.py`
- **Known edge cases:** OOD `REFUSE` remains disabled; exact equality passes the relevant threshold; no-tumor predictions always use `SEGMENTATION_RELIABILITY_NOT_APPLICABLE` with a null reliability score.

#### `gyrus/domain/result_schema.py`

- **Purpose:** Defines immutable, validated inference-result structures and their serialized schema.
- **Why:** Worker output must not enter the UI or persistence layer unless it is internally consistent and provenance-complete.
- **Content:** Class probabilities, categories, content-addressed array references, mask statistics, timestamps, devices, disclosure state, tumor/no-tumor invariants, serialization, and top-level payload validation.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_orchestration.py tests/test_gyrus_domain_persistence_models.py`
- **Known edge cases:** Constructors deeply validate results produced in-process; the lightweight serialized top-level validator will be paired with strict IPC parsing in Stage 2.

#### `gyrus/domain/orchestration.py`

- **Purpose:** Defines the frozen-inference sequence independently of any neural runtime.
- **Why:** O05 must finish before classification disclosure, while silent model forward passes may run without leaking results.
- **Content:** A typed backend `Protocol`, ordered events, silent classification/segmentation execution, OOD validation and policy, post-segmentation reliability policy, fail-closed invalid-OOD behavior, and no-tumor spatial-result suppression.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_orchestration.py`
- **Known edge cases:** GPU/CPU execution, cancellation, process death, IPC, and shared-memory cleanup are Stage 2 responsibilities.

#### `gyrus/domain/mask_provenance.py`

- **Purpose:** Records non-destructive mask edits in an immutable hash chain.
- **Why:** Edited masks must never overwrite the model output or retroactively change the reliability score computed for the original prediction.
- **Content:** Edit operations, before/after mask hashes, timestamps, sequence validation, chain validation, and provenance serialization.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_persistence_models.py`
- **Known edge cases:** Brush geometry, raster application, undo/redo commands, and editor autosave are deferred to Stage 5; this module validates their provenance records.

#### `gyrus/domain/history.py`

- **Purpose:** Provides an opt-in, local-only SQLite index for completed case bundles.
- **Why:** History must remain disabled by default, transactional, contained under the configured local root, and free of patient-identity fields.
- **Content:** Schema initialization, explicit enablement, contained bundle paths, duplicate-safe insert, listing, lookup, deletion, and transaction rollback.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_persistence_models.py`
- **Known edge cases:** The repository indexes already-created bundles; atomic bundle writing, retention controls, and UI confirmation are deferred to Stage 6.

#### `gyrus/domain/report_model.py`

- **Purpose:** Defines the immutable data contract for Gyrus's fixed three-page PDF summary.
- **Why:** The later renderer needs a validated source that preserves thresholds, creator metadata, disclaimer text, and edited-mask provenance on every page.
- **Content:** Three page models, creator `Gyrus 0.1.0`, default filename generation, required disclaimer, exact thresholds, source references, model-versus-edited mask declarations, and consistency checks.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_persistence_models.py`
- **Known edge cases:** Layout, fonts, pagination, image rendering, and byte-level PDF validation are Stage 7 responsibilities.

#### `tests/test_gyrus_domain_manifest.py`

- **Purpose:** Exercises strict frozen-stack integrity checks against a controlled package and the real frozen manifest.
- **Why:** Manifest drift must stop the application before model loading.
- **Content:** Happy-path verification plus manifest hash, artifact hash, unexpected file, traversal, symlink, and threshold-drift failures.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_manifest.py`
- **Known edge cases:** Windows permission failures and files changing during verification will receive process-level handling in Stage 2.

#### `tests/test_gyrus_domain_input_contract.py`

- **Purpose:** Verifies supported uploads and exact frozen preprocessing parity.
- **Why:** A visually plausible but different tensor would silently invalidate the frozen evidence.
- **Content:** PNG/JPEG validation, rejection cases, immutable outputs, tensor shapes/ranges, and equality with the registered final-stack runtime preprocessing.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_input_contract.py`
- **Known edge cases:** Codec availability can differ between packaged Pillow builds and will be covered by packaged-application tests.

#### `tests/test_gyrus_domain_policy.py`

- **Purpose:** Tests every registered policy boundary and fail-closed branch.
- **Why:** Rounded thresholds and inclusive/exclusive boundary mistakes could change the application state.
- **Content:** Exact OOD/reliability equality, values around boundaries, invalid/unavailable scores, no-tumor decisions, and joint policy outcomes.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_policy.py`
- **Known edge cases:** The tests intentionally use exact decimal strings; display rounding must never feed back into these decisions.

#### `tests/test_gyrus_domain_orchestration.py`

- **Purpose:** Verifies event order, disclosure timing, tumor and no-tumor paths, and fail-closed orchestration.
- **Why:** The application may execute silent forward passes early, but it must not disclose them before the OOD gate.
- **Content:** A deterministic fake backend, event-sequence assertions, disclosure assertions, OOD failure behavior, and Amendment 22 no-tumor behavior.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_orchestration.py`
- **Known edge cases:** True process concurrency and cancellation races are intentionally excluded until the Stage 2 worker exists.

#### `tests/test_gyrus_domain_persistence_models.py`

- **Purpose:** Tests edit provenance, local history, and PDF report contracts.
- **Why:** These three areas preserve auditability after inference and require strict invariants before UI integration.
- **Content:** Hash-chain validation, tamper rejection, opt-in history transactions/path containment, report page/disclaimer/threshold checks, and edit provenance consistency.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_persistence_models.py`
- **Known edge cases:** Filesystem-full, antivirus locks, and concurrent UI sessions will be exercised after concrete bundle and PDF writers exist.

#### `tests/test_gyrus_domain_boundaries.py`

- **Purpose:** Enforces architectural and governance boundaries over the domain package.
- **Why:** Stage 1 must stay reusable and testable without Qt, PyTorch, or network access.
- **Content:** AST checks banning Qt, PyTorch, socket, requests, httpx, and urllib imports; exact registered threshold-string checks.
- **Test:** `python -m pytest -q tests/test_gyrus_domain_boundaries.py`
- **Known edge cases:** Dynamic imports are not used; future dependency additions require an explicit boundary review.

#### `docs/app_build_log.md`

- **Purpose:** Records the evidence, decisions, files, gaps, and commits for each application stage.
- **Why:** Amendment 22 requires an append-only record that can be reviewed before the next stage begins.
- **Content:** This Stage 1 entry and future dated stage entries.
- **Test:** Read this file after generation, run `git diff --check`, and verify each stage has a summary, per-file records, test evidence, gaps, and a descriptive implementation commit.
- **Known edge cases:** Corrections are appended as new entries; accepted historical entries are not silently rewritten.

### Verification evidence

- **Focused Stage 1 tests:** `20 passed in 10.65s` before the final formatting-only cleanup.
- **Full repository suite after cleanup:** `232 passed in 18.39s`.
- **Ruff after cleanup:** `All checks passed!`
- **Git whitespace check:** clean after removing one redundant EOF blank line from each implementation file.
- **Architecture boundary:** no Qt, PyTorch, socket, requests, httpx, or urllib imports under `gyrus/domain/`.
- **Frozen package:** real strict verification passed with 31 allowlisted artifacts; manifest SHA256 unchanged.
- **Governance constraints:** zero BRISC sealed-test access; zero training or refitting; zero threshold changes; zero changes under `runs/final_stack/`.

### Known gaps carried to later stages

1. Stage 2 must implement the isolated worker process, read-only frozen-stack loading, CPU/GPU execution, IPC schema parsing, cancellation, timeouts, and shared-memory cleanup.
2. Stage 3 must provide the Qt shell and state machine without moving policy logic into widgets.
3. Stage 5 must implement mask raster editing and undo/redo while emitting the provenance records defined here.
4. Stage 6 must implement opt-in case-bundle creation, retention, and history UI around the repository defined here.
5. Stage 7 must render and validate the three-page PDF from the report model.
6. Stage 8 must map worker/process/filesystem failures into complete user-visible state transitions.
7. Stage 9 must add packaged-runtime, golden-visual, PDF, privacy, failure, parser-security, and accessibility coverage.

### What to test next

Before Stage 2 begins, review the exact threshold comparisons, OOD-first disclosure events, no-tumor fail-closed result, accepted upload formats, strict frozen-package allowlist, and the decision to keep reliability tied to the original model mask after editing. Stage 2 should begin only after those contracts are accepted.
