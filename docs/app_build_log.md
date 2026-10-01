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

## Phase 6, Stage 2 — Isolated frozen worker

- **Date:** 2026-10-01 (Asia/Colombo)
- **Status:** Complete; awaiting user review
- **Implementation commit:** `35a8f4065948316fd1e29d966d5a51ae24d39bd6` — `Add verified spawn worker for frozen Gyrus inference`
- **Frozen-stack manifest SHA256 before and after:** `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`

### Stage summary

Stage 2 adds a single-job, `multiprocessing` spawn worker that verifies all 31 frozen artifacts before declaring readiness, loads the model stack once per worker generation, and uses only an explicitly selected CPU or CUDA device. The worker preprocesses the image, runs the classifier and v1 segmenter silently, computes O02–O05 and applies OOD policy, then assembles the cached E08 output with C03 and conditional R02 reliability. It sends only a complete validated result to the parent. The supervisor validates provenance and array payloads again, restarts after a crash or timeout, and never retries a failed job automatically.

The registered O04 segmenter-entropy score is computed from the raw E08 ensemble. E05, E06, and E07 therefore execute silently during O04, before O05 completes; their predictions stay private and cached until the OOD policy decision. This preserves the frozen O04/O05 score definition and Amendment 22 disclosure order.

### Files

#### `gyrus/worker/__init__.py`

- **Purpose:** Exposes the supported worker API.
- **Why:** Later Qt code needs stable imports without reaching into private process and inference internals.
- **Content:** Explicit exports for typed protocol messages, loader, inference supervisor, and errors.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_protocol.py`
- **Known edge cases:** Qt integration and user-visible error mapping are deferred to Stages 3, 4, and 8.

#### `gyrus/worker/protocol.py`

- **Purpose:** Defines typed request, readiness, success, failure, shutdown, event, diagnostics, and content-addressed array messages.
- **Why:** Parent and child must reject malformed or mismatched IPC data before it reaches the application.
- **Content:** Frozen dataclasses, protocol version, canonical UUID and timestamp checks, explicit device selection, array shape/byte-count/hash validation, and registered stage order.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_protocol.py`
- **Known edge cases:** Only local paths are accepted. A process boundary is trusted to carry pickle-safe Python dataclasses; no external client or socket protocol exists.

#### `gyrus/worker/loader.py`

- **Purpose:** Verifies and loads the frozen stack once per worker session on the requested device.
- **Why:** No job may start with an unverified artifact, changed threshold, missing model, or silent GPU-to-CPU fallback.
- **Content:** Strict 31-artifact verification, verified runtime import without bytecode writes, CPU/CUDA selection, frozen checkpoint loading, device checks, post-load re-verification, and session cleanup.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_protocol.py tests/test_gyrus_worker_inference.py tests/test_gyrus_worker_process.py`
- **Known edge cases:** CUDA inference was not exercised on this Stage 2 test run; an unavailable CUDA request fails explicitly. Packaged runtime driver compatibility remains a later integration check.

#### `gyrus/worker/inference.py`

- **Purpose:** Executes the registered frozen stack in one ordered, non-disclosing worker job.
- **Why:** OOD must complete before classification or spatial output can reach the UI, while the frozen O04 score still requires silent raw-E08 forwards.
- **Content:** Input decode and change detection, preprocessing, silent classifier/v1 passes, O02/O03/O04/O05, OOD decision, cached E08/C03/conditional R02, immutable result construction, mask statistics, array hashes, and event trace.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_inference.py`
- **Known edge cases:** For `no_tumor`, silent O04 maps are discarded and no spatial or reliability score is disclosed. DICOM/NIfTI and batching remain outside the Phase 6 input contract.

#### `gyrus/worker/process.py`

- **Purpose:** Runs inference in a spawned child and supervises one request at a time.
- **Why:** A model crash must not take down the UI, and the next job must use a newly verified worker generation.
- **Content:** Spawn context, one-slot request/response queues, exclusive job lock, startup readiness, early crash/timeout detection, explicit restart without automatic job retry, parent-side result and array provenance checks, and graceful shutdown.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_process.py`
- **Known edge cases:** UI cancellation and progress animation are Stage 3/4 integration work; this supervisor exposes a synchronous single-job API. Abrupt process death during a job returns a failure after detection and starts a new generation.

#### `tests/test_gyrus_worker_protocol.py`

- **Purpose:** Verifies typed IPC and architecture boundaries.
- **Why:** Device fallback, malformed payloads, or a server import would violate the worker design.
- **Content:** Explicit CPU/CUDA tests, unsupported device rejection, UUID and payload integrity, no Qt/network server imports, and spawn/single-slot assertions.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_protocol.py`
- **Known edge cases:** The CUDA-unavailable branch is tested by simulation; actual CUDA execution needs a compatible host.

#### `tests/test_gyrus_worker_inference.py`

- **Purpose:** Checks actual CPU inference against frozen verification outputs and tests the parent trust boundary.
- **Why:** The new execution order must preserve the registered scores, masks, route, and no-tumor policy.
- **Content:** `1e-12` fixture comparisons, exact raw/calibrated/mask hashes, OOD component checks, no-tumor suppression, one-load multiple-job behavior, and corrupted IPC provenance rejection.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_inference.py`
- **Known edge cases:** Synthetic frozen fixtures prove reconstruction, not clinical performance or external transport.

#### `tests/test_gyrus_worker_process.py`

- **Purpose:** Exercises a real Windows spawned worker and fail-closed startup.
- **Why:** In-process tests cannot prove crash isolation, restart, or pre-job hash enforcement.
- **Content:** Successful job, forced child termination, new generation/session, second successful job, and manifest-hash mismatch before readiness.
- **Test:** `python -m pytest -q tests/test_gyrus_worker_process.py`
- **Known edge cases:** Windows process termination uses the `multiprocessing.Process` handle; `os.kill(pid, 0)` was discarded because it interrupts the local test runner on Windows.

#### `docs/app_build_log.md`

- **Purpose:** Append the Stage 2 implementation and verification record.
- **Why:** Amendment 22 requires one readable, append-only entry per application stage.
- **Content:** This summary, per-file records, test evidence, known gaps, implementation commit, and next-stage review points.
- **Test:** Read this entry after writing, run `git diff --check`, and verify the implementation commit is descriptive and the working tree is clean.
- **Known edge cases:** Corrections to this accepted entry must be appended later rather than silently rewriting it.

### Verification evidence

- **Focused worker suite:** `11 passed` (protocol, frozen inference, spawned-process recovery).
- **Full repository suite:** `243 passed in 41.36s`.
- **Ruff:** `All checks passed!`.
- **Frozen reconstruction:** CPU scores match committed verification fixture values within `1e-12`; raw E08, C03 probability, and hard-mask hashes match exactly.
- **Crash recovery:** terminating generation 1 leads to generation 2 with a new process ID and load-session ID; a new job succeeds.
- **Fail-closed startup:** deliberately wrong manifest hash prevents `WorkerReady` and job acceptance.
- **Governance:** zero BRISC sealed-test access; zero neural retraining or refitting; zero threshold changes; no changes to `runs/final_stack/`; no TCP/HTTP server.

### Known gaps carried to later stages

1. Stage 3 must connect the synchronous supervisor to the Qt shell without running inference on the UI thread.
2. Stage 4 must display loading and OOD states while withholding model output until the final IPC response.
3. Stage 8 must map timeout, startup, crash, and malformed-response errors into complete user-visible recovery states.
4. Stage 9 must exercise packaged Windows execution, CUDA availability and device parity where hardware permits, cancellation, and repeated crash/restart stress.
5. The frozen test fixtures verify reconstruction and process behavior. They do not establish generalization, clinical validity, or external calibration.

### What to test next

Before Stage 3, review the parent/worker boundary, silent O04 computation from raw E08, no-tumor result suppression, explicit CPU/CUDA device choice, restart-without-auto-retry behavior, and final response validation. Stage 3 should keep the supervisor off the Qt event loop and preserve the Stage 2 result contract.

### Stage 2 correction — explicit CUDA device index

- **Date:** 2026-10-01 (Asia/Colombo)
- **Correction commit:** `5c1e9fa69bcd907f45acda7d2fadc449fb317e60` — `Fix explicit CUDA device initialization in Gyrus worker`
- **Trigger:** The laptop reported CUDA available, but the first direct GPU smoke check failed while loading because PyTorch 2.10 requires an indexed device for `torch.cuda.set_device`; the loader supplied `cuda` without an index.
- **Decision:** Select `cuda:0` explicitly. CPU behavior, model weights, model selection, inference equations, registered thresholds, and frozen files did not change. The loader still fails instead of falling back if CUDA cannot initialize.
- **`gyrus/worker/loader.py` purpose/why/content:** Fix explicit device initialization so requested GPU inference can start. The one-line change uses `torch.device("cuda:0")` before the same model-loading and verification checks. **Test:** `python -m pytest -q tests/test_gyrus_worker_inference.py::test_cuda_worker_loads_and_runs_on_requested_gpu tests/test_gyrus_worker_process.py::test_spawn_worker_runs_on_explicit_cuda_device`. **Edge case:** other GPU indices are not exposed in the Stage 2 protocol; the selected laptop device is index 0.
- **`tests/test_gyrus_worker_inference.py` purpose/why/content:** Add an actual CUDA inference check because simulated unavailability alone missed this bug. It opens the verified stack on `cuda:0`, analyzes the frozen elliptic fixture, and checks the result device, category, and final disclosure. **Test:** `python -m pytest -q tests/test_gyrus_worker_inference.py::test_cuda_worker_loads_and_runs_on_requested_gpu`. **Edge case:** skipped on machines without CUDA hardware.
- **`tests/test_gyrus_worker_process.py` purpose/why/content:** Add a real spawned CUDA worker check because a direct process test does not prove CUDA works after `spawn`. It verifies the ready device and completed job device are both CUDA. **Test:** `python -m pytest -q tests/test_gyrus_worker_process.py::test_spawn_worker_runs_on_explicit_cuda_device`. **Edge case:** skipped on machines without CUDA hardware.
- **Verification:** Direct CUDA test passed; spawned CUDA test passed; full suite `245 passed in 46.59s`; Ruff passed. The frozen manifest SHA256 remained `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`.
- **Remaining device limit:** Exact `1e-12` reconstruction was established on CPU. The CUDA tests establish successful execution and disclosure ordering; they do not claim CPU/GPU bitwise parity.

## Phase 6, Stage 3 — Qt application shell

- **Date:** 2026-10-01 (Asia/Colombo)
- **Implementation commit:** `18805e3` — `Build Gyrus Qt shell with isolated worker startup and high-DPI support`
- **Summary:** Added the local PySide6 shell, approved animated splash, guarded 14-state workflow, responsive empty screen, verification and device indicators, and a QThread bridge to the Stage 2 worker. The Qt thread receives only final validated worker responses; analysis and startup remain off the UI thread. No model result panel is exposed in this stage.

### Files

| Path | Purpose and why | Content | Test command | Known edge cases |
| --- | --- | --- | --- | --- |
| `pyproject.toml` | Declare GUI and Qt test dependencies reproducibly. | PySide6 and pytest-qt dependency entries. | `uv lock --check` | Windows Qt plugin and GPU driver availability vary by host. |
| `uv.lock` | Pin the resolved environment. | Locked Qt wheels and transitive packages. | `uv lock --check` | A different platform may resolve compatible wheel variants. |
| `gyrus/__main__.py` | Provide a simple local launch command. | Calls the GUI `main()` entry point. | `uv run python -m gyrus --help` | Default CUDA startup fails visibly if unavailable; use `--device cpu` explicitly. |
| `gyrus/gui/__init__.py` | Expose the supported Qt shell API. | Exports app creation, main window, and state machine. | `uv run python -m pytest -q tests/test_gyrus_gui_state.py` | Importing this package requires PySide6. |
| `gyrus/gui/app.py` | Wire startup without running worker work on the Qt event loop. | High-DPI QApplication, CLI device selection, splash/bridge/window signals, invalid-readiness lock. | `uv run python -m pytest -q tests/test_gyrus_gui_shell.py tests/test_gyrus_gui_threading.py` | CPU and CUDA are explicit modes; no automatic fallback or retry. |
| `gyrus/gui/main_window.py` | Present the approved empty shell and fail-closed integrity state. | 1440×900 window, 1180×720 minimum, rail/header/footer, notices, empty workspace, integrity dialog and device chip. | `uv run python -m pytest -q tests/test_gyrus_gui_shell.py` | Choose, History and result panels are deliberately inactive until later stages; an invalid ready payload locks analysis. |
| `gyrus/gui/splash.py` | Show animation while verification is actually pending. | Approved six-panel GIF, static fallback, pending verification lines, success/block transitions. | `uv run python -m pytest -q tests/test_gyrus_gui_shell.py::test_splash_uses_approved_animation_without_early_checkmarks` | The fallback uses the committed static mark if the GIF cannot load; no checkmarks appear before verification. |
| `gyrus/gui/state.py` | Define all 14 states without Qt dependencies. | Enum, guarded transition graph, reasoned transition history. | `uv run python -m pytest -q tests/test_gyrus_gui_state.py` | Stage 4 and later will activate result, edit, export and history transitions. |
| `gyrus/gui/theme.py` | Match the approved visual system at logical display sizes. | Exact Section 10 palette, font fallback, Qt stylesheet and transparent labels. | `uv run python -m pytest -q tests/test_gyrus_gui_shell.py tests/test_gyrus_gui_hidpi.py` | Headless Qt on this host exposes no fonts, so native Windows rendering was also visually checked. |
| `gyrus/gui/work_controller.py` | Keep synchronous worker startup and inference off the UI thread. | QThread service, explicit readiness, final-response-only signal, one busy job, queued shutdown. | `uv run python -m pytest -q tests/test_gyrus_gui_threading.py` | Closing waits up to 60 seconds for worker shutdown; failed jobs are never silently retried. |
| `tests/test_gyrus_gui_state.py` | Guard the full transition contract. | State coverage, allowed/disallowed transitions and history assertions. | `uv run python -m pytest -q tests/test_gyrus_gui_state.py` | Future panels may add transitions only with matching tests. |
| `tests/test_gyrus_gui_shell.py` | Verify shell dimensions and fail-closed startup presentation. | Geometry, footer disclaimer, readiness validation, splash and palette tests. | `uv run python -m pytest -q tests/test_gyrus_gui_shell.py` | Runs with the Qt offscreen plugin; native render was inspected separately. |
| `tests/test_gyrus_gui_threading.py` | Prove real worker execution occurs outside the UI thread. | Spawned CPU worker readiness and one final undisclosed-until-complete fixture result. | `uv run python -m pytest -q tests/test_gyrus_gui_threading.py` | Uses the committed frozen fixture, never the sealed BRISC test set; CUDA parity is not asserted. |
| `tests/test_gyrus_gui_hidpi.py` | Check supported 100%, 125%, 150%, 200% scales. | Isolated Qt subprocess at each scale, logical geometry and device-pixel-ratio assertions. | `uv run python -m pytest -q tests/test_gyrus_gui_hidpi.py` | Offscreen scaling verifies geometry, while final packaged display testing remains Stage 9. |
| `docs/app_build_log.md` | Preserve the append-only stage record for review. | This entry, source links, tests, gaps and commit provenance. | Read this entry and run `git diff --check` | Any correction must be appended, not silently edited into an accepted entry. |

### Verification

- Focused Stage 3 suite: **13 passed**.
- Full repository suite: **258 passed in 111.68 seconds**.
- Ruff: **all checks passed**; `uv lock --check` passed; `git diff --check` passed.
- Native Windows render of the 1440×900 empty shell inspected after correcting transparent label backgrounds. The splash GIF is valid and running in the Qt test.
- Frozen manifest SHA256 remained `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`; `runs/final_stack/` had no tracked changes.
- Zero BRISC sealed-test access, neural retraining/refitting, threshold changes, or network server work.

### Known gaps and next test

1. Stage 4 adds input selection, loading/OOD and result panels. No image can be selected from the Stage 3 shell yet.
2. Later stages add editing, opt-in history, PDF, and full error-state messaging. The states exist now but their panels do not.
3. Stage 9 should test packaged Windows startup on actual 100/125/150/200% displays and compare CPU/GPU outputs within an agreed tolerance. Bitwise device parity is not claimed.
4. Review the shell visually, check the 14-state graph, run the focused and full suites, and confirm the worker bridge has no automatic retry before approving Stage 4.

### Stage 3 correction — verified readiness metadata reaches the shell

- **Date:** 2026-10-01 (Asia/Colombo)
- **Correction commit:** `b61bcde` — `Expose verified artifact count to Gyrus startup shell`
- **Trigger:** A direct `main(["--device", "cpu"])` smoke launch stayed at `BOOT_VERIFYING`. The spawned worker had verified all 31 files, but `WorkerSupervisor.start()` returned a `SupervisorSnapshot` without `artifact_count`; the GUI's readiness check raised `AttributeError`. The prior focused test had only asserted the device and used a synthetic shell snapshot, so it missed this boundary mismatch.
- **Fix:** `gyrus/worker/process.py` now carries the already-validated `WorkerReady.artifact_count` into the immutable supervisor snapshot. No artifact verification logic, worker model, inference equation, or threshold changed.
- **Regression test:** `tests/test_gyrus_gui_threading.py` now hands a real spawned-worker snapshot to `GyrusMainWindow.on_worker_ready()` and requires 31 artifacts and `READY_EMPTY`. This tests the actual Stage 2/3 contract instead of only synthetic readiness.
- **Test commands:** `uv run python -m pytest -q tests/test_gyrus_gui_threading.py tests/test_gyrus_worker_process.py` (**4 passed**); `uv run python -m pytest -q` (**258 passed in 102.45 seconds**); `uv run ruff check .` (**all checks passed**). A separate offscreen `main(["--device", "cpu"])` smoke run observed `BOOT_VERIFYING → READY_EMPTY` and clean exit.
- **Known edge cases:** CPU startup can take around 20 seconds on this machine while the splash remains animated. CUDA startup and CPU/GPU parity are separate Stage 9 checks. A failed job is still not automatically retried.
- **Governance:** No BRISC sealed-test access, retraining/refitting, threshold change, or frozen-stack modification. Manifest SHA256 remains `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`.

## Phase 6, Stage 4 — Input, OOD gate, and read-only results

- **Date:** 2026-10-01 (Asia/Colombo)
- **Implementation commit:** `47de377` — `Build Gyrus input, OOD review, and read-only result panels`
- **Summary:** Enabled one-file picker and drag/drop, off-thread input validation, explicit analysis, a truthful named-stage loading display, an amber OOD disclosure gate, and STANDARD/REVIEW_REQUIRED read-only results. The result inspector pins the final disposition above OOD, raw category probabilities, internal mask reliability, pixel-only measurements, caveats, and expandable provenance. Original, Probability, Overlay, Contour, and Side-by-side views share verified model-space arrays with zoom, pan, fit, and opacity controls.

### Files

| Path | Purpose and why | Content | Test command | Known edge cases |
| --- | --- | --- | --- | --- |
| `gyrus/gui/input_controller.py` | Decode a selected file without freezing the Qt event loop. | QThread validates the PNG/JPEG input contract and returns immutable metadata, a bounded source preview, and exact 128×128 model-space grayscale pixels. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_file_selection_is_validated_without_autorun tests/test_gyrus_gui_core.py::test_bad_image_fails_before_analysis` | The worker validates the file again before inference; a changed file is rejected at the presentation boundary. A large but allowed image can take time to decode. |
| `gyrus/gui/presentation.py` | Keep unverified output out of widgets. | Checks final disclosure stage, manifest and selected-source hashes, exact policy thresholds, policy recomputation, array references, finite probabilities, mask threshold, and foreground count. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_selection_hash_mismatch_fails_closed tests/test_gyrus_gui_core.py::test_threshold_drift_fails_closed` | A mismatch locks result display; a no-tumor prediction has no spatial arrays or mask reliability score. |
| `gyrus/gui/image_canvas.py` | Provide aligned, read-only image inspection. | QGraphicsView with five tabs, calibrated probability heatmap, frozen mask overlay and contour, side-by-side model-space view, hand pan, zoom, fit and overlay opacity. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_standard_result_is_visible_only_after_complete_response` | Original tab uses a bounded source preview; spatial tabs use the frozen 128×128 coordinate frame and are disabled for no-tumor results. Editing is not present. |
| `gyrus/gui/result_panel.py` | Put policy disposition before model labels and keep provenance inspectable. | Pinned status card, scrollable evidence, internal/external limitations, exact thresholds, raw four-class probabilities, no-tumor explanation, truthful loading stages, and amber modal OOD gate. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_ood_review_is_modal_before_result_and_no_tumor_has_no_mask tests/test_gyrus_gui_core.py::test_ood_gate_choose_another_discards_undisclosed_output` | OOD review is a disclosure choice, not a refusal; closing the modal discards the pending result. No mask correctness is promised. |
| `gyrus/gui/main_window.py` | Join selection, validation, worker responses, and guarded UI states. | Local picker, one-file drag/drop, explicit Run action, input metadata, loading, OOD gate, final result, case clearing and fail-closed response handling. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py tests/test_gyrus_gui_shell.py` | Worker output is withheld until final validation. Fatal/inconsistent results lock display; detailed recovery copy is Stage 8. Untrusted filenames and errors render as plain text. |
| `gyrus/gui/app.py` | Connect the existing off-thread worker bridge to the new panels. | Routes submission, final success and failure signals while keeping the supervisor off the UI thread. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_complete_qt_worker_flow_discloses_only_final_result` | A submission race fails visibly; no silent retry or device fallback. |
| `gyrus/gui/theme.py` | Style new controls consistently with the approved palette. | Status borders, model canvas, tabs, indeterminate progress, provenance, and amber review heading. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py tests/test_gyrus_gui_shell.py` | Native Windows rendering was visually checked because the offscreen Qt plugin exposes no system fonts on this host. |
| `tests/test_gyrus_gui_core.py` | Prove the full Stage 4 workflow and trust boundary. | Tests explicit selection, invalid files, drag/drop, complete worker-to-Qt flow, result tabs, responsive dimensions, OOD modal actions, no-tumor suppression, source-hash mismatch and threshold drift. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Uses only committed synthetic frozen-stack fixtures; no BRISC sealed-test images. |
| `tests/test_gyrus_gui_shell.py` | Keep Stage 3 shell assertions aligned with Stage 4 behavior. | Chooser becomes enabled after verification; a result-integrity lock offers no unsafe retry button. | `uv run python -m pytest -q tests/test_gyrus_gui_shell.py` | Boot retry remains available for a recoverable startup failure, not an invalid verified result. |
| `docs/app_build_log.md` | Preserve the append-only application audit trail. | This Stage 4 record with files, tests, limits and implementation commit. | Read this entry; run `git diff --check` | Corrections must be appended in a new record. |

### Verification

- Focused Stage 4 workflow: **9 passed**; all focused GUI tests: **22 passed**.
- Full repository suite: **267 passed in 99.34 seconds**. After the final plain-text label hardening, the affected focused tests passed **14/14** and Ruff passed.
- `uv lock --check`, `git diff --check`, and `uv run ruff check .` passed.
- A real spawned CPU worker produced the STANDARD and OOD/no-tumor fixture paths. Native Windows screenshots of loading, the STANDARD inspector, amber OOD modal, and REVIEW_REQUIRED inspector were visually inspected; previews remain local temporary QA files.
- Frozen manifest SHA256 remained `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`. `runs/final_stack/` had no tracked changes. No BRISC sealed-test access, neural retraining/refitting, policy or mask-threshold change, or network server work.

### Known gaps and next test

1. The worker intentionally discloses no intermediate model events. While analysis runs, the UI marks validated input complete, shows the current frozen work as indeterminate, and leaves later named stages pending; the final trace marks them complete only after the validated response arrives. No percentage is fabricated.
2. Stage 5 adds non-destructive mask editing. Stage 6 adds opt-in history. Stage 7 adds PDF preview/export. Stage 8 expands recovery states and cancellation behavior. Stage 9 covers packaged Windows, accessibility and device-parity checks.
3. External cohorts failed segmentation transport; the OOD policy reviewed every evaluated external case, and external reliability was miscalibrated. The result inspector states these limitations; `STANDARD` means only the internal registered display policy passed.
4. Before Stage 5, inspect the native previews, run the focused GUI suite, and confirm the OOD gate and no-tumor path with the committed synthetic fixtures.

## Phase 6, Stage 5 — Non-destructive mask editor

- **Date:** 2026-10-01 (Asia/Colombo)
- **Implementation commit:** `9da2ee3` — `Add non-destructive Gyrus mask editor with verified provenance exports`
- **Summary:** Added a 56 px in-panel editor strip, add/erase disk brush, 1–32 model-pixel brush size, undo/redo, reset, and Model/Edited/Difference/Split views. The frozen model mask is copied into a separate human layer. The difference view uses cyan for shared model pixels, violet for additions, and pink diagonal hatch for removals. Saving is an explicit local action to `%LOCALAPPDATA%\GyrusResearch\edited_layers\`; it writes a binary PNG and hash-linked provenance JSON in a versioned per-case folder. No history index is created. `USER EDITED` appears only after a successful save. The original reliability score and registered segmentation mask remain unchanged.

### Files

| Path | Purpose and why | Content | Test command | Known edge cases |
| --- | --- | --- | --- | --- |
| `gyrus/domain/mask_editor.py` | Separate manual editing state from immutable model evidence. | Pixel-exact disk rasterization, bounded strokes, undo/redo/reset, original/current/saved masks, chained edit records through `mask_provenance.py`, verification before each operation, explicit local PNG/JSON export, non-overwriting versioned folders. | `uv run python -m pytest -q tests/test_gyrus_mask_editor.py` | A no-op stroke creates no record; off-image points and invalid sizes fail; a broken mask hash locks the UI. A second save creates a new version. Export refuses a network/unknown Windows drive, and an existing target is never overwritten. |
| `gyrus/gui/mask_editor.py` | Let users correct a mask in the image panel without a separate window. | 56 px tool rail; viewport-to-128×128 coordinate mapping; add/erase stroke gestures, middle-button pan, zoom/fit, brush-size slider, four view tabs, live pixel counts, cyan/violet/pink-hatch renderer, and integrity-failure signal. | `uv run python -m pytest -q tests/test_gyrus_mask_editor.py::test_qt_brush_maps_click_to_model_space tests/test_gyrus_mask_editor.py::test_difference_colors_and_hatch_are_distinct` | Model and Split are comparison views, not paint targets. Fit runs after the widget becomes visible so the image does not remain at its pre-layout size. No smoothing or auto-fill is applied. |
| `gyrus/gui/main_window.py` | Guard entry/exit and save against the existing 14-state workflow. | Opens editing only after a disclosed tumor-mask result; keeps worker output unchanged; saves to local application data on explicit click; confirms discard of unsaved work; locks the result on edit-integrity failure; disables conflicting result actions while editing. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_mask_editor_save_preserves_frozen_result tests/test_gyrus_gui_core.py::test_no_tumor_editor_unavailable_and_edit_tamper_locks` | No-tumor results cannot enter the editor. A local I/O failure reports no successful save; a hash mismatch locks result display. The notice timer belongs to the window, so destruction cannot leave a callback to a deleted Qt label. |
| `gyrus/gui/result_panel.py` | Make human changes visible without altering the original policy status. | Adds Edit mask action, `USER EDITED` badge after save, edited pixel count, stored edit provenance, and explicit text that reliability applies only to the original model mask. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_mask_editor_save_preserves_frozen_result` | STANDARD/REVIEW_REQUIRED disposition remains the frozen model decision. Human additions do not change or improve the displayed reliability score. |
| `gyrus/gui/theme.py` | Keep editor controls consistent with the approved interface. | Styles the narrow tool strip, selected brush tool, and violet edited badge using the existing palette. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Native Windows preview was used because this host's offscreen Qt renderer does not expose system fonts. |
| `tests/test_gyrus_mask_editor.py` | Catch editing, export, and visual-difference regressions without real patient images. | Tests immutable model evidence, brush/undo/redo/reset hash chain, explicit export and collision, hash tampering, pixel colors/hatch, bounds, and Qt click-to-pixel mapping. | `uv run python -m pytest -q tests/test_gyrus_mask_editor.py` | Uses synthetic 128×128 arrays; it does not evaluate medical correctness of edits. |
| `tests/test_gyrus_gui_core.py` | Verify that manual editing cannot contaminate a frozen result or bypass disclosure. | Adds end-to-end tests for tumor editor entry, save/badge/provenance, frozen reliability and mask hash, no-tumor suppression, and tamper-triggered integrity lock. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Uses committed synthetic frozen-stack fixtures; no BRISC sealed-test input is read. |
| `docs/app_build_log.md` | Preserve the append-only build record. | This Stage 5 record, verification evidence, gaps, commit provenance, and deferred packaging scope. | Read this entry and run `git diff --check` | Corrections must be appended as new entries, never silently rewritten into an accepted stage. |

### Verification

- Focused editor and GUI regression: **23 passed**. Full repository suite: **276 passed in 98.32 seconds**.
- `uv run ruff check .`, `uv lock --check`, and `git diff --check` passed.
- Native Windows visual previews at 820 px and 640 px logical editor widths confirmed image fit, legible controls, distinct difference colors, and 56 px tool rail. Previews used synthetic masks and stayed in the local temporary directory; they are not clinical evidence.
- Frozen manifest SHA256 remained `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`. `runs/final_stack/` had no tracked changes. No BRISC sealed-test access, retraining/refitting, or threshold change occurred.

### Known gaps and next test

1. Saving a human layer writes a local export only after the user clicks Save. Opt-in indexed history is Stage 6; no history record is created in Stage 5. PDF handling of edited masks is Stage 7.
2. The editor reports pixel counts only; it cannot establish physical size or clinical correctness. Reliability remains a frozen estimate about the original model mask, regardless of the human edit.
3. Packaged Windows and CPU/GPU parity remain unverified. Stage 9 will exercise the packaged app at 100/125/150/200% display scaling and the agreed device-output tolerance.
4. For review, open a disclosed tumor result, edit via Add and Erase, compare Model/Edited/Difference/Split, undo/redo/reset, save explicitly, inspect `USER EDITED` and provenance, then confirm the original reliability score is unchanged.

### Deferred Phase 7 packaging scope — do not implement in Stage 5

- Build Windows standalone `Gyrus.exe` with PyInstaller.
- Bundle Python, PySide6, CPU PyTorch, `runs/final_stack/`, and `assets/`.
- Use `assets/logo/gyrus.ico` for the executable.
- Consider an optional Inno Setup MSI installer.
- Verify launch and manifest checks on a fresh Windows system without Python installed.

### Stage 5 correction — fit timer lifecycle

- **Date:** 2026-10-01 (Asia/Colombo)
- **Correction commit:** `737d3bb` — `Bind mask editor fit timer to widget lifetime`
- **Trigger:** The first-show image-fit callback used a free single-shot timer. An editor closed immediately after showing could leave a queued Python callback referencing a destroyed Qt widget, analogous to the notice-timer issue found in Stage 5 testing.
- **Fix:** The editor now owns one single-shot `QTimer`; show and resize events restart it, and Qt destroys it with the widget. Image-fitting geometry and model-space coordinates are unchanged.
- **Verification:** `uv run ruff check .` passed; **23 focused editor/GUI tests passed**. A native Windows preview at 820 logical pixels showed the full image and difference overlay after the timer change. The previously completed full suite passed **276 tests** before this lifecycle-only correction.
- **Governance:** Frozen manifest SHA256 remained `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`. No BRISC sealed-test access, model work, threshold change, or modification to `runs/final_stack/`.
- **Post-correction full-suite verification:** `uv run python -m pytest -q` passed **276 tests in 97.62 seconds** after commit `737d3bb`. The stage is complete with the lifecycle correction included.

## Phase 6, Stage 6 — Opt-in local history

- **Date:** 2026-10-01 (Asia/Colombo)
- **Implementation commit:** `f7f1e59` — `Add opt-in Gyrus history with verified case bundles and filters`
- **Summary:** Activated the History navigation button and added newest-first saved-case rows, status/category/OOD/edit/date filters, a verified overlay preview, Open and Delete actions, and an empty state explaining that retention is off. A disclosed result is saved only after the user clicks **Save to local history** and confirms the exact file list. The app stores source, result, model evidence, optional saved human layer, and preview in a per-case bundle beneath `%LOCALAPPDATA%\GyrusResearch\cases\`. Files are staged and hashed before the bundle is committed; the SQLite index is inserted last. Reopening verifies the manifest, source, preprocessing, model arrays, edit chain, and frozen policy before display. Export remains disabled until Stage 7.

### Files

| Path | Purpose and why | Content | Test command | Known edge cases |
| --- | --- | --- | --- | --- |
| `gyrus/domain/history.py` | Keep the local index transactional and contained. | Additive schema-2 migration, source-saved and manifest-hash metadata, newest-first filtered/paginated queries, direct-child path checks, and staged deletion with rollback on database failure. | `uv run python -m pytest -q tests/test_gyrus_history_repository.py` | Legacy schema-1 rows migrate but have no verifiable bundle manifest and cannot be reopened through the new verified viewer. A failed final tombstone cleanup is reported explicitly; deletion is not forensic erasure. |
| `gyrus/domain/history_bundle.py` | Save and restore evidence without trusting an unverified history file. | Typed payload/loaded-case records; exact retention file plan; frozen threshold and policy checks; source and model-space hashes; optional edit-chain checks; staged files, bundle manifest, final SQLite insert, and fail-closed load. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | A changed source, missing/extra file, hash mismatch, invalid mask, or policy drift prevents save or preview. No-tumor cases retain no mask. Original source directory is redacted from archived `result.json`; source filename itself may still contain identifying text. |
| `gyrus/domain/mask_editor.py` | Restore an explicitly saved human layer without changing model evidence. | `from_saved` reconstructs the separate edited layer and provenance while retaining the original model mask and reliability scope. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_saved_human_layer_restores_without_recomputing_reliability` | The previous session's undo stack is not replayed on reopen; new edits can form a new chain. Reliability remains tied to the original model mask. |
| `gyrus/gui/history_view.py` | Make saved cases inspectable without opening them blindly. | Search and five filter dimensions, date controls, fixed-height rows, 300 px overlay preview, Open/Delete actions, disabled Stage-7 Export control, and retention-off empty state. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_history_is_opt_in_filters_reopens_and_deletes` | A corrupt bundle does not show a preview. The list shows stored metadata, while preview and Open require full bundle verification. |
| `gyrus/gui/main_window.py` | Connect consent, local storage, and verified restore to the 14-state shell. | Active History navigation; exact-file retention confirmation; fixed-local-drive history location; save, filter, preview, reopen, deletion confirmation, and fail-closed error handling. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Cancelled consent creates no history database. Saving a large permitted source is currently synchronous and can briefly pause the interface; move I/O off-thread during polish if this is noticeable. Deletion reports cleanup failures and makes no forensic-erasure promise. |
| `gyrus/gui/result_panel.py` | Make opt-in retention an explicit result action. | Adds **Save to local history**; no automatic saving on analysis. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | The action is unavailable outside a disclosed result or after that case is saved. |
| `gyrus/gui/state.py` | Restore the pre-history workflow correctly. | Allows History to return to a validated, not-yet-analyzed image. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | History does not interrupt analysis, editing, or export preview. |
| `gyrus/gui/theme.py` | Keep history controls legible in the established visual system. | History table padding, filter and preview styling. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Native screenshot review used synthetic UI data; DPI variants still need packaged-app visual testing. |
| `tests/test_gyrus_history_repository.py` | Prevent index, migration, and containment regressions. | Newest-first pagination and filters, direct-child restriction, deletion, and schema-1-to-2 migration tests. | `uv run python -m pytest -q tests/test_gyrus_history_repository.py` | Temporary synthetic directories and records only. |
| `tests/test_gyrus_gui_core.py` | Verify the consent-to-restore flow end to end. | Opt-in/no-DB behavior, filters, reopen/delete, tamper lock, edited-layer restoration, no-tumor bundle, cancel, failed index rollback, and changed-source rejection. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Uses synthetic frozen-stack fixtures; it does not establish clinical performance. |
| `tests/test_gyrus_gui_shell.py` | Keep navigation expectations aligned with the completed screen. | Asserts History becomes available after verified startup. | `uv run python -m pytest -q tests/test_gyrus_gui_shell.py` | History is unavailable while startup verification is pending or blocked. |
| `docs/app_build_log.md` | Preserve the append-only handoff record. | This Stage 6 inventory, verification, known gaps, and implementation commit. | Read this entry; run `git diff --check` | Subsequent corrections must be appended, not silently substituted into this accepted entry. |

### Verification and review

- Focused history/GUI suite: **24 passed**. Full repository suite: **285 passed in 99.23 seconds**. `uv run ruff check .`, `uv lock --check`, and `git diff --check` passed.
- A native 1440×900 Windows preview of the History screen used a synthetic ellipse and was inspected for row legibility, filters, and preview layout. The preview is a UI check, not a medical-image result.
- Frozen manifest SHA256 remained `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`; no tracked file under `runs/final_stack/` changed. No BRISC sealed-test access, retraining/refitting, segmentation-threshold change, or other registered threshold change occurred.
- For review, start Gyrus, open History before saving and confirm the empty state; analyze a permitted local image, click **Save to local history**, inspect the file list and consent dialog, then filter, preview, reopen, and delete. Check that a manually saved edit reopens with `USER EDITED` while the original reliability score remains frozen. The Export button should remain disabled until Stage 7.

### Known gaps

1. PDF export from history is Stage 7. There is no automatic retention, cloud sync, or identity-field form.
2. The archived source filename and image content can themselves contain identifying information; the confirmation states this, and storage remains local and opt-in. Deletion removes Gyrus-managed files and the index but cannot promise forensic erasure, backup removal, or recovery from storage media.
3. History save copies a permitted source on the UI thread. If noticeable with larger files, Stage 9 polish should move the copy and hash work to a bounded background task while retaining consent and atomic commit semantics.
4. Fresh-Windows packaging, high-DPI variants, and CPU/GPU parity remain planned verification work. Stage 7 may consume the verified saved bundle for PDF export but must not loosen its integrity checks.

## Phase 6, Stage 7 — Deterministic three-page PDF export

- **Date:** 2026-10-01 (Asia/Colombo)
- **Implementation commit:** `8a782f7` — `Add deterministic three-page Gyrus PDF preview and local export`
- **Summary:** Enabled Export PDF on live results and verified saved cases. A typed, frozen `PdfReportModel` now renders exactly three A4 pages from validated evidence; repeated rendering of the same payload yields identical bytes. Page 1 shows status, reason, input metadata, four category probabilities, source preview, and model overlay. Page 2 shows the C03 probability image, frozen 0.50 mask, pixel-only measurements, and separate saved human-mask/difference panels when present; the no-tumor path explains why no mask exists. Page 3 shows OOD and reliability evidence, full-precision thresholds, stack and input hashes, execution information, and limitations. Each page has the required top banner and footer disclaimer. The PDF creator is `Gyrus 0.1.0`; there are no form fields or editable diagnosis field. A Qt `QPdfView` modal renders and verifies all three pages before offering an atomic save to a local path.

### Files

| Path | Purpose and why | Content | Test command | Known edge cases |
| --- | --- | --- | --- | --- |
| `gyrus/domain/report_model.py` | Reuse the Stage 1 immutable three-page contract for both live and archived results. | Adds a builder for serialized, verified result payloads while preserving the original typed-result builder, fixed page order, edit-scope check, and exact threshold strings. | `uv run python -m pytest -q tests/test_gyrus_domain_persistence_models.py tests/test_gyrus_pdf_export.py` | The generation timestamp is the frozen analysis completion time, so repeat exports cannot drift; malformed model data fails before rendering. |
| `gyrus/domain/pdf_export.py` | Make PDF content deterministic and independent of widgets. | Fail-closed source/preprocessing/policy/array/edit-chain verification, local bundled Bitstream Vera fonts, deterministic ReportLab PDF bytes, three A4 page layouts, source and overlay images, probability/mask/difference views, hashes, thresholds, and disclaimer on every page. | `uv run python -m pytest -q tests/test_gyrus_pdf_export.py` | No-tumor has no fabricated mask. Long text wraps. A changed source, mismatched mask, invalid edit chain, or threshold drift prevents preview. Measurements remain model pixels, not millimeters. |
| `gyrus/gui/pdf_preview.py` | Preview the exact PDF bytes before local save. | 1160×780 modal with `QPdfDocument`/`QPdfView`, bounded three-page load verification, summary sidebar, user-chosen local destination, known sync-folder rejection, structural/disclaimer validation, staged write, and atomic non-overwriting creation or explicit replacement. | `uv run python -m pytest -q tests/test_gyrus_pdf_export.py::test_preview_loads_three_pages_and_atomic_local_save` | Network/unknown drives, symlink paths, known cloud-sync directories, bad PDFs, and unapproved overwrite are rejected. An unrecognized third-party sync tool outside known paths cannot be detected by the app. |
| `gyrus/gui/main_window.py` | Enforce the report trust boundary for result and history export. | Builds one PDF payload from the disclosed result or freshly reverified saved bundle; enters `EXPORT_PREVIEW` only after rendering and Qt document validation; returns to its prior result/history state after the modal closes. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py::test_live_and_saved_pdf_preview_use_verified_three_page_document tests/test_gyrus_gui_core.py::test_saved_pdf_export_reverifies_bundle_before_preview` | A changed archived bundle locks result disclosure; invalid live evidence also locks. Export does not alter the model mask, reliability score, or history record. Rendering and source verification are currently synchronous. |
| `gyrus/gui/result_panel.py` | Expose export only for a disclosed, validated result. | Adds the Export PDF action below the existing result actions. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Disabled before a final result and while editing. Saved edits appear only after they have been committed to the separate human layer. |
| `gyrus/gui/history_view.py` | Permit PDF export from a verified saved case. | Activates Export PDF for a selected, verified bundle and clears it when no valid case is selected. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | Selection is reverified immediately before rendering, so later bundle tampering cannot use an old preview. |
| `gyrus/gui/state.py` | Keep modal export inside the existing guarded state machine. | Adds `HISTORY ↔ EXPORT_PREVIEW` transitions; result-to-preview transitions already existed. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | The preview modal blocks other workflow actions and returns to the exact underlying state. |
| `tests/test_gyrus_pdf_export.py` | Verify PDF fidelity and fail-closed behavior. | Synthetic frozen-stack cases test three pages, exact disclaimer and thresholds, metadata, deterministic bytes, no-tumor path, saved edit separation, array/source/threshold tampering, Qt preview, atomic save, cloud-path rejection, and difference colors. | `uv run python -m pytest -q tests/test_gyrus_pdf_export.py` | Synthetic fixtures demonstrate software behavior, not medical validity. |
| `tests/test_gyrus_gui_core.py` | Verify live and history export integration. | Adds modal state/return checks and a changed-bundle-after-selection test. | `uv run python -m pytest -q tests/test_gyrus_gui_core.py` | The saved case is reloaded before PDF construction; a corrupted bundle cannot be exported. |
| `pyproject.toml` | Declare renderer and PDF inspection requirements reproducibly. | Adds `reportlab` and `pypdf`. | `uv lock --check` | Stage 9 Windows packaging must include ReportLab's bundled Vera fonts and Qt PDF components. |
| `uv.lock` | Freeze the resolved PDF dependencies. | Locked ReportLab and pypdf versions and transitive metadata. | `uv lock --check` | Reproducibility still depends on the recorded OS/runtime and packaged-font verification. |
| `docs/app_build_log.md` | Preserve the append-only stage record. | This Stage 7 file inventory, verification, limitations, and implementation commit. | Read this entry and run `git diff --check` | Any correction must be appended as a new record. |

### Verification and review

- Focused PDF and GUI checks: **23 passed** after the final renderer checks. Full repository suite: **293 passed in 105.15 seconds**. `uv run ruff check .`, `uv lock --check`, and `git diff --check` passed.
- The same synthetic tumor-case payload produced byte-identical PDFs on repeated renders. `pypdf` verified three pages, creator metadata, no form fields, page-by-page banner/disclaimer, full-precision thresholds, and provenance hashes. Qt PDF loaded all three pages. Native 1000×1414 renders of all three pages, an edited-mask page, and a 1160×780 preview-modal capture were visually inspected: no clipped text, missing images, or unreadable footer was found. Preview files stayed in the local temporary directory and are not model-performance evidence.
- Frozen manifest SHA256 remained `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`; no tracked file in `runs/final_stack/` changed. No BRISC sealed-test access, retraining/refitting, segmentation-threshold change, or other registered threshold change occurred.
- For review, open a disclosed result and click Export PDF; inspect all three pages before saving. Repeat from a saved case in History, then change one bundle byte and confirm export locks rather than using the old preview. Check a no-tumor result and a saved human edit separately.

### Known gaps

1. The app itself makes no network request and rejects network drives and recognizable sync folders for PDF saves. It cannot detect every third-party sync agent that may watch a user-selected local folder. The save dialog starts in `%LOCALAPPDATA%\GyrusResearch\exports\`.
2. PDF source revalidation and rendering currently run on the UI thread; a large permitted image can pause the interface briefly. This is a Stage 9 polish candidate, as with Stage 6 history copy.
3. Fresh-Windows packaging must include ReportLab's bundled Vera font files, pypdf, and Qt PDF components, then recheck byte stability and visual layout at the supported display scales.
4. No print-to-server or cloud export is implemented. The PDF is a fixed research summary and does not turn a model output into a medical conclusion.

## Phase 6, Stage 8 — Fail-closed errors and local diagnostics

- **Date:** 2026-10-01 (Asia/Colombo)
- **Implementation commit:** `f7b4808` — `Stage 8: Add fail-closed error handling and local diagnostics`
- **Summary:** Reused the existing coded domain exceptions and typed worker failure protocol. Unexpected failures now get a short safe explanation and reference in the UI, while developer exceptions and tracebacks go to a bounded rotating log at `%LOCALAPPDATA%\GyrusResearch\logs\gyrus.log` on a local drive. Artifact, OOD, result, and saved-evidence verification failures lock disclosure. A bad image remains recoverable. PDF and edit-layer I/O errors leave the current case intact without claiming success. Worker close interrupts an active job without an automatic retry or partial result.

### Files

| Path | Purpose and why | Content | Test command | Known edge cases |
| --- | --- | --- | --- | --- |
| `gyrus/domain/error_reporting.py` | Centralize safe UI copy and private diagnostics without replacing established domain error codes. | Immutable error notice with code/reference, local-only rotating file handler (2 MB, three backups), full exception logging, and a copyable safe summary. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py` | If local app data is unavailable, linked, or on an unsupported drive, file logging is unavailable and the error still fails closed. Multiple processes writing during a log rollover warrant packaged-app stress testing. |
| `gyrus/gui/app.py` | Initialize diagnostics and keep submission exceptions from becoming raw UI text. | Configures the local log at launch; logs failed submissions and sends only a safe failure token to the window. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py tests/test_gyrus_gui_threading.py` | Worker start and analysis remain off the UI thread; no device switch occurs automatically. |
| `gyrus/gui/input_controller.py` | Keep unexpected preview failures out of the UI. | Logs the exception and emits a safe `INPUT_PREVIEW_FAILED` message; registered input contract rejections retain their stable codes. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py tests/test_gyrus_gui_core.py` | File validation is still a separate bounded Qt thread; a bad source never starts inference. |
| `gyrus/gui/main_window.py` | Give blocking and recoverable failures clear actions without showing tracebacks. | Safe error records, reference IDs, Copy details/Open local log/Close actions, verified retry classification, OOD and policy hard stops, protected PDF/history/edit errors, and disabled controls in locked state. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py tests/test_gyrus_gui_shell.py tests/test_gyrus_gui_core.py` | Artifact hash failure names the affected relative artifact. Only CUDA availability/initialization and startup timeout offer verification retry; changing device requires an explicit relaunch. |
| `gyrus/gui/pdf_preview.py` | Prevent destination errors from leaking raw exceptions or claiming an export. | Handles export-directory and atomic-save failures with a safe message and local traceback. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py tests/test_gyrus_pdf_export.py` | The preview and case remain available after a destination failure. PDF render and Qt document loading remain synchronous; a five-second Qt load bound already exists. |
| `gyrus/gui/work_controller.py` | Preserve typed failure routing and end active work on application close. | Logs startup/job exceptions; a busy close signals the isolated supervisor before queued Qt shutdown. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py tests/test_gyrus_gui_threading.py` | Closing does not retry the interrupted job. A hard OS kill cannot guarantee a final log flush. |
| `gyrus/worker/process.py` | Contain model and policy faults in the spawned process. | Logs startup/inference exceptions; treats only invalid source images as recoverable; returns a fatal failure for OOD/policy/result faults; reports CUDA OOM without fallback; cancels a job on close; retains the existing job timeout and restart protocol. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py tests/test_gyrus_worker_process.py` | A timeout or crash restarts the worker generation but never resubmits the failed job. CPU is selectable only by explicit launch. |
| `tests/test_gyrus_error_handling.py` | Prove error behavior at trust and I/O boundaries. | Twenty-one synthetic tests covering traceback separation, locked controls, copyable details, invalid input, OOD failures, artifact naming, PDF failures, corrupt history preservation, permission errors, worker failure classification, OOM, timeout, and close cancellation. | `uv run python -m pytest -q tests/test_gyrus_error_handling.py` | Synthetic fixtures verify software behavior; they do not measure clinical performance or CPU/GPU parity. |
| `docs/app_build_log.md` | Maintain the append-only stage record. | This Stage 8 file inventory, verification, and known gaps. | `git diff --check` | Corrections must be appended as new records rather than editing an accepted stage entry. |

### Verification and review

- **314 full-suite tests passed in 103.19 seconds**, including 21 new Stage 8 tests. `uv run ruff check .`, `uv lock --check`, and `git diff --check` passed.
- `verify_frozen_stack('runs/final_stack')` verified **31 artifacts** and manifest SHA256 `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`. No file under `runs/final_stack/` changed. No BRISC sealed-test access, neural retraining/refitting, segmentation-threshold change, or other threshold change occurred.
- For review, try one unreadable PNG, a read-only PDF destination, and a corrupt synthetic history index. The first must allow another image, the second must retain the result without a saved-file claim, and the third must lock saved-case display while leaving the database bytes untouched. The locked view should offer Copy details and Open local log without showing a traceback.

### Known gaps

1. History save/copy and PDF source verification/rendering still run synchronously on the UI thread. Stage 9 polish should move the expensive work off-thread while preserving atomic commits and the existing verification boundary.
2. The app has no automatic GPU-to-CPU fallback. On device failure, the user must relaunch explicitly with `--device cpu`; packaged-app device selection needs a usable Windows workflow.
3. Rotating-file rollover under simultaneous GUI/worker logging and abrupt OS termination need fresh-Windows stress testing. Log files are local but can contain local paths and developer exception details; they should be treated as sensitive user data.
4. Fresh-Windows packaging, high-DPI visual checks, and CPU/GPU numerical parity remain Stage 9 / Phase 8 verification work. Neither the frozen stack nor any registered threshold was changed.

## Phase 6, Stage 9 — Integration and visual verification

- **Date:** 2026-10-02 (Asia/Colombo)
- **Implementation commit:** `7ac20aa` — `Verify Gyrus visual, accessibility, input, and device parity behavior`
- **Summary:** Added native Windows screenshot baselines for both supported shell widths, checked primary accessible control names and opt-in history behavior, hardened bounded image decoding against a changed file, and compared frozen CPU and CUDA inference on two non-test fixtures. The first parity run exposed a CUDA CuDNN TF32 difference amplified by a steep C03 isotonic step (maximum calibrated-map difference 0.03055); disabling CuDNN TF32 for the explicit CUDA worker reduced the fixture difference to zero without changing any model artifact, preprocessing rule, calibration mapping, or policy threshold. This is app execution precision, not a new research result.

### Files

| Path | Purpose and why | Content | Test command | Known edge cases |
| --- | --- | --- | --- | --- |
| `gyrus/domain/input_contract.py` | Bind decoding to exactly the bytes that were size-checked and hashed. | Reads at most the configured limit plus one byte, detects a changed-length read, and verifies/decodes two in-memory views of the same bytes. | `uv run python -m pytest -q tests/test_gyrus_domain_input_contract.py` | A hostile concurrent writer could change bytes without changing length between metadata inspection and read; the resulting bytes are still the ones hashed and decoded. |
| `gyrus/worker/loader.py` | Make explicit CUDA inference numerically comparable to the frozen CPU reference. | Disables CuDNN TF32 convolution before CUDA model loading; all registered thresholds and frozen artifacts remain unchanged. | `uv run python -m pytest -q tests/test_gyrus_device_parity.py` | Parity is demonstrated on two fixed non-test images within a QA tolerance of `1e-4`, not universally or bitwise for every possible input. No silent CPU fallback was added. |
| `gyrus/gui/main_window.py` | Give analysis and error actions screen-reader names. | Accessible names for Run analysis, Copy details, Open log, and Close application. | `uv run python -m pytest -q tests/test_gyrus_stage9_quality.py` | This is a control-name check; it is not a full assistive-technology usability study. |
| `gyrus/gui/image_canvas.py` | Make image navigation controls discoverable. | Accessible names for the viewer, zoom, fit, and opacity controls. | `uv run python -m pytest -q tests/test_gyrus_stage9_quality.py` | The canvas's spatial content needs a human-readable textual alternative in a later accessibility review. |
| `gyrus/gui/history_view.py` | Label saved-case navigation for assistive tools. | Accessible names for search, case table, and preview. | `uv run python -m pytest -q tests/test_gyrus_stage9_quality.py` | Patient-derived filenames can still be sensitive if the user opts to save them; no identity form is added. |
| `gyrus/gui/pdf_preview.py` | Name PDF preview and save actions. | Accessible names for `QPdfView` and local-save button. | `uv run python -m pytest -q tests/test_gyrus_pdf_export.py tests/test_gyrus_stage9_quality.py` | Qt PDF's detailed page reading behavior depends on platform accessibility support. |
| `scripts/capture_gyrus_golden.py` | Regenerate reviewed visual baselines without model inference or patient data. | Native Windows Qt capture of verified empty-shell states at 1440×900 and 1180×720 logical requests. | `uv run python -m pytest -q tests/test_gyrus_stage9_quality.py` | Native window height may be clamped by the desktop work area; screenshots record the resulting physical pixels. |
| `tests/golden/gyrus_ready_1440x900.png` | Preserve the reviewed default-width empty screen. | Native readable screenshot; no image or model output. | `uv run python -m pytest -q tests/test_gyrus_stage9_quality.py` | Exact comparison is host-specific; a different pixel size skips with an explicit reason. |
| `tests/golden/gyrus_ready_1180x720.png` | Preserve the reviewed minimum-width empty screen. | Native readable screenshot; no image or model output. | `uv run python -m pytest -q tests/test_gyrus_stage9_quality.py` | Font or operating-system rendering changes at identical pixel size can require deliberate baseline review. |
| `tests/test_gyrus_domain_input_contract.py` | Detect malformed or changed inputs before analysis. | Adds disguised-SVG and truncated-read cases. | `uv run python -m pytest -q tests/test_gyrus_domain_input_contract.py` | The test simulates a shortened read; filesystem races can have more forms. |
| `tests/test_gyrus_device_parity.py` | Check device-dependent disclosure and mask outcomes. | Compares classifier probabilities, OOD score/status, reliability score, final policy, binary model mask, and calibrated map on two frozen non-test fixtures. | `uv run python -m pytest -q tests/test_gyrus_device_parity.py` | Skips without CUDA hardware; uses `1e-4` only as a software QA tolerance, never as a registered decision threshold. |
| `tests/test_gyrus_stage9_quality.py` | Guard visual, accessibility, and retention defaults. | Native golden comparisons, key accessible names and footer disclaimer, and absence of a history folder before explicit opt-in. | `uv run python -m pytest -q tests/test_gyrus_stage9_quality.py` | Native golden comparison skips on non-Windows or a different display pixel size; the remaining tests run under Qt offscreen. |
| `docs/app_build_log.md` | Keep the requested append-only stage handoff. | This entry records the implementation, checks, and deferred work. | `git diff --check` | Future corrections must be appended as a separate dated record. |

### Verification and review

- **321 tests passed in 159.08 seconds**. `uv run ruff check .`, `uv lock --check`, and `git diff --check` passed. The two native Windows screenshot comparisons passed on the current laptop, and the default-width image was visually inspected for readable text and unclipped content.
- `verify_frozen_stack('runs/final_stack')` reverified **31 artifacts** and manifest SHA256 `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`; no tracked file in `runs/final_stack/` changed. The registered segmentation `0.50`, reliability `0.9490250173235835`, and OOD `0.36854275476100085` thresholds remain unchanged. No BRISC sealed-test access or neural retraining/refitting occurred.
- Stage 8's PDF, history, error-path, worker crash/restart, policy, parser, and GUI tests remained in the full suite. The new tests strengthen specific Stage 9 gaps without treating synthetic fixture performance as clinical evidence.

### Known gaps and next phase

1. History copying and PDF source verification/rendering can still pause the UI. Moving these operations to bounded background work is Phase 7 polish; the existing integrity and atomic-write behavior must be retained.
2. The native golden images cover the verified empty shell. Result, OOD gate, edit, history, PDF, and error screens have functional Qt tests and prior manual visual inspection, but do not yet have cross-machine pixel baselines.
3. A standalone `Gyrus.exe`, bundled CPU runtime/assets, and a clean Windows-without-Python trial are Phase 7 packaging work, not claimed here.
4. CPU/CUDA parity is established only for two frozen synthetic fixtures. Phase 8 must record its chosen device before sealed evaluation and must not interpret this QA result as universal bitwise parity.
