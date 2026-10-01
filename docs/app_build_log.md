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
