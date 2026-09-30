# Application design proposal

**Status:** proposal only; no application implementation is authorized by this document.
**Application name:** **Gyrus**.
**Application type:** local-only, single-user Windows desktop research application.
**Binding disclaimer:** **Research prototype — not for clinical use**.

**Naming contract:** window title `Gyrus — Brain MRI Research Viewer`; executable `Gyrus.exe`; local folder `%LOCALAPPDATA%\GyrusResearch\`; icon `gyrus.ico`; process name `Gyrus`; PDF creator `Gyrus 0.1.0`; default PDF filename `gyrus-research-summary_<case-id>.pdf`.

## 1. What the app is

The application is a local research workstation for inspecting one two-dimensional brain MRI image at a time with the frozen project stack. It verifies the frozen artifacts before use, evaluates whether the upload resembles the BRISC development distribution, shows the model-predicted BRISC category, renders the frozen E08 probability map and hard mask when a tumor category is predicted, calculates the R02 internal mask-reliability score, and assigns either `STANDARD` or `REVIEW_REQUIRED`. It is deliberately an inspection tool rather than an automated diagnosis screen: every output remains visible with its provenance and limitations, external-looking inputs are routed to review rather than presented normally, manual edits are stored separately from the model mask, and nothing is transmitted off the machine.

## 2. What the user does

### 2.1 Normal workflow

1. **Launch the application.** A short startup screen verifies the SHA256 of every allowlisted artifact in `runs/final_stack/`, confirms that no unexpected artifact exists, loads the thresholds from the manifest as exact strings, checks GPU availability, and runs a lightweight frozen-fixture self-test. The Analyze button remains disabled until integrity passes.
2. **See the ready screen.** The user sees `Local only`, `Frozen stack verified`, the manifest hash prefix, GPU/CPU execution mode, supported file types, and the permanent disclaimer.
3. **Choose one image.** The user drags a PNG or JPEG into the drop area or uses **Choose image**. Analysis does not begin automatically. The app decodes the image in the worker, validates it, displays its filename, dimensions, mode, and SHA256, and shows a preview.
4. **Confirm the input.** The user checks the preview and clicks **Run frozen analysis**. The button text names the action; it must not say “Diagnose,” “Scan for cancer,” or “Confirm tumor.”
5. **Wait through a truthful pipeline display.** The app shows named stages rather than a fabricated percentage: integrity check, input validation, frozen preprocessing, OOD evidence, OOD policy, category output, E08 ensemble, C03 mapping, R02 reliability, and result assembly.
6. **Handle the OOD gate.** If the OOD score is at or above `0.36854275476100085`, an amber gate explains that the input differs from the registered development distribution. REFUSE is disabled, so the output remains available for inspection, but its final display state is `REVIEW_REQUIRED`. The gate states that OOD is a source-shift signal and is weak at ranking the correctness of an individual mask.
7. **Inspect the result.** The user sees the final display status first, then the OOD result, model-predicted category and four raw class probabilities, the original image, probability overlay, hard mask, R02 score when applicable, pixel-only mask measurements, and a collapsible provenance section.
8. **Compare views.** The user can switch among Original, Probability, Mask overlay, Contour, and Side-by-side. Zoom, pan, reset, and opacity controls remain available without changing underlying data.
9. **Optionally edit the mask.** The editor creates a separate human-edited layer. It never overwrites the frozen prediction. Once edited, the app labels the case `USER EDITED`, freezes the original model reliability score, and says that the score applies to the original model mask only.
10. **Optionally save locally.** Retention is off by default. The user may select **Save to local history**, which writes a versioned case bundle under the current Windows user profile. The dialog states exactly which source and derived files will be retained.
11. **Optionally export a PDF.** The user previews a fixed-layout research summary and chooses a local path. Every page contains the disclaimer. Printing uses the normal local system print dialog after preview.
12. **Start another case or close.** Unsaved source data, intermediate tensors, and temporary renders are discarded when the case is cleared or the app closes.

### 2.2 Result policy

For a tumor-category prediction, the display rule is:

```text
OOD_ACCEPT = ood_score < 0.36854275476100085
RELIABILITY_STANDARD = reliability_score >= 0.9490250173235835

if OOD_ACCEPT and RELIABILITY_STANDARD:
    final_display_status = STANDARD
else:
    final_display_status = REVIEW_REQUIRED
```

The segmentation threshold remains exactly `0.50`. Equality at the reliability threshold belongs to `STANDARD`. REFUSE remains disabled.

A predicted `no_tumor` label has no tumor mask and therefore no valid R02 segmentation-reliability target. My recommendation is to show it as a classification-only output and fail closed to `REVIEW_REQUIRED / SEGMENTATION_RELIABILITY_NOT_APPLICABLE`. This adds an application reason code rather than a new score or threshold, but it should be registered before implementation because the current P03 message catalogue does not define this path. The UI must say **Model-predicted label: no_tumor**; it must never say **No tumor detected** as a clinical conclusion.

## 3. What the user sees

### 3.1 Information hierarchy

The interface follows this order on every result screen:

1. **Final display disposition:** `STANDARD` or `REVIEW_REQUIRED`.
2. **Reason:** OOD review, reliability below threshold, invalid/unavailable score, classification-only no-tumor path, or integrity/error state.
3. **OOD evidence:** score, threshold, `ACCEPT` or `REVIEW`, plus its narrow interpretation.
4. **Model-predicted BRISC category:** label and raw four-class probabilities.
5. **Spatial output:** probability map, model hard mask, and pixel-only mask statistics.
6. **Internal segmentation reliability:** R02 score and threshold, explicitly tied to the original model mask.
7. **Provenance:** stack version, model components, hashes, execution device, input hash, timestamps, and processing duration.
8. **Research disclaimer:** persistent in the app footer and repeated in exports.

This order prevents a large class label from visually overriding the OOD or review decision. The final status is the largest element; the class label is deliberately secondary.

### 3.2 Application states

The state machine is explicit:

- `BOOT_VERIFYING`
- `BOOT_BLOCKED`
- `READY_EMPTY`
- `INPUT_VALIDATING`
- `READY_TO_ANALYZE`
- `ANALYZING`
- `OOD_REVIEW_GATE`
- `RESULT_STANDARD`
- `RESULT_REVIEW_REQUIRED`
- `EDITING_MASK`
- `EXPORT_PREVIEW`
- `HISTORY`
- `RECOVERABLE_ERROR`
- `INTEGRITY_ERROR_LOCKED`

The user never sees an uninitialized panel, stale output from a previous case, or a green status while the current image is still running.

### 3.3 Panels

- **Navigation rail:** Analyze, History, About/Integrity.
- **Header:** current case name, privacy indicator, stack-integrity chip, device mode.
- **Image workspace:** original image and overlays, zoom/pan, view tabs, opacity.
- **Result inspector:** final status, OOD, category probabilities, reliability, mask statistics, provenance.
- **Action bar:** analyze, clear, edit mask, save locally, export PDF.
- **Persistent footer:** disclaimer and “No network · Local processing.”
- **Notification region:** non-modal notices for save success, CPU fallback, or local storage warnings.

## 4. Technology recommendation

### 4.1 Desktop framework: PySide6 with Qt Widgets

I recommend a native PySide6 desktop application built with Qt Widgets. PySide6 is Qt’s official Python binding, so it keeps the existing Python/PyTorch stack in one language while providing a mature windowing, accessibility, model-view, printing, and graphics system. [Qt for Python documentation](https://doc.qt.io/qtforpython-6/)

The central image surface should use `QGraphicsScene` and `QGraphicsView`. They provide a scrollable two-dimensional scene with transformations and coordinate mapping, which suits aligned image, probability, contour, and edit layers. [Qt Graphics View documentation](https://doc.qt.io/qtforpython-6/overviews/qtwidgets-graphicsview.html)

I would not use Streamlit. Its browser-dashboard interaction model makes precise mask editing, keyboard control, process isolation, PDF preview, and a disciplined state machine harder. I would not use Electron because it adds a large JavaScript runtime and a Python bridge without improving this local single-user workflow. A native Qt app is the most direct route to a serious research instrument.

### 4.2 Inference backend: one long-lived spawned worker process

The GUI process must not load or mutate model artifacts. A long-lived Python worker process should:

- load `runs/final_stack/` through the strict frozen loader;
- verify all hashes before accepting work;
- keep neural models resident in GPU or CPU memory;
- accept one typed job at a time over a local `multiprocessing` pipe or queue;
- return immutable result records and rendered arrays;
- expose no TCP/HTTP server;
- restart cleanly after a crash or cancelled job.

Python’s process-based concurrency works on Windows using the `spawn` start method, and a separate process prevents a long PyTorch operation from freezing the Qt event loop. [Python multiprocessing documentation](https://docs.python.org/3.12/library/multiprocessing.html)

The UI may use Qt threads for small file-hash or rendering tasks, but CUDA/PyTorch inference belongs in the persistent worker. Only one analysis job is allowed at a time on this laptop.

### 4.3 Important OOD-order constraint

I disagree with a literal interpretation of “OOD detection runs before classification.” The frozen O05 feature order includes classifier uncertainty and segmenter entropy. Therefore the registered O05 score cannot be computed without internally executing the classifier and a segmenter forward pass. Replacing it with an O02/O03-only precheck would create a new detector and require a new registered threshold.

The correct Phase 6 interpretation is **OOD before classification disclosure and policy use**:

1. preprocess the input;
2. silently compute the classifier and v1 segmenter evidence required by O05;
3. calculate and apply O05;
4. do not display or persist the category or mask before the OOD decision;
5. if OOD reviews the case, retain that review disposition while completing the remaining E08 components for inspection because REFUSE is disabled;
6. release the complete result to the UI with OOD shown first.

If “before classification” means no classifier forward pass may execute, it conflicts with the frozen O05 artifact and cannot be satisfied without a new preregistered OOD experiment. The application must document this distinction rather than pretending otherwise.

### 4.4 Domain layer

Use small typed Python modules, independent of Qt, for:

- manifest verification;
- input validation and preprocessing contract;
- frozen inference orchestration;
- OOD and reliability policy decisions;
- result schema validation;
- mask-edit provenance;
- history repository;
- PDF report model.

Transport records should be versioned dataclasses containing only plain strings, numbers, lists, and array references. Thresholds remain canonical decimal strings in persisted results. The policy compares the computed score against the value parsed from the frozen manifest; no copied or rounded UI constant is allowed.

### 4.5 Local storage

Use Python’s built-in SQLite interface for a single local database at:

```text
%LOCALAPPDATA%\GyrusResearch\app.db
```

SQLite is sufficient for one user and avoids a background database service. Python 3.12 includes the DB-API `sqlite3` interface. [Python standard-library documentation](https://docs.python.org/3.12/library/)

Retention is opt-in. A saved case uses:

```text
%LOCALAPPDATA%\GyrusResearch\cases\<case_uuid>\
    source.<ext>
    result.json
    model_mask.png
    edited_mask.png        # only if edited
    overlay_preview.png
    report.pdf             # only if exported into the case
    provenance.json
```

The database stores the case ID, timestamps, filename, hashes, predicted category, OOD state, final status, edit flag, and bundle path. It stores no patient name, date of birth, hospital number, or diagnostic note because Phase 6 does not need them.

Writes must be transactional and atomic. Whole-file artifacts should use `QSaveFile` or an equivalent temporary-write-and-rename pattern so an interrupted save never leaves a partially written final document. [Qt `QSaveFile` documentation](https://doc.qt.io/qtforpython-6.10/PySide6/QtCore/QSaveFile.html)

The application will not claim encryption. Local Windows account controls and NTFS permissions are the Phase 6 boundary. Encrypted storage is deferred until its recovery and key-management behavior can be designed properly.

### 4.6 PDF generation and preview

Generate a deterministic, fixed-layout PDF locally with Qt’s painting/PDF facilities. Preview it inside the application with `QPdfView`, which supports single-page and multipage display. [Qt PDF viewer documentation](https://doc.qt.io/qtforpython-6/PySide6/QtPdfWidgets/QPdfView.html)

The PDF generator receives a complete immutable report model. It must not query widgets for values. This separates visual state from exported evidence and makes the report testable.

### 4.7 Testing

The Phase 6 test suite should include:

- **Unit tests:** exact threshold boundaries, status precedence, invalid score handling, no-tumor path, mask statistics, edit provenance, schema migrations.
- **Frozen-stack integration tests:** all three Stage 8 fixtures, artifact hashes, O03/O05 reproduction, GPU and CPU hard-mask agreement, and read-only stack checks before and after a session.
- **UI tests:** `pytest-qt` tests for every state, keyboard navigation, focus order, disabled actions, OOD gate behavior, edit undo/redo, and history filtering.
- **Golden visual tests:** a small set of stable screenshots for empty, standard, review, OOD, and error states at 100% and 150% Windows scaling. These supplement structural tests; they do not replace them.
- **PDF tests:** page count, required disclaimer on every page, exact thresholds in technical details, edited/model mask separation, no diagnostic language, and deterministic rendering metadata.
- **Privacy tests:** monkeypatch socket and DNS creation to fail, assert the app still works, and scan dependencies/application code for telemetry clients.
- **Failure tests:** worker termination, GPU out-of-memory, corrupt image, huge image, disk-full simulation, database lock/corruption, and PDF save interruption.
- **Security-oriented parser tests:** decompression-bomb limits, malformed JPEG/PNG files, path traversal in filenames, unexpected alpha channels, and oversized dimensions.
- **Accessibility tests:** keyboard-only use, screen-reader labels, high-DPI scaling, minimum contrast, and status comprehension without color.

## 5. What appears first

The first useful screen is not a metrics dashboard. It is a calm **New analysis** screen with one primary action and the system’s limits visible before the user loads data.

The upper-right header shows:

- `LOCAL ONLY`
- `FROZEN STACK VERIFIED`
- `GPU: RTX 4060` or `CPU MODE`

The main drop zone says:

> Drop one 2D PNG or JPEG here
> or choose an image from this computer

Below it:

> The application produces research labels and masks. It does not provide a diagnosis. Images are processed locally and are not uploaded.

The right panel shows **Before you begin**:

- one 2D image at a time;
- BRISC-style categories only;
- pixel measurements only;
- external-looking inputs are placed in review;
- history is not saved unless requested.

The footer permanently states **Research prototype — not for clinical use**.

I strongly oppose opening with accuracy cards, a brain illustration, a giant “Detect Tumor” button, or a green shield. Those choices would advertise certainty before the user sees the evidence limits and would make the product feel like a student demo or an unsafe medical claim.

## 6. Five-minute skeptical-reviewer demonstration

The demo uses disclosed, deterministic examples from development-validation or public external cohorts. It never uses the sealed BRISC test set. The selected cases demonstrate states; they are not presented as a performance sample.

### 0:00–0:40 — Integrity and scope

- Launch the app.
- Show manifest verification, local-only indicator, exact frozen thresholds in the provenance drawer, and the permanent disclaimer.
- State the main negative result first: H8 failed; this is an audited research viewer, not a clinical detector.

### 0:40–1:35 — A clear internal `STANDARD` case

- Load a high-overlap BRISC development-validation meningioma or pituitary case.
- Show OOD `ACCEPT`, model-predicted category probabilities, E08 overlay, R02 score, and `STANDARD` status.
- Toggle Original → Probability → Contour to demonstrate registration and explain that `STANDARD` means only that the two internal policy checks passed.

### 1:35–2:30 — A weak internal glioma case

- Load a registered weak or zero-Dice BRISC development-validation glioma case.
- Show `REVIEW_REQUIRED`, weaker glioma context, fragmented or misplaced mask, and the exact policy reason.
- State the Amendment 21 conclusion: small target area is the strongest observed association, partly coupled to Dice, and not a causal explanation.

### 2:30–3:35 — Independent external input

- Load an OpenNeuro pituitary or UPENN-GBM example.
- Show the OOD gate before revealing the result.
- Explain that P06 reviewed every X02 and X03 case, so the policy’s external behavior is total abstention, not successful selective coverage.
- Explain that O07 AUROC near 0.60/0.56 means the OOD score is not a reliable judge of this individual mask.

### 3:35–4:25 — Edit provenance

- Enter the mask editor, add and erase a small region, undo once, then save the edit.
- Show the untouched model mask beside the edited mask.
- Point out that the R02 score remains attached to the original model output and is not recomputed as if the human edit were another model prediction.

### 4:25–5:00 — Export and auditability

- Open the PDF preview.
- Show the repeated disclaimer, status reason, original and edited masks, exact thresholds, model-stack version, and SHA256 provenance.
- End with the limitation screen, not the strongest result screen.

## 7. Scope and deferrals

### 7.1 In scope for Phase 6

- Windows desktop application shell.
- Read-only frozen-stack loader and startup self-test.
- Single PNG/JPEG upload with explicit analysis action.
- OOD-first disclosure policy using frozen O05/O06.
- Four-class BRISC label display with raw probabilities.
- E08 probability map and mask for tumor-category predictions.
- C03 calibrated probability display.
- R02 reliability and final two-state display policy.
- Dedicated no-tumor fail-closed application rule after registration.
- Image viewer with overlay, contour, zoom, pan, reset, and opacity.
- Non-destructive mask editor with undo/redo and provenance.
- Pixel-only mask area, bounding box, centroid, and component count.
- Opt-in local history.
- PDF preview, local export, and local printing.
- Integrity, privacy, accessibility, and failure-handling tests.
- Developer-mode and packaged-mode local execution on the target laptop.

### 7.2 Deferred

- DICOM, NIfTI, multi-slice, and 3D volume import.
- Batch analysis and folder watching.
- PACS, DICOM networking, hospital integration, or cloud APIs.
- Patient identifiers, clinical notes, appointments, or user accounts.
- Millimetre measurements without verified spacing.
- Tumor cause, danger, prognosis, depth below skull, organ failure, or treatment guidance.
- Automatic tumor grading or additional tumor categories.
- Neural retraining, external calibration, threshold tuning, or online learning.
- Multiple simultaneous users or remote access.
- Encrypted case vault and enterprise key management.
- Digital signing, FDA/CE workflows, audit certification, or clinical deployment.
- Automatic software/model updates. Updating the frozen stack must remain an explicit, separately verified release action.

PNG/JPEG-only import is intentional. Adding a DICOM badge to look professional would be the wrong choice when DICOM windowing, orientation, multi-frame behavior, and metadata handling have not been registered and validated for this application.

## 8. What could go wrong

### 8.1 Startup and integrity

- Missing, modified, or unexpected final-stack artifact.
- Manifest schema mismatch.
- Incompatible PyTorch/CUDA/runtime environment.
- Frozen fixture divergence.
- Read permission failure.

These produce `INTEGRITY_ERROR_LOCKED`. Analysis is disabled. The app shows the mismatched path and expected/actual hash, with **Copy technical details** and **Close application**. It never offers “continue anyway.”

### 8.2 Input problems

- Unsupported type, corrupt file, truncated image, animated/multi-frame input.
- Image exceeds decoded-pixel or file-size limits.
- Zero-size, constant, nearly blank, alpha-only, CMYK, palette, or unexpected bit-depth image.
- Orientation metadata changes visual interpretation.
- Filename contains control characters or path tricks.
- Source file changes between preview and analysis.

Validation records the file hash before preview and rechecks it immediately before inference. A changed file is rejected and must be selected again.

### 8.3 Runtime failures

- GPU unavailable, driver reset, CUDA out-of-memory, worker crash, or timeout.
- CPU fallback produces a tolerance or mask mismatch.
- User cancels during inference.
- Multiple Run clicks.

Only one job runs. Cancel discards all partial output and restarts the worker if required. GPU failure offers a clearly labeled CPU retry; it does not silently change execution mode. If CPU verification fails, the case ends in error.

### 8.4 Model and policy edge cases

- OOD score invalid or unavailable.
- Reliability score `None`, NaN, infinite, outside `[0,1]`, or exactly on the threshold.
- Empty mask, one-pixel mask, multiple components, mask touching the border.
- Predicted `no_tumor`, where segmentation reliability is inapplicable.
- Predicted glioma, where internal subgroup performance is weaker.
- OOD review with high reliability, or OOD accept with low reliability.

Invalid OOD or reliability fails closed to review. Final status uses the most restrictive applicable reason. OOD and reliability remain separate rows so the user can see disagreement. No weighted “overall confidence” is invented.

### 8.5 Editing problems

- Brush stroke outside image bounds.
- Accidental large erase.
- Undo history exhaustion.
- Edit saved while result export is in progress.
- Edited mask mistaken for model output.

The editor clips strokes, limits brush size, supports undo/redo, and requires **Save edited layer**. Model and edited masks use different colors and permanent labels. Export serializes from an immutable snapshot so it cannot mix two edit versions.

### 8.6 Storage and export

- Disk full, invalid destination, locked file, database lock, corrupt history index.
- Source image removed after a path-only session.
- PDF renderer failure or printer cancellation.
- User believes deleting a history row guarantees physical erasure from an SSD.

Case bundles use atomic writes and are added to SQLite only after every required file commits. History deletion removes application-managed files and records but does not promise forensic secure erasure.

## 9. Feedback requested before implementation

These are the areas where I want explicit review:

1. **OOD sequencing language.** The frozen O05 dependency makes literal pre-classifier execution impossible. The proposed rule is OOD before classification disclosure. This needs acceptance in the Phase 6 protocol.
2. **No-tumor policy.** R02 has no valid target for a classification-only no-tumor result. I recommend mandatory review, but the reason code and copy need registration.
3. **History retention.** I recommend opt-in history with a local source copy. Feedback should decide whether all history should instead be metadata-only or disabled entirely.
4. **Mask editing.** It is useful for inspection and demonstration, but could imply an annotation/clinical workflow. The copy and export labels must keep its role narrow.
5. **Terminology.** “Model-predicted category,” “internal reliability,” “review required,” and “OOD source-shift signal” should be tested with nontechnical users to ensure they do not read them as diagnosis, confidence, or safety.
6. **Full-precision threshold display.** Machine-readable values must remain exact. The main UI can show a shorter score for readability only if the details panel and export preserve the full value and the status is always computed from the manifest value.
7. **CPU fallback.** The desired tolerance, expected latency, and whether CPU results can be exported require a registered acceptance rule.

## 10. Complete UI and interface proposal

### 10.1 Aesthetic direction

The visual direction is **quiet scientific instrument**: dark neutral surfaces around MRI imagery, compact typography, restrained blue interaction accents, cyan model overlays, and amber review states. It should resemble a serious image-analysis workstation, not a hospital product and not a futuristic AI dashboard.

The app is dark by default because black image surroundings preserve MRI contrast and reduce glare. There are no gradients, glass effects, animated brains, pulsing AI icons, decorative charts, or accuracy speedometers. Motion is limited to a 160 ms panel transition and a small indeterminate activity indicator.

Status is never communicated by color alone. Every state uses an icon, uppercase label, reason, and border treatment.

### 10.2 Window and layout

**Reference canvas:** 1440 × 900 logical pixels at 100% Windows scaling.
**Minimum window:** 1180 × 720.
**Maximum:** normal desktop maximize behavior.
**High DPI:** layouts use logical pixels and are verified at 100%, 125%, 150%, and 200% scaling.

At 1440 × 900:

| Region | Position/size | Purpose |
|---|---:|---|
| Navigation rail | x 0, y 0, 72 × 900 | Analyze, History, About/Integrity |
| Header | x 72, y 0, 1368 × 64 | Case, privacy, integrity, device |
| Content margin | 24 px | Breathing room around working panels |
| Image panel | x 96, y 88, 820 × 748 | Image, overlays, view controls |
| Main gutter | 20 px | Separation |
| Inspector | x 936, y 88, 480 × 748 | Status and result details |
| Footer | x 72, y 868, 1368 × 32 | Disclaimer and local-only status |

The image panel contains a 48 px view-tab header, a 636 px canvas, and a 64 px action strip. The inspector scrolls vertically while its final-status card remains pinned. Below 1280 px width, the inspector becomes a 412 px right dock and the image panel shrinks to 640 px. It never overlays the image.

### 10.3 Color palette

| Token | Hex | Use |
|---|---|---|
| App background | `#0B1016` | Main background |
| Navigation | `#0E151D` | Left rail |
| Surface | `#151D27` | Panels/cards |
| Elevated surface | `#1B2633` | Selected/raised content |
| Border | `#2A3848` | Dividers and card outlines |
| Primary text | `#F3F7FB` | Titles and values |
| Secondary text | `#A7B3C2` | Body and labels |
| Muted text | `#748296` | Metadata |
| Primary action | `#4EA8DE` | Primary buttons and links |
| Focus ring | `#7CC7F5` | Keyboard focus |
| Standard | `#2FC38C` | `STANDARD`, with text/icon |
| Review | `#F2B84B` | `REVIEW_REQUIRED` and OOD review |
| Error | `#F06464` | Integrity/runtime error |
| Predicted mask | `#35D0E0` | 32% fill, 100% contour |
| Probability map | `#2A6FDB`→`#35D0E0`→`#F2B84B` | Perceptual low-to-high scale |
| Manual addition | `#A78BFA` | Edited-mask added pixels |
| Manual removal | `#FF6B8A` | Cross-hatched removed pixels |
| Image canvas | `#05070A` | MRI background |
| PDF ink | `#17202A` | White PDF body text |

Standard green and review amber are reserved for status. Class probabilities never use status colors.

### 10.4 Typography

Use **Segoe UI Variable Text** and **Segoe UI Variable Display**, falling back to **Segoe UI**. They are native to the target Windows environment and avoid web-font loading. Use **Cascadia Mono** for hashes, thresholds, paths, and technical values.

| Style | Size/line | Weight |
|---|---:|---:|
| Window title | 24/32 px | 600 |
| Result status | 20/28 px | 700 |
| Section heading | 16/24 px | 600 |
| Card title | 13/20 px | 600 |
| Body | 13/20 px | 400 |
| Button | 13/18 px | 600 |
| Metadata | 11/16 px | 400 |
| Technical mono | 11/17 px | 400 |
| PDF title | 20 pt | 700 |
| PDF body | 9.5 pt | 400 |

No body text is smaller than 11 logical pixels in the desktop UI. Text can scale with the operating-system accessibility setting without clipping.

### 10.5 Buttons, badges, and visual language

- **Primary button:** 40 px high, 12–20 px horizontal padding, 6 px radius, `#4EA8DE` fill, dark text. One primary action per view.
- **Secondary button:** 40 px high, transparent/elevated fill, 1 px `#2A3848` border.
- **Destructive button:** outlined `#F06464`; solid red only in the final confirmation dialog.
- **Toolbar button:** 36 × 36 px, icon plus tooltip, visible selected state.
- **Status badge:** 28 px high, 4 px radius, icon + uppercase label; never a vague colored dot.
- **Focus:** 2 px `#7CC7F5` outer ring and keyboard-visible focus state.
- **Cards:** 8 px radius, 1 px border, 16 px internal padding. Avoid floating shadows; separation comes from surface and border.
- **Icons:** simple 18/20 px outline icons with text labels for important actions.

### 10.6 Empty state

```text
┌──────┬──────────────────────────────────────────────────────────────────────────────┐
│  G   │ Gyrus — Brain MRI Research Viewer  LOCAL ONLY  ✓ VERIFIED   GPU: RTX 4060   │ 64
├──────┼───────────────────────────────────────────────┬──────────────────────────────┤
│  ●   │                                               │ Before you begin             │
│Analyze│       ┌───────────────────────────────┐       │                              │
│      │       │       DROP ONE IMAGE          │       │ • One 2D PNG/JPEG           │
│  ◷   │       │                               │       │ • BRISC labels only         │
│History│      │       [ Choose image ]        │       │ • Pixel measurements only  │
│      │       │                               │       │ • External inputs → review │
│  i   │       │ Processed locally.            │       │ • History is opt-in         │
│About │       └───────────────────────────────┘       │                              │
│      │                                               │ Frozen manifest              │
│      │  No file selected                             │ 7e67de9f16b1…                │
├──────┴───────────────────────────────────────────────┴──────────────────────────────┤
│ Research prototype — not for clinical use                    No network · Local only │ 32
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.7 Loading state

The canvas retains the validated source image. The inspector becomes a stage log. Completed steps have check marks, the active step has a spinner, and future steps are muted. There is no fake “73%.”

```text
┌──────┬───────────────────────────────────────────────┬──────────────────────────────┐
│ rail │ Case: IMG_00421.jpg                           │ ANALYSIS IN PROGRESS         │
├──────┼───────────────────────────────────────────────┼──────────────────────────────┤
│      │                                               │ ✓ Input validated            │
│      │                SOURCE IMAGE                   │ ✓ Frozen preprocessing       │
│      │                                               │ ◌ Computing OOD evidence     │
│      │                    MRI                        │ · Applying OOD policy        │
│      │                                               │ · Preparing category output │
│      │                                               │ · E08 ensemble               │
│      │                                               │ · C03 + R02                  │
│      │                                               │                              │
│      │ [Fit] [100%]                                  │ [ Cancel analysis ]          │
├──────┴───────────────────────────────────────────────┴──────────────────────────────┤
│ Research prototype — not for clinical use                    No network · Local only │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.8 OOD detected state

OOD uses amber, never red. Red is reserved for system failure. The result is not refused because REFUSE is disabled. The dialog is a disclosure gate with two actions: **View research output** and **Choose another image**.

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ ⚠ INPUT OUTSIDE REGISTERED DEVELOPMENT DISTRIBUTION                                │
│                                                                                     │
│ OOD score          0.9821                                                           │
│ Review threshold   0.36854275476100085                                              │
│ OOD policy         REVIEW                                                           │
│                                                                                     │
│ This input differs from the BRISC development distribution. The resulting output   │
│ is placed in REVIEW_REQUIRED. The OOD score detects source shift; it is weak at      │
│ determining whether this individual mask is correct.                                │
│                                                                                     │
│ REFUSE is disabled. The research output remains available for inspection.           │
│                                                                                     │
│ [ Choose another image ]                             [ View research output ]        │
│                                                                                     │
│ Research prototype — not for clinical use                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.9 `STANDARD` result

```text
┌──────┬───────────────────────────────────────────────┬──────────────────────────────┐
│ rail │ Case: IMG_00421.jpg   LOCAL ONLY  ✓ VERIFIED  │ ✓ STANDARD                  │
├──────┼───────────────────────────────────────────────┼──────────────────────────────┤
│      │ Original | Probability | Overlay | Contour    │ OOD                          │
│      │                                               │ ACCEPT   0.0842              │
│      │          MRI + CYAN MODEL MASK                │ threshold 0.36854275476100085│
│      │                                               ├──────────────────────────────┤
│      │                                               │ MODEL-PREDICTED CATEGORY     │
│      │                                               │ Meningioma                   │
│      │                                               │ glio .02  meni .94           │
│      │                                               │ pitu .03  no_tumor .01       │
│      │                                               ├──────────────────────────────┤
│      │                                               │ INTERNAL MASK RELIABILITY    │
│      │ [−] 100% [+] [Fit]  Opacity ━━━●━━           │ 0.9736   ✓ STANDARD          │
│      │                                               │ threshold 0.9490250173235835 │
├──────┼───────────────────────────────────────────────┼──────────────────────────────┤
│      │ [Clear] [Edit mask] [Save locally]            │ Mask area 284 px             │
│      │                            [Export PDF]        │ Components 1  [Provenance ▾] │
├──────┴───────────────────────────────────────────────┴──────────────────────────────┤
│ Research prototype — not for clinical use                    No network · Local only │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

The wording below the status reads: “This output met the internal research policy. Review the image, category label, and highlighted region.” It never says trusted, safe, accurate, or confirmed.

### 10.10 `REVIEW_REQUIRED` result

```text
┌──────┬───────────────────────────────────────────────┬──────────────────────────────┐
│ rail │ Case: GLIOMA_071.jpg                          │ ⚠ REVIEW_REQUIRED            │
├──────┼───────────────────────────────────────────────┼──────────────────────────────┤
│      │ Original | Probability | Overlay | Contour    │ Reason                       │
│      │                                               │ RELIABILITY BELOW THRESHOLD  │
│      │          MRI + CYAN MODEL MASK                ├──────────────────────────────┤
│      │                                               │ OOD                          │
│      │                                               │ ACCEPT   0.1410              │
│      │                                               ├──────────────────────────────┤
│      │                                               │ MODEL-PREDICTED CATEGORY     │
│      │                                               │ Glioma                       │
│      │                                               │ Glioma ranking was weaker in │
│      │                                               │ internal development tests.  │
│      │ [−] 100% [+] [Fit]  Opacity ━━━●━━           ├──────────────────────────────┤
│      │                                               │ Reliability 0.6124           │
├──────┼───────────────────────────────────────────────┼ threshold 0.9490250173235835 ┤
│      │ [Clear] [Edit mask] [Save locally] [Export]   │ [Why review is required]     │
├──────┴───────────────────────────────────────────────┴──────────────────────────────┤
│ Research prototype — not for clinical use                    No network · Local only │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

The model output stays visible because review requires inspection. Review styling surrounds the inspector and header; it does not tint the MRI or obscure the mask.

### 10.11 Mask editor after an edit

The editor is a mode inside the image panel, not a separate unconnected window. A 56 px vertical tool strip sits inside the left edge of the canvas. The status inspector remains visible.

```text
┌──────┬───────────────────────────────────────────────┬──────────────────────────────┐
│ rail │ Mask editor · Case GLIOMA_071.jpg             │ ⚠ REVIEW_REQUIRED            │
├──────┼───┬───────────────────────────────────────────┼──────────────────────────────┤
│      │ + │ Model | Edited | Difference | Split       │ USER EDITED                  │
│      │ − │                                           │ Original model status kept.  │
│      │ ○ │      CYAN = MODEL MASK                    │                              │
│      │ ↶ │      VIOLET = HUMAN ADDITION              │ Reliability 0.6124           │
│      │ ↷ │      PINK HATCH = HUMAN REMOVAL           │ Applies to original mask only│
│      │   │                                           ├──────────────────────────────┤
│      │   │                                           │ Edited mask                  │
│      │   │                                           │ Area 312 px (+28)            │
│      │   │                                           │ Components 1                │
│      │   │ Brush 8 px  [−] 200% [+]                  │                              │
├──────┴───┴───────────────────────────────────────────┼──────────────────────────────┤
│ [Discard edits] [Reset to model]                     │ [Save edited layer]          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Tools are Add, Erase, Brush size, Undo, Redo, and Reset. There is no automatic smoothing because it would be an unregistered transformation. Pixel counts update live and remain labeled as mask pixels, not tumor dimensions.

### 10.12 History view

History contains only cases explicitly saved by the user. The default empty history explains that retention is off. At 1440 px, the list is 860 px wide and the preview is 440 px, separated by 20 px.

```text
┌──────┬──────────────────────────────────────────────────────────────────────────────┐
│ rail │ Local history                     [Search…] [Status ▾] [Clear history…]      │
├──────┼──────────────────────────────────────────────┬───────────────────────────────┤
│      │ DATE       FILE        CATEGORY    STATUS    │ Selected case                 │
│      │ 09/30 18:2 GLIOMA_071  glioma      REVIEW    │ [small overlay preview]       │
│      │ 09/30 17:5 IMG_00421   meningioma  STANDARD  │                               │
│      │ 09/29 15:1 sub-012     pituitary   REVIEW    │ OOD REVIEW                    │
│      │                                              │ Reliability 0.6124            │
│      │ ● source saved locally   ✎ edited            │ Saved source: Yes             │
│      │                                              │ Edited: Yes                  │
│      │                                              │ [Open] [Export] [Delete]     │
├──────┴──────────────────────────────────────────────┴───────────────────────────────┤
│ Research prototype — not for clinical use       History is stored only on this PC  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

History sorting is newest first. Filters cover final status, category, OOD decision, edit state, and date. There is no patient-name field.

### 10.13 PDF export preview

The preview is a 1160 × 780 modal: 760 px PDF viewport, 16 px gutter, 336 px export summary, and 24 px margins.

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Export research summary                                               [×]           │
├───────────────────────────────────────────────────┬─────────────────────────────────┤
│                                                   │ 3 pages                         │
│         A4 PAGE PREVIEW                           │                                 │
│  ┌─────────────────────────────────────────────┐  │ Includes                        │
│  │ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE  │  │ ✓ Source and overlay           │
│  │ Case summary            REVIEW_REQUIRED     │  │ ✓ Model + edited mask          │
│  │                                             │  │ ✓ OOD/reliability reasons      │
│  │ [original]              [overlay]            │  │ ✓ Thresholds and hashes        │
│  │                                             │  │ ✓ Limitations/disclaimer       │
│  │ Model-predicted category: glioma            │  │                                 │
│  │ Reliability applies to original mask only   │  │ Destination                     │
│  │                                             │  │ C:\…\case-summary.pdf          │
│  │ footer disclaimer · page 1 of 3             │  │                                 │
│  └─────────────────────────────────────────────┘  │ [Choose…]                      │
│                                                   │ [Cancel] [Save PDF]            │
├───────────────────────────────────────────────────┴─────────────────────────────────┤
│ Research prototype — not for clinical use                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Page 1:** case status, reason, source metadata, category probabilities, original and overlay.
**Page 2:** probability map, model mask, edited mask/difference if present, pixel-only measurements.
**Page 3:** OOD and reliability details, exact thresholds, stack and input hashes, execution environment, limitations.

Every page has a top banner and footer disclaimer. `REVIEW_REQUIRED` appears in the page header when applicable. The PDF contains no editable “diagnosis” field.

### 10.14 Error states

Recoverable input errors keep the app usable. Integrity errors lock analysis.

```text
┌──────┬───────────────────────────────────────────────┬──────────────────────────────┐
│ rail │ Analysis unavailable                          │ ✕ INTEGRITY ERROR            │
├──────┼───────────────────────────────────────────────┼──────────────────────────────┤
│      │                                               │ Frozen artifact mismatch     │
│      │      THE FROZEN STACK DID NOT VERIFY          │                              │
│      │                                               │ File                         │
│      │      Analysis has been disabled.              │ models/segmenter_e06_…pt     │
│      │                                               │ Expected SHA256              │
│      │      No result was produced.                  │ b98afabc…                    │
│      │                                               │ Actual SHA256                │
│      │                                               │ 2fd91c0a…                    │
│      │                                               │                              │
│      │                                               │ [Copy technical details]     │
│      │                                               │ [Close application]          │
├──────┴───────────────────────────────────────────────┴──────────────────────────────┤
│ Research prototype — not for clinical use                    No network · Local only │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Recoverable errors use the same layout but offer **Choose another image** or **Retry on CPU** where valid. No error dialog exposes a traceback by default; technical details are expandable and copyable.

### 10.15 Professional-quality details

The features that make this feel like a real research tool are restraint and traceability:

- startup artifact verification before the upload control is enabled;
- an explicit state machine with no stale-result flashes;
- exact provenance and input hashing;
- OOD and reliability shown separately;
- result wording drawn from the frozen message catalogue;
- original and edited masks preserved as separate artifacts;
- no retention by default;
- atomic local saves;
- deterministic PDF structure;
- keyboard navigation and high-DPI support;
- structured error codes and copyable diagnostics;
- no silent CPU/GPU fallback;
- no network stack, telemetry, login, ads, update prompt, or cloud branding;
- no claims that a green status means clinically normal, correct, or safe.

The interface should look finished because its hierarchy is consistent, not because it adds visual spectacle. The most professional behavior is showing uncertainty, provenance, and failure without hiding the underlying output.
