# Gyrus Windows release record

Gyrus 0.1.0 is a local, single-user research viewer. The frozen classifier, segmenter, calibrator, reliability model, OOD detector, and registered policy are unchanged. **Research prototype — not for clinical use.** The interface is a way to inspect those outputs and their limitations, not evidence that external segmentation generalizes.

The registered thresholds remain segmentation `0.50`, reliability `0.9490250173235835`, and OOD review `0.36854275476100085` (REFUSE disabled). The policy uses the full-precision values; the UI may round a displayed score for readability.

## Release package

`scripts/build_gyrus_windows.ps1` builds a CPU-only, one-folder Windows package in an isolated Python 3.12 environment. The entry point is `dist\Gyrus\Gyrus.exe`; all neighboring `_internal` files must stay with it. The build copies `runs/final_stack/` byte for byte and verifies its manifest and 31 allowlisted artifacts. The built executable then exercises a spawned worker on two **synthetic software fixtures** with a minimal Windows system PATH. This verifies local execution without depending on the development Python installation. It does not measure medical performance.

The Windows package defaults to CPU for portability. CUDA remains an explicit source/development launch option; this CPU package does not bundle a CUDA runtime. No GPU-to-CPU fallback occurs after failure. No installer or signed binary is supplied. Testing on a separate clean Windows installation without Python is still required before claiming broad machine compatibility.

## Interface and PDF review

- The result inspector leads with the registered display status, a readable explanation, the predicted category, four category-probability bars, image views, and pixel-only mask measurements. Exact thresholds and artifact hashes remain in the expandable technical section.
- The image canvas fits the result when its viewport first becomes visible, so the MRI does not collapse to a tiny thumbnail. Zoom, pan, fit, opacity, and view tabs remain available.
- The OOD gate stays disclosure-first. The result remains withheld until frozen O05 completes; a user must explicitly choose to view a flagged research output.
- History save and PDF render/reverification run off the Qt UI thread. Controls are disabled during those local tasks, and closing waits for the current task to finish. History remains opt-in and the PDF remains a verified, local, three-page report.
- Every PDF page retains the research banner and footer. Page 1 states the result and shows the original and overlay; Page 2 shows spatial evidence and edits where present; Page 3 explains OOD, reliability, exact thresholds, hashes, and limitations in a readable hierarchy. Raw policy identifiers are not the primary copy.

## Reproduce the build and checks

From the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_gyrus_windows.ps1
$env:PYTHONPATH=((Resolve-Path '.').Path + ';' + (Resolve-Path 'scripts').Path)
uv run python -m pytest -q
uv run ruff check .
uv lock --check
git diff --check
```

The build writes `build\gyrus-packaging-self-test.json`. It must say `PASS`, report 31 artifacts, and include the two CPU synthetic cases. Its manifest SHA256 must be `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`.

On 2026-10-03, the clean windowed build passed those checks and parsed a three-page packaged PDF with its disclaimer. The local one-folder package measured **709,854,063 bytes**. The `Gyrus.exe` SHA256 was `ccfd6e13d4fc28162682781b625bf348975291a6e730d60abfa437ddb40162c9`; this hash identifies this local build, not the entire package. The source regression suite had **324 passing tests**; 27 focused GUI/PDF/explanation tests passed after the final copy adjustment. Ruff and the uv lock check passed.

## Release boundaries

- The BRISC sealed test set remains closed. Phase 8 will evaluate it once under a registered device and procedure.
- No neural model was retrained or refit, no registered threshold changed, and `runs/final_stack/` was not modified in this release work.
- The one-folder Windows package is not code signed and has not been tested on an independent fresh Windows PC. Platform, permissions, fonts, display scaling, and antivirus behavior may differ there.
- The packaged self-test uses synthetic fixtures only. It cannot establish accuracy, patient independence, clinical safety, or external transport.
- E08 and its OOD/reliability stack are intentionally complex; the interface does not imply that a single color overlay is a diagnosis. The documented external cohorts showed weak segmentation transfer and total policy abstention.
