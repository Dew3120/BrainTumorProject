# Gyrus desktop guide

Gyrus is a local research viewer for one PNG or JPEG brain MRI image at a time. It is a **research prototype, not a clinical tool**, and its output must not be used for patient care. Its independent-cohort segmentation results were weak; a green `STANDARD` label only describes the registered internal display policy, not medical correctness.

## Start the application

For the Windows package, keep the **entire** `dist\Gyrus\` folder together and double-click `dist\Gyrus\Gyrus.exe`. It is a one-folder package; copying only `Gyrus.exe` will not work. This portable build bundles a CPU-only PyTorch runtime and starts on CPU. CUDA is available only from a development environment with the CUDA runtime installed, using `uv run python -m gyrus --device cuda`. Gyrus does not silently change devices when one fails.

For a development checkout, open PowerShell in the project root and run `uv run python -m gyrus`. All analysis stays on the local machine. History is off until you explicitly save a case.

## Analyze an image

1. Wait for the frozen-stack verification to finish. A verification failure blocks analysis.
2. Click **Choose image** or drop one local PNG/JPEG image. Check the filename, dimensions, mode, and SHA256 shown before analysis.
3. Click **Run frozen analysis**. The named stages show progress; they are not a percentage estimate.
4. If the amber input-distribution gate appears, choose another image or explicitly open the research output. The gate means the image differs from development images; it does not measure whether a proposed mask is correct.
5. Read the prominent `STANDARD` or `REVIEW REQUIRED` badge, category and four probabilities, mask view, and plain-language explanation. `REVIEW REQUIRED` means manual inspection is necessary under the registered policy. Even `STANDARD` requires inspection and does not establish clinical validity.
6. Use Original, Probability, Overlay, Contour, or Side-by-side to inspect the result. Zoom, pan, reset, and opacity controls change only the display. Technical provenance is available in its expandable section.

For a model-predicted no-tumor case, there is no tumor mask or mask-reliability score. Gyrus places it in review rather than presenting an unsupported segmentation confidence.

## Edit, retain, and export

The mask editor keeps the original model mask separate from human edits. Add/erase, undo/redo, and reset work on the edited layer; saving creates a hash-chained provenance record. The reliability score always describes the original model mask and is **not recomputed** after a human edit.

**Save to local history** is opt-in per case. The confirmation lists retained files. History can reopen, filter, export, or delete a saved case. Delete removes app-managed files and database records; it is not a promise of forensic erasure.

**Export PDF** opens a three-page preview before a local save. Each page carries the research disclaimer. The report includes the original image, model output, any saved human edit, exact frozen thresholds, and technical provenance. The PDF is a fixed research summary, not a diagnosis. If export fails, the result remains available and no partial PDF should be left at the destination.

## If something fails

Gyrus shows a safe explanation and local log reference instead of a traceback. Use **Copy details** or **Open local log** if you need to report a problem. Logs can contain local paths and should be treated as sensitive. A failed job is never silently retried. On a device error, close and relaunch with an explicit device choice.

The Windows package can verify itself using synthetic fixtures, without opening any BRISC test image:

```powershell
./Gyrus.exe --self-test-report "$env:TEMP\gyrus-self-test.json"
Get-Content -LiteralPath "$env:TEMP\gyrus-self-test.json"
```

A passing self-test proves artifact verification and software execution for those fixtures; it is not evidence of clinical performance.
