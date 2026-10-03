# Build a CPU-only Windows package in an isolated environment. Run from repo root.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$buildPython = Join-Path $projectRoot '.build-venv\Scripts\python.exe'
$buildPyInstaller = Join-Path $projectRoot '.build-venv\Scripts\pyinstaller.exe'

if (-not (Test-Path -LiteralPath $buildPython)) {
    uv venv .build-venv --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated build environment.' }
}
uv pip install --python $buildPython --link-mode copy torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned CPU PyTorch runtime.' }
uv pip install --python $buildPython --link-mode copy numpy==2.5.3 scipy==1.18.1 scikit-image==0.26.0 scikit-learn==1.9.1 monai==1.6.0 reportlab==5.0.1 pypdf==6.19.0 pyside6==6.11.2 pyinstaller==6.22.3
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned application/build packages.' }

$previousPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem;$projectRoot\.build-venv\Scripts"
    & $buildPyInstaller --noconfirm --clean --distpath dist --workpath build packaging\Gyrus.spec
} finally {
    $env:PATH = $previousPath
}
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller could not build Gyrus.' }

& $buildPython -c "from gyrus.domain.manifest import verify_frozen_stack; s=verify_frozen_stack('dist/Gyrus/_internal/runs/final_stack'); print('Verified', len(s.artifacts), 'frozen artifacts', s.manifest_sha256)"
if ($LASTEXITCODE -ne 0) { throw 'The copied frozen package did not verify.' }
$bundleExe = Join-Path $projectRoot 'dist\Gyrus\Gyrus.exe'
$selfTestReport = Join-Path $projectRoot 'build\gyrus-packaging-self-test.json'
if (Test-Path -LiteralPath $selfTestReport) { Remove-Item -LiteralPath $selfTestReport }
$previousPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem"
    $selfTestProcess = Start-Process -FilePath $bundleExe -ArgumentList @(
        '--self-test-report', ('"' + $selfTestReport + '"')
    ) -WindowStyle Hidden -Wait -PassThru
} finally {
    $env:PATH = $previousPath
}
if ($selfTestProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $selfTestReport)) {
    throw 'The packaged CPU worker did not complete its synthetic self-test.'
}
$selfTest = Get-Content -LiteralPath $selfTestReport -Raw | ConvertFrom-Json
if ($selfTest.status -ne 'PASS' -or $selfTest.artifact_count -ne 31 -or $selfTest.cases.Count -ne 2 -or $selfTest.pdf_pages -ne 3) {
    throw 'The packaged CPU worker self-test report is incomplete.'
}
Write-Host 'Gyrus package: dist\Gyrus\Gyrus.exe (keep the entire Gyrus folder together).'
