"""Windows one-folder package; build with the isolated CPU environment.

The frozen stack is copied as data, byte for byte. At runtime its manifest and
all 31 allowlisted artifacts are independently verified before model loading.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


root = Path(SPECPATH).resolve().parent
datas = [
    (str(root / "runs" / "final_stack"), "runs/final_stack"),
    (str(root / "assets" / "logo"), "assets/logo"),
    *collect_data_files("reportlab", includes=["fonts/*.ttf"]),
]

# final_stack_runtime.py is verified data loaded by importlib at runtime, so
# the freezer cannot discover its dependencies through static analysis.
dynamic_runtime_imports = [
    "monai.networks.nets",
    "torchvision.models",
    "scipy.ndimage",
    "scipy.special",
    "scipy.stats",
    "skimage.feature",
    "sklearn.linear_model._logistic",
    "sklearn.preprocessing._data",
]

analysis = Analysis(
    [str(root / "scripts" / "launch_gyrus.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=dynamic_runtime_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["mlflow", "kagglehub", "xgboost", "pandas", "matplotlib", "pydicom", "nibabel"],
    noarchive=False,
)
# The Codex shell's PATH contains Poppler's ICU 78 DLL. Qt6Core on Windows
# imports the unversioned Windows ICU API; Poppler's same-named DLL exports
# only versioned symbols and makes QtCore fail to import (WinError 127).
# Use the OS-provided icuuc.dll. Never ship the incompatible PATH copy.
analysis.binaries = [
    entry for entry in analysis.binaries
    if Path(entry[0]).name.lower() not in {"icuuc.dll", "icudt78.dll"}
]
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Gyrus",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(root / "assets" / "logo" / "gyrus.ico"),
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="Gyrus",
)
