"""PyInstaller-safe entry point for the local Gyrus desktop application."""

from __future__ import annotations

from multiprocessing import freeze_support

if __name__ == "__main__":
    # PyInstaller's spawned worker must exit its boot path before Qt starts.
    freeze_support()
    from gyrus.gui.app import main

    raise SystemExit(main())
