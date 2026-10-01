"""Capture deterministic, model-free Qt shell states for Windows visual regression."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

# Qt's Windows offscreen plugin can render Latin text as missing-glyph boxes.
# Capture against the native Windows platform used by the delivered application.
os.environ["QT_QPA_PLATFORM"] = "windows"
os.environ.setdefault("QT_SCALE_FACTOR", "1")

from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.gui.app import create_application
from gyrus.gui.main_window import GyrusMainWindow


def capture(path: Path, *, width: int, height: int) -> None:
    app = create_application(["Gyrus-golden"])
    window = GyrusMainWindow()
    window.resize(width, height)
    window.show()
    window.on_worker_ready(SimpleNamespace(
        artifact_count=31,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
        device="cpu",
    ))
    app.processEvents()
    image = window.grab().toImage()
    if image.isNull() or not image.save(str(path), "PNG"):
        raise RuntimeError("Qt could not capture the Gyrus shell")
    window.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a model-free Gyrus shell image")
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    if (args.width, args.height) not in {(1440, 900), (1180, 720)}:
        parser.error("only registered shell sizes 1440x900 and 1180x720 are supported")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture(args.output, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
