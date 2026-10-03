"""Run verified local file operations without blocking the Qt event loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class LocalTask(QThread):
    """One bounded UI-owned file operation; errors return to the GUI thread."""

    succeeded = Signal(object)
    failed = Signal(object)

    def __init__(self, operation: Callable[[], Any]) -> None:
        super().__init__()
        self._operation = operation

    def run(self) -> None:
        try:
            result = self._operation()
        except Exception as exc:  # noqa: BLE001 - the UI logs and handles all failures
            self.failed.emit(exc)
        else:
            self.succeeded.emit(result)
