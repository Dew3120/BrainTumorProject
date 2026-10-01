"""Safe presentation of failures with private, rotating local diagnostics."""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

_LOGGER = logging.getLogger("gyrus")
_LOG_PATH: Path | None = None


def configure_error_log() -> Path | None:
    """Create a bounded local log; logging failure must not enable analysis."""

    global _LOG_PATH
    if _LOG_PATH is not None:
        return _LOG_PATH
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    root = Path(local).expanduser()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        return None
    if os.name == "nt" and (
        str(root).startswith("\\\\")
        or ctypes.windll.kernel32.GetDriveTypeW(root.anchor) not in (2, 3)
    ):
        return None
    path = root / "GyrusResearch" / "logs" / "gyrus.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        _LOGGER.addHandler(handler)
        _LOGGER.setLevel(logging.INFO)
    except OSError:
        return None
    _LOG_PATH = path
    return path


def error_log_path() -> Path | None:
    return _LOG_PATH


@dataclass(frozen=True, slots=True)
class ErrorNotice:
    code: str
    safe_message: str
    reference: str
    blocking: bool

    @property
    def copy_details(self) -> str:
        location = str(_LOG_PATH) if _LOG_PATH else "Local log unavailable"
        return (
            f"Gyrus error: {self.code}\nReference: {self.reference}\n"
            f"Message: {self.safe_message}\nLog: {location}"
        )


def report_error(
    code: str,
    safe_message: str,
    *,
    exception: BaseException | None = None,
    blocking: bool = False,
) -> ErrorNotice:
    """Log developer context locally, returning only safe UI copy."""

    reference = uuid4().hex[:12]
    configure_error_log()
    if exception is None:
        _LOGGER.error("ref=%s code=%s safe=%s", reference, code, safe_message)
    else:
        _LOGGER.error(
            "ref=%s code=%s safe=%s developer=%s",
            reference, code, safe_message, type(exception).__name__,
            exc_info=(type(exception), exception, exception.__traceback__),
        )
    return ErrorNotice(code, safe_message, reference, blocking)
