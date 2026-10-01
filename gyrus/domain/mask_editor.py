"""Pixel-exact manual mask editing, separate from the frozen model output."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .mask_provenance import (
    MaskEditProvenance,
    MaskEditTool,
    begin_mask_edit,
    mask_sha256,
    record_mask_edit,
)


class MaskEditorError(ValueError):
    """An edit or export cannot be verified."""


class MaskEditIntegrityError(MaskEditorError):
    """The original mask or provenance chain changed unexpectedly."""


class MaskEditSession:
    """Keep model, working, and saved masks distinct; record every state change."""

    def __init__(self, case_id: str, model_mask: np.ndarray, reliability_score: float | None):
        value = np.asarray(model_mask)
        if value.shape != (128, 128) or value.dtype != np.uint8 or not np.isin(value, (0, 1)).all():
            raise MaskEditorError("Expected a binary 128×128 model mask")
        self.model = value.copy().astype(bool)
        self.model.setflags(write=False)
        self.current = self.model.copy()
        self.provenance = begin_mask_edit(case_id, self.model)
        self.reliability_score = reliability_score
        self.saved_mask: np.ndarray | None = None
        self.saved_provenance: MaskEditProvenance | None = None
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []

    @property
    def dirty(self) -> bool:
        return bool(self.provenance.records) and (
            self.saved_provenance is None
            or self.provenance.current_mask_sha256 != self.saved_provenance.current_mask_sha256
            or len(self.provenance.records) != len(self.saved_provenance.records)
        )

    def _verify(self) -> None:
        if mask_sha256(self.model) != self.provenance.model_mask_sha256:
            raise MaskEditIntegrityError("Original model mask changed")
        if mask_sha256(self.current) != self.provenance.current_mask_sha256:
            raise MaskEditIntegrityError("Edit hash chain does not match working mask")
        previous = self.provenance.model_mask_sha256
        for index, record in enumerate(self.provenance.records, 1):
            if record.sequence != index or record.before_sha256 != previous:
                raise MaskEditIntegrityError("Edit hash chain is broken")
            previous = record.after_sha256
        if previous != self.provenance.current_mask_sha256:
            raise MaskEditIntegrityError("Edit chain terminal hash mismatch")

    def _apply(self, after: np.ndarray, tool: MaskEditTool, **parameters: int | str) -> bool:
        self._verify()
        if np.array_equal(after, self.current):
            return False
        next_provenance = record_mask_edit(
            self.provenance, self.current, after, tool=tool,
            timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            parameters=parameters,
        )
        self.current = after.copy()
        self.provenance = next_provenance
        self._verify()
        return True

    def brush(self, points: list[tuple[int, int]], size: int, *, add: bool) -> bool:
        """One stroke is one reversible record; size is diameter in model pixels."""
        if not isinstance(size, int) or not 1 <= size <= 32 or not points:
            raise MaskEditorError("Brush size or stroke invalid")
        if any(not (0 <= x < 128 and 0 <= y < 128) for x, y in points):
            raise MaskEditorError("Stroke outside model space")
        after = self.current.copy()
        radius = size / 2
        for x, y in points:
            left, right = max(0, int(x - radius)), min(127, int(x + radius))
            top, bottom = max(0, int(y - radius)), min(127, int(y + radius))
            yy, xx = np.ogrid[top : bottom + 1, left : right + 1]
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            patch = after[top : bottom + 1, left : right + 1]
            patch[disk] = add
        before = self.current.copy()
        changed = self._apply(after, MaskEditTool.BRUSH_ADD if add else MaskEditTool.BRUSH_REMOVE,
                              size=size, points=len(points))
        if changed:
            self._undo.append(before)
            self._redo.clear()
        return changed

    def undo(self) -> bool:
        if not self._undo:
            return False
        previous = self._undo.pop()
        current = self.current.copy()
        self._apply(previous, MaskEditTool.UNDO)
        self._redo.append(current)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        next_mask = self._redo.pop()
        current = self.current.copy()
        self._apply(next_mask, MaskEditTool.REDO)
        self._undo.append(current)
        return True

    def reset(self) -> bool:
        before = self.current.copy()
        changed = self._apply(self.model, MaskEditTool.RESET_TO_MODEL)
        if changed:
            self._undo.append(before)
            self._redo.clear()
        return changed

    def commit(self) -> None:
        """Save an immutable in-session edit chain, without enabling history."""
        self._verify()
        if not self.provenance.edited:
            raise MaskEditorError("No edited mask to save")
        self.saved_mask = self.current.copy()
        self.saved_mask.setflags(write=False)
        self.saved_provenance = self.provenance

    def discard_since_save(self) -> None:
        """Return to the last committed layer, or discard the unsaved session."""
        if self.saved_mask is None or self.saved_provenance is None:
            raise MaskEditorError("No committed edit layer exists")
        self.current = self.saved_mask.copy()
        self.provenance = self.saved_provenance
        self._undo.clear()
        self._redo.clear()
        self._verify()

    def export(self, parent: str | Path) -> Path:
        """Export only after explicit user action, with no implicit case history."""
        self._verify()
        if not self.provenance.edited:
            raise MaskEditorError("No edited mask to save")
        pending_mask = self.current.copy()
        pending_provenance = self.provenance
        parent = Path(parent).resolve()
        if not parent.is_dir():
            raise MaskEditorError("Export parent directory does not exist")
        if os.name == "nt" and (
            str(parent).startswith("\\\\")
            or ctypes.windll.kernel32.GetDriveTypeW(parent.anchor) not in (2, 3)
        ):
            raise MaskEditorError("Edited masks may only be saved on this computer")
        target = parent / f"gyrus-edit_{self.provenance.case_id}_v{len(self.provenance.records):04d}"
        if target.exists():
            raise MaskEditorError("Edit export already exists; existing evidence was not overwritten")
        temporary = parent / f".gyrus-edit-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            Image.fromarray(pending_mask.astype(np.uint8) * 255, mode="L").save(temporary / "edited_mask.png")
            (temporary / "provenance.json").write_text(
                json.dumps(pending_provenance.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            with Image.open(temporary / "edited_mask.png") as image:
                verified = np.asarray(image.convert("L"), dtype=np.uint8)
            if mask_sha256(verified) != pending_provenance.current_mask_sha256:
                raise MaskEditIntegrityError("Exported mask verification failed")
            os.rename(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.saved_mask = pending_mask
        self.saved_mask.setflags(write=False)
        self.saved_provenance = pending_provenance
        return target
