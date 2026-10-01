"""Manual editing must preserve model evidence and produce a verifiable human layer."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from gyrus.domain.mask_editor import MaskEditIntegrityError, MaskEditSession
from gyrus.domain.mask_provenance import mask_sha256
from gyrus.gui.mask_editor import MaskEditor, render_edit_view


def _session() -> MaskEditSession:
    model = np.zeros((128, 128), dtype=np.uint8)
    model[50:60, 50:60] = 1
    return MaskEditSession(str(uuid4()), model, 0.912345)


def test_brush_undo_redo_reset_preserve_model_and_reliability():
    session = _session()
    original_hash = mask_sha256(session.model)
    assert session.brush([(80, 80)], 5, add=True)
    assert session.brush([(55, 55)], 5, add=False)
    assert session.current[80, 80] and not session.current[55, 55]
    assert session.undo() and session.current[55, 55]
    assert session.redo() and not session.current[55, 55]
    assert session.reset() and np.array_equal(session.current, session.model)
    assert session.provenance.edited is False
    assert [r.tool.value for r in session.provenance.records] == [
        "BRUSH_ADD", "BRUSH_REMOVE", "UNDO", "REDO", "RESET_TO_MODEL"
    ]
    assert mask_sha256(session.model) == original_hash
    assert session.reliability_score == 0.912345


def test_save_exports_mask_and_chain_only_after_explicit_action(tmp_path):
    session = _session()
    assert list(tmp_path.iterdir()) == []
    session.brush([(80, 80)], 3, add=True)
    folder = session.export(tmp_path)
    assert folder.parent == tmp_path
    assert session.saved_mask is not None and not session.saved_mask.flags.writeable
    with Image.open(folder / "edited_mask.png") as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    record = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    assert mask_sha256(mask) == record["current_mask_sha256"]
    assert record["model_mask_sha256"] == mask_sha256(session.model)
    assert record["reliability_scope"] == "ORIGINAL_MODEL_MASK_ONLY"
    assert record["records"][0]["parameters"]["size"] == 3
    session.brush([(90, 90)], 3, add=True)
    assert session.dirty
    session.discard_since_save()
    assert not session.dirty
    assert session.current[80, 80] and not session.current[90, 90]


def test_repeat_save_versions_and_collision_does_not_commit(tmp_path):
    session = _session()
    session.brush([(80, 80)], 3, add=True)
    first = session.export(tmp_path)
    session.brush([(90, 90)], 3, add=True)
    second = session.export(tmp_path)
    assert first != second
    assert first.is_dir() and second.is_dir()
    saved_chain_length = len(session.saved_provenance.records)
    session.brush([(100, 100)], 3, add=True)
    collision = tmp_path / f"gyrus-edit_{session.provenance.case_id}_v{len(session.provenance.records):04d}"
    collision.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        session.export(tmp_path)
    assert session.dirty
    assert len(session.saved_provenance.records) == saved_chain_length


def test_edit_hash_mismatch_fails_closed():
    session = _session()
    session.current[0, 0] = True
    with pytest.raises(MaskEditIntegrityError, match="hash chain"):
        session.brush([(70, 70)], 3, add=True)
    with pytest.raises(MaskEditIntegrityError, match="hash chain"):
        session.commit()


def test_difference_colors_and_hatch_are_distinct():
    gray = np.full((128, 128), 30, dtype=np.uint8)
    model = np.zeros((128, 128), dtype=bool)
    model[10:20, 10:20] = True
    edited = model.copy()
    edited[10:20, 10:20] = False
    edited[30:35, 30:35] = True
    rgb = render_edit_view(gray, model, edited, 2)
    assert np.array_equal(rgb[31, 31], (167, 139, 250))
    removal_pixels = {tuple(value) for value in rgb[10:20, 10:20].reshape(-1, 3)}
    assert (255, 107, 138) in removal_pixels
    assert len(removal_pixels) > 1
    assert render_edit_view(gray, model, edited, 3).shape == (128, 256, 3)


def test_invalid_size_and_out_of_bounds_do_not_modify_mask():
    session = _session()
    original_hash = mask_sha256(session.current)
    with pytest.raises(ValueError):
        session.brush([(0, 0)], 0, add=True)
    with pytest.raises(ValueError):
        session.brush([(128, 0)], 3, add=True)
    assert mask_sha256(session.current) == original_hash


def test_qt_brush_maps_click_to_model_space(qtbot):
    widget = MaskEditor()
    qtbot.addWidget(widget)
    widget.resize(820, 748)
    widget.show()
    session = _session()
    selection = SimpleNamespace(model_gray_128=bytes([40]) * (128 * 128))
    widget.set_case(session, selection)
    assert "100 px" in widget.coordinate_label.text()
    viewport_point = widget.view.mapFromScene(80.5, 80.5)
    QTest.mouseClick(widget.view.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point)
    assert session.current[80, 80]
    assert "+" in widget.coordinate_label.text()
    widget.erase_button.click()
    QTest.mouseClick(widget.view.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point)
    assert not session.current[80, 80]
    assert len(session.provenance.records) == 2
