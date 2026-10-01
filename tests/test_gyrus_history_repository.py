"""History index filtering, migration, and contained deletion contracts."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from gyrus.domain.errors import HistoryRepositoryError
from gyrus.domain.history import HistoryRecord, HistoryRepository


def _record(cases: Path, *, hour: int, category: str, status: str, ood: str,
            edited: bool = False) -> HistoryRecord:
    case_id = str(uuid4())
    folder = cases / case_id
    folder.mkdir(parents=True)
    return HistoryRecord(
        case_id=case_id,
        saved_at=(datetime(2026, 10, 1, tzinfo=UTC) + timedelta(hours=hour)).isoformat(),
        filename=f"case_{hour}.png",
        source_sha256="a" * 64,
        predicted_category=category,
        ood_status=ood,
        final_status=status,
        edited=edited,
        bundle_path=str(folder),
        model_mask_sha256="b" * 64,
        edited_mask_sha256="c" * 64 if edited else None,
        source_saved=True,
        bundle_manifest_sha256="d" * 64,
    )


def test_filters_sort_and_pagination(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    repo = HistoryRepository(tmp_path / "history.db", cases)
    repo.initialize()
    early = _record(cases, hour=1, category="glioma", status="REVIEW_REQUIRED", ood="ACCEPT", edited=True)
    late = _record(cases, hour=3, category="pituitary", status="STANDARD", ood="ACCEPT")
    middle = _record(cases, hour=2, category="glioma", status="REVIEW_REQUIRED", ood="REVIEW_REQUIRED")
    for record in (early, late, middle):
        repo.save(record)
    assert repo.list_recent() == [late, middle, early]
    assert repo.list_recent(limit=1, offset=1) == [middle]
    assert repo.list_recent(category="glioma", edited=True) == [early]
    assert repo.list_recent(status="STANDARD", ood="ACCEPT") == [late]
    assert repo.list_recent(date_from="2026-10-01", date_to="2026-10-01") == [late, middle, early]
    assert repo.list_recent(query="case_2") == [middle]
    with pytest.raises(HistoryRepositoryError, match="HISTORY_FILTER_INVALID"):
        repo.list_recent(date_from="yesterday")


def test_delete_removes_only_direct_child_bundle(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    repo = HistoryRepository(tmp_path / "history.db", cases)
    repo.initialize()
    record = _record(cases, hour=1, category="glioma", status="STANDARD", ood="ACCEPT")
    (Path(record.bundle_path) / "source.png").write_bytes(b"app managed")
    repo.save(record)
    repo.delete(record.case_id)
    assert repo.get(record.case_id) is None
    assert not Path(record.bundle_path).exists()
    with pytest.raises(HistoryRepositoryError, match="HISTORY_NOT_FOUND"):
        repo.delete(record.case_id)


def test_index_rejects_nested_bundle(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    repo = HistoryRepository(tmp_path / "history.db", cases)
    repo.initialize()
    record = _record(cases, hour=1, category="glioma", status="STANDARD", ood="ACCEPT")
    nested = cases / "other" / record.case_id
    nested.mkdir(parents=True)
    with pytest.raises(HistoryRepositoryError, match="HISTORY_BUNDLE_PATH_ESCAPE"):
        repo.save(replace(record, bundle_path=str(nested)))


def test_schema_one_migrates_additively(tmp_path):
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata(key, value) VALUES('schema_version', '1');
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY, saved_at TEXT NOT NULL, filename TEXT NOT NULL,
                source_sha256 TEXT NOT NULL, predicted_category TEXT NOT NULL,
                ood_status TEXT NOT NULL, final_status TEXT NOT NULL,
                edited INTEGER NOT NULL, bundle_path TEXT NOT NULL UNIQUE,
                model_mask_sha256 TEXT, edited_mask_sha256 TEXT
            );
        """)
    repo = HistoryRepository(path, tmp_path / "cases")
    repo.initialize()
    with sqlite3.connect(path) as connection:
        version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(cases)")}
    assert version == "2"
    assert {"source_saved", "bundle_manifest_sha256"} <= columns
