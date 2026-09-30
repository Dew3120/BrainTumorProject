"""Opt-in, local-only SQLite history repository with no clinical identity fields."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from .constants import CLASS_NAMES, DisplayStatus
from .errors import HistoryRepositoryError

_SCHEMA_VERSION = 1


def _validate_sha256(value: str | None, field: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64 or any(
        ch not in "0123456789abcdef" for ch in value
    ):
        raise HistoryRepositoryError("HISTORY_HASH_INVALID", field)


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    case_id: str
    saved_at: str
    filename: str
    source_sha256: str
    predicted_category: str
    ood_status: str
    final_status: str
    edited: bool
    bundle_path: str
    model_mask_sha256: str | None = None
    edited_mask_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            UUID(self.case_id)
        except ValueError as exc:
            raise HistoryRepositoryError("HISTORY_CASE_ID_INVALID", self.case_id) from exc
        try:
            saved = datetime.fromisoformat(self.saved_at)
        except ValueError as exc:
            raise HistoryRepositoryError("HISTORY_TIMESTAMP_INVALID", self.saved_at) from exc
        if saved.tzinfo is None or saved.utcoffset() is None:
            raise HistoryRepositoryError("HISTORY_TIMESTAMP_INVALID", "Timezone required")
        if not self.filename or Path(self.filename).name != self.filename:
            raise HistoryRepositoryError("HISTORY_FILENAME_INVALID", self.filename)
        _validate_sha256(self.source_sha256, "source_sha256")
        _validate_sha256(self.model_mask_sha256, "model_mask_sha256", optional=True)
        _validate_sha256(self.edited_mask_sha256, "edited_mask_sha256", optional=True)
        if self.predicted_category not in CLASS_NAMES:
            raise HistoryRepositoryError(
                "HISTORY_CATEGORY_INVALID", self.predicted_category
            )
        if self.ood_status not in {"ACCEPT", "REVIEW_REQUIRED"}:
            raise HistoryRepositoryError("HISTORY_OOD_STATUS_INVALID", self.ood_status)
        if self.final_status not in {
            DisplayStatus.STANDARD.value,
            DisplayStatus.REVIEW_REQUIRED.value,
        }:
            raise HistoryRepositoryError(
                "HISTORY_FINAL_STATUS_INVALID", self.final_status
            )
        if self.edited != (self.edited_mask_sha256 is not None):
            raise HistoryRepositoryError(
                "HISTORY_EDIT_STATE_INVALID",
                "edited flag and edited-mask hash disagree",
            )


class HistoryRepository:
    """Transactional repository created only when local history is enabled."""

    def __init__(self, database_path: str | Path, cases_root: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.cases_root = Path(cases_root).expanduser().resolve()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.cases_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    saved_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    ood_status TEXT NOT NULL,
                    final_status TEXT NOT NULL,
                    edited INTEGER NOT NULL CHECK (edited IN (0, 1)),
                    bundle_path TEXT NOT NULL UNIQUE,
                    model_mask_sha256 TEXT,
                    edited_mask_sha256 TEXT
                );
                CREATE INDEX IF NOT EXISTS cases_saved_at_idx
                    ON cases(saved_at DESC, case_id ASC);
                """
            )
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
            elif existing[0] != str(_SCHEMA_VERSION):
                raise HistoryRepositoryError(
                    "HISTORY_SCHEMA_UNSUPPORTED",
                    f"Expected {_SCHEMA_VERSION}, got {existing[0]}",
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _validated_bundle(self, value: str) -> str:
        path = Path(value).expanduser().resolve()
        if not path.is_relative_to(self.cases_root):
            raise HistoryRepositoryError(
                "HISTORY_BUNDLE_PATH_ESCAPE",
                "Case bundle must remain inside the configured cases directory.",
            )
        if not path.is_dir():
            raise HistoryRepositoryError(
                "HISTORY_BUNDLE_MISSING",
                "The case bundle must exist before its index record is saved.",
            )
        return str(path)

    def save(self, record: HistoryRecord) -> None:
        """Insert one opt-in case; duplicates fail rather than overwrite evidence."""

        bundle_path = self._validated_bundle(record.bundle_path)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO cases(
                        case_id, saved_at, filename, source_sha256,
                        predicted_category, ood_status, final_status, edited,
                        bundle_path, model_mask_sha256, edited_mask_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.case_id,
                        record.saved_at,
                        record.filename,
                        record.source_sha256,
                        record.predicted_category,
                        record.ood_status,
                        record.final_status,
                        int(record.edited),
                        bundle_path,
                        record.model_mask_sha256,
                        record.edited_mask_sha256,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise HistoryRepositoryError(
                "HISTORY_DUPLICATE_CASE",
                "The case ID or bundle path is already stored.",
            ) from exc
        except sqlite3.Error as exc:
            raise HistoryRepositoryError("HISTORY_WRITE_FAILED", str(exc)) from exc

    def get(self, case_id: str) -> HistoryRecord | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise HistoryRepositoryError("HISTORY_READ_FAILED", str(exc)) from exc
        return None if row is None else self._record(row)

    def list_recent(self, *, limit: int = 100, query: str | None = None) -> list[HistoryRecord]:
        if isinstance(limit, bool) or not 1 <= limit <= 500:
            raise HistoryRepositoryError("HISTORY_LIMIT_INVALID", repr(limit))
        sql = "SELECT * FROM cases"
        values: list[object] = []
        if query:
            sql += " WHERE filename LIKE ? ESCAPE '\\' OR case_id LIKE ? ESCAPE '\\'"
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.extend((f"%{escaped}%", f"%{escaped}%"))
        sql += " ORDER BY saved_at DESC, case_id ASC LIMIT ?"
        values.append(limit)
        try:
            with self._connect() as connection:
                rows = connection.execute(sql, values).fetchall()
        except sqlite3.Error as exc:
            raise HistoryRepositoryError("HISTORY_READ_FAILED", str(exc)) from exc
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> HistoryRecord:
        return HistoryRecord(
            case_id=row["case_id"],
            saved_at=row["saved_at"],
            filename=row["filename"],
            source_sha256=row["source_sha256"],
            predicted_category=row["predicted_category"],
            ood_status=row["ood_status"],
            final_status=row["final_status"],
            edited=bool(row["edited"]),
            bundle_path=row["bundle_path"],
            model_mask_sha256=row["model_mask_sha256"],
            edited_mask_sha256=row["edited_mask_sha256"],
        )
