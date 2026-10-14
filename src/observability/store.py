"""Historical metric snapshot store (SQLite).

Persists every `TableMetrics` snapshot produced by the profiler so later
roadmap items (anomaly detection, schema-change detection, drift checks,
the dashboard) can read back a table's metric history over time. Snapshots
are kept as JSON so the full nested shape (and any future field added to
`TableMetrics`) round-trips exactly, while `table_name` and `collected_at`
are indexed columns so lookups don't require deserializing every row.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional, Union

from .metrics import TableMetrics

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_table_name ON snapshots (table_name);
CREATE INDEX IF NOT EXISTS idx_snapshots_table_collected ON snapshots (table_name, collected_at);
"""


class SnapshotStore:
    """A SQLite-backed, append-only store of `TableMetrics` snapshots.

    One store file can hold history for many tables; rows are never
    updated or deleted, so `history()` always reflects everything ever
    profiled for a table.
    """

    def __init__(self, db_path: Union[str, Path] = "observability_metrics.db"):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def save(self, metrics: TableMetrics) -> int:
        """Persist one snapshot and return its new row id."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO snapshots (table_name, collected_at, row_count, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    metrics.table_name,
                    metrics.collected_at.isoformat(),
                    metrics.row_count,
                    metrics.model_dump_json(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def history(self, table_name: str, *, limit: Optional[int] = None) -> list[TableMetrics]:
        """Return snapshots for ``table_name``, ordered oldest first.

        With ``limit`` set, returns the most recent ``limit`` snapshots
        (still ordered oldest first) instead of the full history.
        """
        if limit is None:
            rows = self._query(
                "SELECT payload FROM snapshots WHERE table_name = ? "
                "ORDER BY collected_at ASC, id ASC",
                (table_name,),
            )
        else:
            rows = self._query(
                "SELECT payload FROM ("
                "  SELECT payload, collected_at, id FROM snapshots WHERE table_name = ? "
                "  ORDER BY collected_at DESC, id DESC LIMIT ?"
                ") ORDER BY collected_at ASC, id ASC",
                (table_name, limit),
            )
        return [TableMetrics.model_validate_json(row["payload"]) for row in rows]

    def latest(self, table_name: str) -> Optional[TableMetrics]:
        """Return the most recent snapshot for ``table_name``, or None if there isn't one."""
        results = self.history(table_name, limit=1)
        return results[-1] if results else None

    def table_names(self) -> list[str]:
        """Return the distinct table names that have at least one stored snapshot."""
        rows = self._query("SELECT DISTINCT table_name FROM snapshots ORDER BY table_name", ())
        return [row["table_name"] for row in rows]

    def count(self, table_name: Optional[str] = None) -> int:
        """Return the number of stored snapshots, optionally filtered to one table."""
        if table_name is None:
            rows = self._query("SELECT COUNT(*) AS n FROM snapshots", ())
        else:
            rows = self._query(
                "SELECT COUNT(*) AS n FROM snapshots WHERE table_name = ?", (table_name,)
            )
        return int(rows[0]["n"])

    def _query(self, sql: str, params: tuple) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchall()
