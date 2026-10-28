"""Schema-change detector: tracks a table's column list and types over time.

Captures a point-in-time `TableSchema` (column names and SQLAlchemy type
strings) via the same generic reflection approach the profiler
(`observability.profiler`) uses, then diffs consecutive snapshots to surface
added columns, removed columns, and type changes. Snapshots are plain
pydantic models so they can round-trip through persistence the same way
`TableMetrics` does via `SnapshotStore` (wiring schema history into that
store, or a dedicated one, is left to a later roadmap item); this module
only defines capture and diffing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import MetaData, Table
from sqlalchemy.engine import Engine

from .config import TableConfig


class ColumnSchema(BaseModel):
    """The name and type of a single column at capture time."""

    name: str
    dtype: str = Field(..., description="String form of the column's SQLAlchemy type, e.g. 'INTEGER'.")

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("column name must not be blank")
        return value.strip()


class TableSchema(BaseModel):
    """A snapshot of one table's column list and types at a point in time."""

    table_name: str
    columns: list[ColumnSchema] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("table_name")
    @classmethod
    def _table_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("table_name must not be blank")
        return value.strip()

    def column_names(self) -> set[str]:
        """Return the set of column names in this snapshot."""
        return {c.name for c in self.columns}

    def dtype(self, name: str) -> Optional[str]:
        """Return the dtype string for ``name``, or None if absent."""
        return next((c.dtype for c in self.columns if c.name == name), None)


class TypeChange(BaseModel):
    """A single column whose type differs between two snapshots."""

    column: str
    old_dtype: str
    new_dtype: str


class SchemaChange(BaseModel):
    """The diff between two consecutive schema snapshots of the same table."""

    table_name: str
    captured_at: str
    added_columns: list[str] = Field(default_factory=list)
    removed_columns: list[str] = Field(default_factory=list)
    type_changes: list[TypeChange] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """True if this transition added, removed, or retyped any column."""
        return bool(self.added_columns or self.removed_columns or self.type_changes)


def capture_schema(
    engine: Engine, table_config: TableConfig, *, now: Optional[datetime] = None
) -> TableSchema:
    """Reflect ``table_config.name`` through ``engine`` into a `TableSchema`.

    Uses the same SQLAlchemy reflection as `observability.profiler`, so it
    works against any supported dialect. Only catalog/metadata lookups are
    issued -- no data-reading queries.
    """
    metadata = MetaData()
    table = Table(table_config.name, metadata, autoload_with=engine)
    columns = [ColumnSchema(name=column.name, dtype=str(column.type)) for column in table.columns]
    return TableSchema(
        table_name=table_config.name,
        columns=columns,
        captured_at=now or datetime.now(timezone.utc),
    )


def diff_schemas(old: TableSchema, new: TableSchema) -> SchemaChange:
    """Compare two snapshots of the same table and return what changed.

    Raises ValueError if the snapshots belong to different tables --
    diffing across tables is almost certainly a caller bug, not a real
    schema change.
    """
    if old.table_name != new.table_name:
        raise ValueError(
            f"cannot diff schemas for different tables: {old.table_name!r} vs {new.table_name!r}"
        )

    old_names = old.column_names()
    new_names = new.column_names()

    added = sorted(new_names - old_names)
    removed = sorted(old_names - new_names)

    type_changes: list[TypeChange] = []
    for name in sorted(old_names & new_names):
        old_dtype = old.dtype(name)
        new_dtype = new.dtype(name)
        if old_dtype != new_dtype:
            type_changes.append(TypeChange(column=name, old_dtype=old_dtype, new_dtype=new_dtype))

    return SchemaChange(
        table_name=new.table_name,
        captured_at=new.captured_at.isoformat(),
        added_columns=added,
        removed_columns=removed,
        type_changes=type_changes,
    )


def detect_schema_changes(history: list[TableSchema]) -> list[SchemaChange]:
    """Diff every consecutive pair of snapshots in ``history``.

    ``history`` must be ordered oldest-first (matching `SnapshotStore`'s
    contract for metric history). Returns one `SchemaChange` per
    transition -- i.e. ``len(history) - 1`` results -- so a lone first
    snapshot (nothing to compare it against) never produces a result on
    its own.
    """
    if len(history) < 2:
        return []
    return [diff_schemas(old, new) for old, new in zip(history[:-1], history[1:])]


def latest_schema_change(results: list[SchemaChange]) -> Optional[SchemaChange]:
    """Return the most recent transition's SchemaChange, if it has changes.

    Convenience for callers (e.g. the alert rule engine, a later roadmap
    item) that only care whether the table's schema just changed, not the
    whole history of transitions -- mirrors
    `observability.anomaly.latest_anomalies` for metric anomalies.
    """
    if not results:
        return None
    return results[-1] if results[-1].has_changes else None
