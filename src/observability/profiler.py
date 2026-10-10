"""Profiler module: scans a live table and produces a TableMetrics snapshot.

Works against any SQLAlchemy-supported dialect (DuckDB, PostgreSQL, SQLite,
...) via reflection, so the same code path profiles a DuckDB file, a
Postgres warehouse table, or (as in the test suite) an in-memory SQLite
table used as a fast stand-in for CI. Only aggregate SQL is issued -- the
table's rows are never pulled into memory -- so this scales to large tables.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import MetaData, Table, func, select
from sqlalchemy.engine import Connection, Engine

from .config import TableConfig
from .metrics import ColumnMetrics, TableMetrics


def profile_table(
    engine: Engine,
    table_config: TableConfig,
    *,
    now: Optional[datetime] = None,
) -> TableMetrics:
    """Scan one table through ``engine`` and return a `TableMetrics` snapshot.

    Reflects the table's columns, then computes row count, per-column
    null %, distinct %, and min/max, plus freshness (if
    ``table_config.freshness_column`` is set) using aggregate queries.
    """
    metadata = MetaData()
    table = Table(table_config.name, metadata, autoload_with=engine)

    with engine.connect() as conn:
        row_count = _row_count(conn, table)
        columns = [_profile_column(conn, table, column, row_count) for column in table.columns]

        freshness_seconds = None
        if table_config.freshness_column:
            freshness_seconds = _compute_freshness(
                conn, table, table_config.freshness_column, now=now
            )

    return TableMetrics(
        table_name=table_config.name,
        row_count=row_count,
        columns=columns,
        freshness_seconds=freshness_seconds,
    )


def _row_count(conn: Connection, table: Table) -> int:
    return conn.execute(select(func.count()).select_from(table)).scalar_one()


def _profile_column(conn: Connection, table: Table, column, row_count: int) -> ColumnMetrics:
    null_count = conn.execute(
        select(func.count()).select_from(table).where(column.is_(None))
    ).scalar_one()
    distinct_count = conn.execute(
        select(func.count(func.distinct(column))).select_from(table)
    ).scalar_one()
    min_value, max_value = conn.execute(select(func.min(column), func.max(column))).one()

    null_pct = (null_count / row_count) if row_count else 0.0
    distinct_pct = (distinct_count / row_count) if row_count else 0.0

    return ColumnMetrics(
        name=column.name,
        # Aggregate counts can round-trip to e.g. 1.0000000000000002 for
        # some DB drivers; clamp into the [0, 1] range the model requires.
        null_pct=min(max(null_pct, 0.0), 1.0),
        distinct_pct=min(max(distinct_pct, 0.0), 1.0),
        min_value=None if min_value is None else str(min_value),
        max_value=None if max_value is None else str(max_value),
    )


def _compute_freshness(
    conn: Connection,
    table: Table,
    freshness_column: str,
    *,
    now: Optional[datetime],
) -> Optional[float]:
    """Seconds between ``now`` and the most recent value in a timestamp column."""
    most_recent = conn.execute(select(func.max(table.c[freshness_column]))).scalar_one()
    if most_recent is None or not isinstance(most_recent, datetime):
        return None

    if most_recent.tzinfo is None:
        most_recent = most_recent.replace(tzinfo=timezone.utc)

    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    return max((reference - most_recent).total_seconds(), 0.0)
