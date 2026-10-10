"""Unit tests for observability.profiler.

Uses a file-backed SQLite database (via SQLAlchemy) as a fast, dependency-free
stand-in for a DuckDB/Postgres table: profile_table() only relies on
SQLAlchemy's generic Core API (reflection + aggregate SELECTs), so the same
code path exercised here against SQLite is what runs against DuckDB or
Postgres in production.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from observability.config import TableConfig
from observability.profiler import profile_table


@pytest.fixture()
def sqlite_engine(tmp_path):
    db_path = tmp_path / "profiler_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE orders ("
                "id INTEGER, status TEXT, created_at TIMESTAMP"
                ")"
            )
        )
        rows = [
            (1, "shipped", "2026-01-01 00:00:00"),
            (2, "shipped", "2026-01-01 01:00:00"),
            (3, "cancelled", "2026-01-01 02:00:00"),
            (4, None, "2026-01-01 03:00:00"),
        ]
        conn.execute(
            text("INSERT INTO orders (id, status, created_at) VALUES (:id, :status, :created_at)"),
            [{"id": r[0], "status": r[1], "created_at": r[2]} for r in rows],
        )
    return engine


def _orders_table_config(**overrides) -> TableConfig:
    defaults = {"name": "orders", "connection": "warehouse"}
    defaults.update(overrides)
    return TableConfig(**defaults)


def test_profile_table_row_count(sqlite_engine):
    metrics = profile_table(sqlite_engine, _orders_table_config())

    assert metrics.table_name == "orders"
    assert metrics.row_count == 4


def test_profile_table_column_metrics(sqlite_engine):
    metrics = profile_table(sqlite_engine, _orders_table_config())

    status = metrics.column("status")
    assert status is not None
    assert status.null_pct == pytest.approx(0.25)
    assert status.distinct_pct == pytest.approx(0.5)
    assert status.min_value == "cancelled"
    assert status.max_value == "shipped"

    id_col = metrics.column("id")
    assert id_col is not None
    assert id_col.null_pct == pytest.approx(0.0)
    assert id_col.distinct_pct == pytest.approx(1.0)
    assert id_col.min_value == "1"
    assert id_col.max_value == "4"


def test_profile_table_freshness(sqlite_engine):
    reference = datetime(2026, 1, 1, 4, 0, 0, tzinfo=timezone.utc)

    metrics = profile_table(
        sqlite_engine,
        _orders_table_config(freshness_column="created_at"),
        now=reference,
    )

    assert metrics.freshness_seconds == pytest.approx(timedelta(hours=1).total_seconds())


def test_profile_table_without_freshness_column_is_none(sqlite_engine):
    metrics = profile_table(sqlite_engine, _orders_table_config())

    assert metrics.freshness_seconds is None


def test_profile_table_empty_table_has_zeroed_percentages(tmp_path):
    db_path = tmp_path / "empty.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE empty_table (id INTEGER, name TEXT)"))

    metrics = profile_table(engine, _orders_table_config(name="empty_table"))

    assert metrics.row_count == 0
    name_col = metrics.column("name")
    assert name_col is not None
    assert name_col.null_pct == 0.0
    assert name_col.distinct_pct == 0.0
