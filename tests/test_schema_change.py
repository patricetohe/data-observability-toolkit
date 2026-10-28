"""Unit tests for observability.schema_change (schema-change detection)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from observability.config import TableConfig
from observability.schema_change import (
    ColumnSchema,
    TableSchema,
    capture_schema,
    detect_schema_changes,
    diff_schemas,
    latest_schema_change,
)


def _schema(table_name: str, columns: dict[str, str], day: int) -> TableSchema:
    return TableSchema(
        table_name=table_name,
        columns=[ColumnSchema(name=name, dtype=dtype) for name, dtype in columns.items()],
        captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
    )


def _orders_table_config(**overrides) -> TableConfig:
    defaults = {"name": "orders", "connection": "warehouse"}
    defaults.update(overrides)
    return TableConfig(**defaults)


@pytest.fixture()
def sqlite_engine(tmp_path):
    db_path = tmp_path / "schema_change_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE orders (id INTEGER, status TEXT, created_at TIMESTAMP)"))
    return engine


def test_capture_schema_reflects_columns_and_types(sqlite_engine):
    schema = capture_schema(sqlite_engine, _orders_table_config())

    assert schema.table_name == "orders"
    assert schema.column_names() == {"id", "status", "created_at"}
    assert schema.dtype("id") == "INTEGER"
    assert schema.dtype("status") == "TEXT"


def test_capture_schema_uses_provided_now(sqlite_engine):
    reference = datetime(2026, 1, 1, tzinfo=timezone.utc)
    schema = capture_schema(sqlite_engine, _orders_table_config(), now=reference)
    assert schema.captured_at == reference


def test_diff_schemas_no_changes_is_empty():
    old = _schema("public.orders", {"id": "INTEGER", "status": "TEXT"}, 0)
    new = _schema("public.orders", {"id": "INTEGER", "status": "TEXT"}, 1)

    change = diff_schemas(old, new)

    assert change.has_changes is False
    assert change.added_columns == []
    assert change.removed_columns == []
    assert change.type_changes == []


def test_diff_schemas_detects_added_column():
    old = _schema("public.orders", {"id": "INTEGER"}, 0)
    new = _schema("public.orders", {"id": "INTEGER", "discount": "NUMERIC"}, 1)

    change = diff_schemas(old, new)

    assert change.has_changes is True
    assert change.added_columns == ["discount"]
    assert change.removed_columns == []


def test_diff_schemas_detects_removed_column():
    old = _schema("public.orders", {"id": "INTEGER", "legacy_flag": "BOOLEAN"}, 0)
    new = _schema("public.orders", {"id": "INTEGER"}, 1)

    change = diff_schemas(old, new)

    assert change.has_changes is True
    assert change.removed_columns == ["legacy_flag"]
    assert change.added_columns == []


def test_diff_schemas_detects_type_change():
    old = _schema("public.orders", {"id": "INTEGER", "amount": "INTEGER"}, 0)
    new = _schema("public.orders", {"id": "INTEGER", "amount": "NUMERIC"}, 1)

    change = diff_schemas(old, new)

    assert change.has_changes is True
    assert len(change.type_changes) == 1
    assert change.type_changes[0].column == "amount"
    assert change.type_changes[0].old_dtype == "INTEGER"
    assert change.type_changes[0].new_dtype == "NUMERIC"


def test_diff_schemas_rejects_different_tables():
    old = _schema("public.orders", {"id": "INTEGER"}, 0)
    new = _schema("public.customers", {"id": "INTEGER"}, 1)

    with pytest.raises(ValueError):
        diff_schemas(old, new)


def test_detect_schema_changes_needs_at_least_two_snapshots():
    only_one = [_schema("public.orders", {"id": "INTEGER"}, 0)]
    assert detect_schema_changes(only_one) == []
    assert detect_schema_changes([]) == []


def test_detect_schema_changes_returns_one_result_per_transition():
    history = [
        _schema("public.orders", {"id": "INTEGER"}, 0),
        _schema("public.orders", {"id": "INTEGER", "status": "TEXT"}, 1),
        _schema("public.orders", {"id": "INTEGER", "status": "TEXT"}, 2),
        _schema("public.orders", {"id": "INTEGER"}, 3),
    ]

    results = detect_schema_changes(history)

    assert len(results) == 3
    assert results[0].added_columns == ["status"]
    assert results[1].has_changes is False
    assert results[2].removed_columns == ["status"]


def test_latest_schema_change_none_when_no_results():
    assert latest_schema_change([]) is None


def test_latest_schema_change_none_when_last_transition_unchanged():
    history = [
        _schema("public.orders", {"id": "INTEGER"}, 0),
        _schema("public.orders", {"id": "INTEGER", "status": "TEXT"}, 1),
        _schema("public.orders", {"id": "INTEGER", "status": "TEXT"}, 2),
    ]
    results = detect_schema_changes(history)
    assert latest_schema_change(results) is None


def test_latest_schema_change_returns_last_transition_when_changed():
    history = [
        _schema("public.orders", {"id": "INTEGER"}, 0),
        _schema("public.orders", {"id": "INTEGER", "status": "TEXT"}, 1),
    ]
    results = detect_schema_changes(history)

    change = latest_schema_change(results)

    assert change is not None
    assert change.added_columns == ["status"]
