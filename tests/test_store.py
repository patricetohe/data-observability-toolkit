"""Unit tests for observability.store (historical metric snapshot store)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from observability.metrics import ColumnMetrics, TableMetrics
from observability.store import SnapshotStore


def _snapshot(table_name: str, row_count: int, when: datetime) -> TableMetrics:
    return TableMetrics(
        table_name=table_name,
        row_count=row_count,
        collected_at=when,
        columns=[
            ColumnMetrics(name="id", null_pct=0.0, distinct_pct=1.0, min_value="1", max_value=str(row_count)),
        ],
    )


@pytest.fixture
def store(tmp_path):
    return SnapshotStore(tmp_path / "metrics.db")


def test_save_then_latest_round_trips(store):
    snap = _snapshot("public.orders", 100, datetime(2026, 1, 1, tzinfo=timezone.utc))

    row_id = store.save(snap)

    assert row_id == 1
    latest = store.latest("public.orders")
    assert latest is not None
    assert latest.row_count == 100
    assert latest.column("id").max_value == "100"


def test_latest_returns_none_when_no_snapshots(store):
    assert store.latest("does.not_exist") is None


def test_history_is_ordered_oldest_first(store):
    store.save(_snapshot("public.orders", 100, datetime(2026, 1, 3, tzinfo=timezone.utc)))
    store.save(_snapshot("public.orders", 50, datetime(2026, 1, 1, tzinfo=timezone.utc)))
    store.save(_snapshot("public.orders", 75, datetime(2026, 1, 2, tzinfo=timezone.utc)))

    history = store.history("public.orders")

    assert [snap.row_count for snap in history] == [50, 75, 100]


def test_history_limit_keeps_most_recent_in_order(store):
    for day, row_count in [(1, 10), (2, 20), (3, 30), (4, 40)]:
        store.save(_snapshot("public.orders", row_count, datetime(2026, 1, day, tzinfo=timezone.utc)))

    history = store.history("public.orders", limit=2)

    assert [snap.row_count for snap in history] == [30, 40]


def test_history_is_scoped_per_table(store):
    store.save(_snapshot("public.orders", 10, datetime(2026, 1, 1, tzinfo=timezone.utc)))
    store.save(_snapshot("public.customers", 5, datetime(2026, 1, 1, tzinfo=timezone.utc)))

    assert len(store.history("public.orders")) == 1
    assert len(store.history("public.customers")) == 1


def test_table_names_lists_distinct_tables(store):
    store.save(_snapshot("public.orders", 10, datetime(2026, 1, 1, tzinfo=timezone.utc)))
    store.save(_snapshot("public.customers", 5, datetime(2026, 1, 1, tzinfo=timezone.utc)))
    store.save(_snapshot("public.orders", 11, datetime(2026, 1, 2, tzinfo=timezone.utc)))

    assert store.table_names() == ["public.customers", "public.orders"]


def test_count_overall_and_per_table(store):
    store.save(_snapshot("public.orders", 10, datetime(2026, 1, 1, tzinfo=timezone.utc)))
    store.save(_snapshot("public.customers", 5, datetime(2026, 1, 1, tzinfo=timezone.utc)))
    store.save(_snapshot("public.orders", 11, datetime(2026, 1, 2, tzinfo=timezone.utc)))

    assert store.count() == 3
    assert store.count("public.orders") == 2
    assert store.count("public.customers") == 1


def test_persists_across_store_instances(tmp_path):
    db_path = tmp_path / "metrics.db"
    SnapshotStore(db_path).save(_snapshot("public.orders", 42, datetime(2026, 1, 1, tzinfo=timezone.utc)))

    reopened = SnapshotStore(db_path)

    assert reopened.latest("public.orders").row_count == 42
