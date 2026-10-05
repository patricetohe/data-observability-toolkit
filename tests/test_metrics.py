"""Unit tests for observability.metrics."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from observability.metrics import ColumnMetrics, TableMetrics


def _sample_table_metrics() -> TableMetrics:
    return TableMetrics(
        table_name="public.orders",
        row_count=1000,
        freshness_seconds=120.0,
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        columns=[
            ColumnMetrics(name="id", null_pct=0.0, distinct_pct=1.0, min_value="1", max_value="1000"),
            ColumnMetrics(name="status", null_pct=0.05, distinct_pct=0.004, min_value="cancelled", max_value="shipped"),
        ],
    )


def test_table_metrics_round_trip():
    metrics = _sample_table_metrics()

    assert metrics.table_name == "public.orders"
    assert metrics.row_count == 1000
    assert len(metrics.columns) == 2


def test_column_lookup_by_name():
    metrics = _sample_table_metrics()

    status = metrics.column("status")
    assert status is not None
    assert status.null_pct == pytest.approx(0.05)

    assert metrics.column("does_not_exist") is None


def test_is_stale_with_known_freshness():
    metrics = _sample_table_metrics()

    assert metrics.is_stale(60) is True
    assert metrics.is_stale(600) is False


def test_is_stale_without_freshness_is_false():
    metrics = TableMetrics(table_name="public.customers", row_count=10)

    assert metrics.freshness_seconds is None
    assert metrics.is_stale(0) is False


def test_to_flat_dict_namespaces_column_metrics():
    metrics = _sample_table_metrics()

    flat = metrics.to_flat_dict()

    assert flat["table_name"] == "public.orders"
    assert flat["row_count"] == 1000
    assert flat["id.null_pct"] == 0.0
    assert flat["id.min"] == "1"
    assert flat["status.max"] == "shipped"
    assert flat["collected_at"] == "2026-01-01T00:00:00+00:00"


def test_null_pct_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ColumnMetrics(name="id", null_pct=1.5, distinct_pct=0.5)


def test_blank_table_name_rejected():
    with pytest.raises(ValidationError):
        TableMetrics(table_name="   ", row_count=0)


def test_negative_row_count_rejected():
    with pytest.raises(ValidationError):
        TableMetrics(table_name="public.orders", row_count=-1)
