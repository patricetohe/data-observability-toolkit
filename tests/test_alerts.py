"""Unit tests for observability.alerts (threshold-based alert rule engine)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from observability.alerts import (
    Alert,
    AlertSeverity,
    evaluate_all,
    evaluate_drift,
    evaluate_freshness,
    evaluate_null_pct,
    evaluate_row_count_anomaly,
    evaluate_schema_change,
)
from observability.config import AlertThresholds
from observability.drift import detect_drift
from observability.metrics import ColumnMetrics, TableMetrics
from observability.schema_change import ColumnSchema, TableSchema, diff_schemas


def _snapshot(
    table_name: str,
    day: int,
    *,
    row_count: int = 100,
    null_pct: float = 0.0,
    distinct_pct: float = 1.0,
    freshness_seconds: float | None = None,
) -> TableMetrics:
    return TableMetrics(
        table_name=table_name,
        row_count=row_count,
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        freshness_seconds=freshness_seconds,
        columns=[ColumnMetrics(name="id", null_pct=null_pct, distinct_pct=distinct_pct)],
    )


def test_evaluate_null_pct_flags_column_over_max():
    metrics = _snapshot("public.orders", 0, null_pct=0.5)
    thresholds = AlertThresholds(null_pct_max=0.2)

    alerts = evaluate_null_pct(metrics, thresholds)

    assert len(alerts) == 1
    assert alerts[0].rule == "null_pct_max"
    assert alerts[0].severity == AlertSeverity.WARNING
    assert alerts[0].table_name == "public.orders"
    assert "id" in alerts[0].message


def test_evaluate_null_pct_under_max_is_silent():
    metrics = _snapshot("public.orders", 0, null_pct=0.05)
    thresholds = AlertThresholds(null_pct_max=0.2)

    assert evaluate_null_pct(metrics, thresholds) == []


def test_evaluate_freshness_flags_stale_table():
    metrics = _snapshot("public.orders", 0, freshness_seconds=48 * 3600)
    thresholds = AlertThresholds(freshness_max_hours=24.0)

    alerts = evaluate_freshness(metrics, thresholds)

    assert len(alerts) == 1
    assert alerts[0].rule == "freshness_max_hours"
    assert alerts[0].severity == AlertSeverity.CRITICAL


def test_evaluate_freshness_unknown_is_silent():
    metrics = _snapshot("public.orders", 0, freshness_seconds=None)
    thresholds = AlertThresholds(freshness_max_hours=24.0)

    assert evaluate_freshness(metrics, thresholds) == []


def test_evaluate_freshness_within_max_is_silent():
    metrics = _snapshot("public.orders", 0, freshness_seconds=3600)
    thresholds = AlertThresholds(freshness_max_hours=24.0)

    assert evaluate_freshness(metrics, thresholds) == []


def test_evaluate_row_count_anomaly_flags_spike():
    history = [_snapshot("public.orders", day, row_count=rc) for day, rc in enumerate([100, 102, 98, 101, 99, 900])]
    thresholds = AlertThresholds(row_count_zscore_max=3.0)

    alerts = evaluate_row_count_anomaly(history, thresholds)

    assert len(alerts) == 1
    assert alerts[0].rule == "row_count_zscore_max"
    assert alerts[0].table_name == "public.orders"


def test_evaluate_row_count_anomaly_stable_history_is_silent():
    history = [_snapshot("public.orders", day, row_count=rc) for day, rc in enumerate([100, 101, 99, 100, 102, 101])]
    thresholds = AlertThresholds(row_count_zscore_max=3.0)

    assert evaluate_row_count_anomaly(history, thresholds) == []


def test_evaluate_row_count_anomaly_empty_history_is_silent():
    assert evaluate_row_count_anomaly([], AlertThresholds()) == []


def test_evaluate_schema_change_none_is_silent():
    assert evaluate_schema_change(None) == []


def test_evaluate_schema_change_flags_added_and_removed_columns():
    old = TableSchema(table_name="public.orders", columns=[ColumnSchema(name="id", dtype="INTEGER")])
    new = TableSchema(
        table_name="public.orders",
        columns=[ColumnSchema(name="id", dtype="BIGINT"), ColumnSchema(name="status", dtype="VARCHAR")],
    )
    change = diff_schemas(old, new)

    alerts = evaluate_schema_change(change)

    assert len(alerts) == 1
    assert alerts[0].rule == "schema_change"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert "status" in alerts[0].message
    assert "id" in alerts[0].message


def test_evaluate_schema_change_no_changes_is_silent():
    same = TableSchema(table_name="public.orders", columns=[ColumnSchema(name="id", dtype="INTEGER")])
    change = diff_schemas(same, same)

    assert evaluate_schema_change(change) == []


def test_evaluate_drift_none_is_silent():
    assert evaluate_drift(None) == []


def test_evaluate_drift_flags_significant_shift_as_critical():
    baseline = _snapshot("public.orders", 0, null_pct=0.01, distinct_pct=0.9)
    current = _snapshot("public.orders", 1, null_pct=0.9, distinct_pct=0.9)
    drift_result = detect_drift(baseline, current)

    alerts = evaluate_drift(drift_result, psi_threshold=0.1)

    assert len(alerts) == 1
    assert alerts[0].rule == "data_drift_psi"
    assert alerts[0].severity == AlertSeverity.CRITICAL


def test_evaluate_drift_no_shift_is_silent():
    baseline = _snapshot("public.orders", 0, null_pct=0.01, distinct_pct=0.9)
    current = _snapshot("public.orders", 1, null_pct=0.01, distinct_pct=0.9)
    drift_result = detect_drift(baseline, current)

    assert evaluate_drift(drift_result, psi_threshold=0.1) == []


def test_evaluate_all_combines_every_rule():
    history = [_snapshot("public.orders", day, row_count=100, null_pct=0.5) for day in range(5)]
    latest = _snapshot("public.orders", 5, row_count=100, null_pct=0.5)
    history.append(latest)
    thresholds = AlertThresholds(null_pct_max=0.2, row_count_zscore_max=3.0, freshness_max_hours=24.0)

    alerts = evaluate_all(
        latest_metrics=latest,
        metrics_history=history,
        thresholds=thresholds,
    )

    assert all(isinstance(a, Alert) for a in alerts)
    rules = {a.rule for a in alerts}
    assert "null_pct_max" in rules
    # stable history -> no row_count anomaly, no schema/drift inputs given
    assert "row_count_zscore_max" not in rules
    assert "schema_change" not in rules
    assert "data_drift_psi" not in rules
