"""Unit tests for observability.drift (PSI-based data drift detection)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from observability.drift import (
    DriftResult,
    detect_drift,
    detect_drift_history,
    latest_drift,
    psi,
    psi_severity,
)
from observability.metrics import ColumnMetrics, TableMetrics


def _snapshot(table_name: str, day: int, *, null_pct: float, distinct_pct: float) -> TableMetrics:
    return TableMetrics(
        table_name=table_name,
        row_count=100,
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        columns=[ColumnMetrics(name="status", null_pct=null_pct, distinct_pct=distinct_pct)],
    )


def test_psi_identical_distributions_is_zero():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_psi_is_symmetric_only_in_shape_not_value():
    # PSI is not a true distance metric (not symmetric), but a large shift
    # in either direction should register as a large positive value.
    forward = psi([0.01, 0.99], [0.5, 0.5])
    backward = psi([0.5, 0.5], [0.01, 0.99])
    assert forward > 0.25
    assert backward > 0.25


def test_psi_rejects_mismatched_bucket_counts():
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [1.0])


def test_psi_severity_thresholds():
    assert psi_severity(0.0) == "none"
    assert psi_severity(0.05) == "none"
    assert psi_severity(0.1) == "moderate"
    assert psi_severity(0.2) == "moderate"
    assert psi_severity(0.25) == "significant"
    assert psi_severity(1.0) == "significant"


def test_detect_drift_no_change_has_no_drift():
    baseline = _snapshot("public.orders", 0, null_pct=0.01, distinct_pct=0.9)
    current = _snapshot("public.orders", 1, null_pct=0.01, distinct_pct=0.9)

    result = detect_drift(baseline, current)

    assert isinstance(result, DriftResult)
    assert result.has_drift() is False
    assert result.drifted_columns() == []
    assert all(c.severity == "none" for c in result.columns)


def test_detect_drift_flags_significant_null_pct_shift():
    baseline = _snapshot("public.orders", 0, null_pct=0.01, distinct_pct=0.9)
    current = _snapshot("public.orders", 1, null_pct=0.6, distinct_pct=0.9)

    result = detect_drift(baseline, current)

    assert result.has_drift() is True
    flagged = result.drifted_columns()
    assert len(flagged) == 1
    assert flagged[0].column == "status"
    assert flagged[0].metric == "null_pct"
    assert flagged[0].severity == "significant"


def test_detect_drift_only_compares_shared_columns():
    baseline = TableMetrics(
        table_name="public.orders",
        row_count=100,
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        columns=[
            ColumnMetrics(name="status", null_pct=0.0, distinct_pct=0.5),
            ColumnMetrics(name="legacy_flag", null_pct=0.0, distinct_pct=1.0),
        ],
    )
    current = TableMetrics(
        table_name="public.orders",
        row_count=100,
        collected_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        columns=[
            ColumnMetrics(name="status", null_pct=0.0, distinct_pct=0.5),
            ColumnMetrics(name="discount", null_pct=0.0, distinct_pct=0.2),
        ],
    )

    result = detect_drift(baseline, current)

    compared_columns = {c.column for c in result.columns}
    assert compared_columns == {"status"}


def test_detect_drift_rejects_different_tables():
    baseline = _snapshot("public.orders", 0, null_pct=0.0, distinct_pct=0.5)
    current = _snapshot("public.customers", 1, null_pct=0.0, distinct_pct=0.5)

    with pytest.raises(ValueError):
        detect_drift(baseline, current)


def test_detect_drift_history_needs_at_least_two_snapshots():
    only_one = [_snapshot("public.orders", 0, null_pct=0.0, distinct_pct=0.5)]
    assert detect_drift_history(only_one) == []
    assert detect_drift_history([]) == []


def test_detect_drift_history_returns_one_result_per_transition():
    history = [
        _snapshot("public.orders", 0, null_pct=0.01, distinct_pct=0.9),
        _snapshot("public.orders", 1, null_pct=0.01, distinct_pct=0.9),
        _snapshot("public.orders", 2, null_pct=0.7, distinct_pct=0.9),
    ]

    results = detect_drift_history(history)

    assert len(results) == 2
    assert results[0].has_drift() is False
    assert results[1].has_drift() is True


def test_latest_drift_none_when_no_results():
    assert latest_drift([]) is None


def test_latest_drift_none_when_last_transition_stable():
    history = [
        _snapshot("public.orders", 0, null_pct=0.01, distinct_pct=0.9),
        _snapshot("public.orders", 1, null_pct=0.01, distinct_pct=0.9),
    ]
    results = detect_drift_history(history)
    assert latest_drift(results) is None


def test_latest_drift_returns_last_transition_when_drifted():
    history = [
        _snapshot("public.orders", 0, null_pct=0.01, distinct_pct=0.9),
        _snapshot("public.orders", 1, null_pct=0.65, distinct_pct=0.9),
    ]
    results = detect_drift_history(history)

    result = latest_drift(results)

    assert result is not None
    assert result.has_drift() is True
