"""Unit tests for observability.anomaly (rolling z-score anomaly detection)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from observability.anomaly import (
    detect_column_null_pct_anomalies,
    detect_metric_anomalies,
    detect_row_count_anomalies,
    latest_anomalies,
    rolling_zscores,
)
from observability.metrics import ColumnMetrics, TableMetrics


def _snapshot(table_name: str, row_count: int, day: int, null_pct: float = 0.0) -> TableMetrics:
    return TableMetrics(
        table_name=table_name,
        row_count=row_count,
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        columns=[ColumnMetrics(name="id", null_pct=null_pct, distinct_pct=1.0)],
    )


def test_rolling_zscores_insufficient_history_is_none():
    means, stdevs, zs = zip(*rolling_zscores([10.0]))
    assert means == (None,)
    assert zs == (None,)


def test_rolling_zscores_flags_spike():
    values = [10.0, 11.0, 9.0, 10.0, 10.0, 100.0]
    triples = rolling_zscores(values, window=5)
    # first two points have < 2 prior values -> undefined
    assert triples[0] == (None, None, None)
    assert triples[1] == (None, None, None)
    # the spike's z-score should be large and positive
    spike_mean, spike_stdev, spike_z = triples[-1]
    assert spike_mean is not None and spike_stdev is not None
    assert spike_z is not None
    assert spike_z > 3.0


def test_rolling_zscores_zero_variance_is_undefined():
    triples = rolling_zscores([5.0, 5.0, 5.0, 5.0], window=5)
    # third point: prior=[5.0, 5.0] -> stdev 0 -> z undefined, mean reported
    assert triples[2][0] == 5.0
    assert triples[2][1] == 0.0
    assert triples[2][2] is None


def test_rolling_zscores_rejects_bad_window():
    with pytest.raises(ValueError):
        rolling_zscores([1.0, 2.0], window=0)


def test_detect_row_count_anomalies_stable_history_no_anomaly():
    history = [_snapshot("public.orders", rc, day) for day, rc in enumerate([100, 102, 98, 101, 99, 100])]

    results = detect_row_count_anomalies(history)

    assert len(results) == len(history)
    assert all(not r.is_anomaly for r in results)
    assert all(r.table_name == "public.orders" for r in results)
    assert all(r.metric_name == "row_count" for r in results)


def test_detect_row_count_anomalies_flags_drop():
    row_counts = [1000, 1010, 990, 1005, 995, 10]  # last point: a near-total data loss
    history = [_snapshot("public.orders", rc, day) for day, rc in enumerate(row_counts)]

    results = detect_row_count_anomalies(history, window=5, threshold=3.0)

    assert results[-1].is_anomaly is True
    assert results[-1].z_score is not None and results[-1].z_score < 0
    assert all(not r.is_anomaly for r in results[:-1])


def test_detect_row_count_anomalies_empty_history():
    assert detect_row_count_anomalies([]) == []


def test_detect_column_null_pct_anomalies_flags_spike():
    null_pcts = [0.01, 0.02, 0.01, 0.015, 0.02, 0.9]
    history = [_snapshot("public.orders", 100, day, null_pct=np) for day, np in enumerate(null_pcts)]

    results = detect_column_null_pct_anomalies(history, "id", window=5, threshold=3.0)

    assert results[-1].metric_name == "id.null_pct"
    assert results[-1].is_anomaly is True


def test_detect_metric_anomalies_skips_snapshots_missing_the_metric():
    history = [_snapshot("public.orders", 100, day) for day in range(3)]

    def extractor(m: TableMetrics):
        col = m.column("does_not_exist")
        return col.null_pct if col else None

    assert detect_metric_anomalies(history, metric_name="ghost", extractor=extractor) == []


def test_latest_anomalies_empty_when_not_anomalous():
    history = [_snapshot("public.orders", rc, day) for day, rc in enumerate([100, 101, 99, 100, 102])]
    results = detect_row_count_anomalies(history)
    assert latest_anomalies(results) == []


def test_latest_anomalies_returns_only_the_last_point_when_anomalous():
    row_counts = [1000, 1010, 990, 1005, 995, 10]
    history = [_snapshot("public.orders", rc, day) for day, rc in enumerate(row_counts)]
    results = detect_row_count_anomalies(history)

    flagged = latest_anomalies(results)

    assert len(flagged) == 1
    assert flagged[0].value == 10.0
