"""Anomaly detection: rolling z-score on metric history.

Looks at the sequence of historical snapshots for a table (as returned by
`SnapshotStore.history`) and flags points whose metric value deviates from
the rolling mean of the preceding window by more than a configurable number
of standard deviations. This is the simplest useful anomaly signal: it
needs no external baseline, works with as little as a handful of snapshots,
and plugs directly into `AlertThresholds.row_count_zscore_max` from
`observability.config` (a later roadmap item wires the two together into
the alert rule engine).
"""
from __future__ import annotations

import statistics
from typing import Callable, Optional

from pydantic import BaseModel

from .metrics import TableMetrics


class AnomalyResult(BaseModel):
    """The rolling z-score verdict for a single historical data point."""

    table_name: str
    metric_name: str
    collected_at: str
    value: float
    rolling_mean: Optional[float] = None
    rolling_stdev: Optional[float] = None
    z_score: Optional[float] = None
    is_anomaly: bool = False


def rolling_zscores(
    values: list[float], *, window: int = 5
) -> list[tuple[Optional[float], Optional[float], Optional[float]]]:
    """Compute a rolling (mean, stdev, z_score) triple for each value.

    For each index ``i``, the mean and stdev are computed over the
    ``window`` values immediately before it (``values[max(0, i-window):i]``),
    never including the point itself -- so a point can never influence its
    own z-score. Returns ``(None, None, None)`` wherever there are fewer
    than 2 prior points (not enough history to compute a spread) or the
    prior window has zero variance (stdev == 0), since a z-score is
    undefined in both cases.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    results: list[tuple[Optional[float], Optional[float], Optional[float]]] = []
    for i, value in enumerate(values):
        prior = values[max(0, i - window) : i]
        if len(prior) < 2:
            results.append((None, None, None))
            continue
        mean = statistics.fmean(prior)
        stdev = statistics.stdev(prior)
        if stdev == 0:
            results.append((mean, 0.0, None))
            continue
        z = (value - mean) / stdev
        results.append((mean, stdev, z))
    return results


def detect_metric_anomalies(
    history: list[TableMetrics],
    *,
    metric_name: str,
    extractor: Callable[[TableMetrics], Optional[float]],
    window: int = 5,
    threshold: float = 3.0,
) -> list[AnomalyResult]:
    """Run rolling z-score detection over ``history`` for one metric.

    ``history`` must be ordered oldest-first, matching
    `SnapshotStore.history`'s contract. ``extractor`` pulls the metric
    value out of each snapshot (e.g. ``lambda m: m.row_count``); snapshots
    where it returns ``None`` (metric not collected for that snapshot) are
    skipped entirely and do not count toward the rolling window.
    """
    if not history:
        return []

    table_name = history[0].table_name
    dated = [(m.collected_at.isoformat(), extractor(m)) for m in history]
    present = [(collected_at, value) for collected_at, value in dated if value is not None]
    if not present:
        return []

    values = [value for _, value in present]
    triples = rolling_zscores(values, window=window)

    results: list[AnomalyResult] = []
    for (collected_at, value), (mean, stdev, z) in zip(present, triples):
        is_anomaly = z is not None and abs(z) > threshold
        results.append(
            AnomalyResult(
                table_name=table_name,
                metric_name=metric_name,
                collected_at=collected_at,
                value=value,
                rolling_mean=mean,
                rolling_stdev=stdev,
                z_score=z,
                is_anomaly=is_anomaly,
            )
        )
    return results


def detect_row_count_anomalies(
    history: list[TableMetrics], *, window: int = 5, threshold: float = 3.0
) -> list[AnomalyResult]:
    """Convenience wrapper: rolling z-score anomaly detection on row_count."""
    return detect_metric_anomalies(
        history,
        metric_name="row_count",
        extractor=lambda m: float(m.row_count),
        window=window,
        threshold=threshold,
    )


def detect_column_null_pct_anomalies(
    history: list[TableMetrics], column_name: str, *, window: int = 5, threshold: float = 3.0
) -> list[AnomalyResult]:
    """Convenience wrapper: rolling z-score anomaly detection on a column's null_pct."""

    def _extract(m: TableMetrics) -> Optional[float]:
        col = m.column(column_name)
        return col.null_pct if col is not None else None

    return detect_metric_anomalies(
        history,
        metric_name=f"{column_name}.null_pct",
        extractor=_extract,
        window=window,
        threshold=threshold,
    )


def latest_anomalies(results: list[AnomalyResult]) -> list[AnomalyResult]:
    """Filter a results list down to just the most recent point, if anomalous.

    Convenience for callers (e.g. the alert rule engine, a later roadmap
    item) that only care whether the *newest* snapshot is anomalous, not
    the whole history.
    """
    if not results:
        return []
    return [results[-1]] if results[-1].is_anomaly else []
