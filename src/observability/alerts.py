"""Alert rule engine: threshold-based rules driven by `AppConfig.alert_thresholds`.

This is the "last mile" that ties every detector module together into
actionable alerts:

- `observability.metrics.TableMetrics` -> per-column null rate, freshness
- `observability.anomaly` -> rolling z-score anomalies on row_count
- `observability.schema_change.SchemaChange` -> added/removed/retyped columns
- `observability.drift.DriftResult` -> per-column PSI drift

Each ``evaluate_*`` function checks one rule and returns the `Alert`s it
raised (often zero). `evaluate_all` is the entry point a CLI or scheduler
(later roadmap items) calls once per table per run, combining every rule
into one flat, ordered list. Alert *delivery* (webhook, local log, etc.)
is a separate, later roadmap item -- this module only decides whether an
alert fires and what it says.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .anomaly import detect_row_count_anomalies, latest_anomalies
from .config import AlertThresholds
from .drift import DriftResult
from .metrics import TableMetrics
from .schema_change import SchemaChange


class AlertSeverity(str, Enum):
    """How urgently an alert needs attention."""

    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """A single raised alert: which table, which rule, and why."""

    table_name: str
    rule: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def evaluate_null_pct(metrics: TableMetrics, thresholds: AlertThresholds) -> list[Alert]:
    """Flag every column whose null rate exceeds ``thresholds.null_pct_max``."""
    alerts: list[Alert] = []
    for col in metrics.columns:
        if col.null_pct > thresholds.null_pct_max:
            alerts.append(
                Alert(
                    table_name=metrics.table_name,
                    rule="null_pct_max",
                    severity=AlertSeverity.WARNING,
                    message=(
                        f"column '{col.name}' null rate is {col.null_pct:.1%}, "
                        f"exceeding the {thresholds.null_pct_max:.1%} max"
                    ),
                )
            )
    return alerts


def evaluate_freshness(metrics: TableMetrics, thresholds: AlertThresholds) -> list[Alert]:
    """Flag the table if its freshness exceeds ``thresholds.freshness_max_hours``.

    Uses `TableMetrics.is_stale`, which already treats "freshness never
    computed" as not stale -- so a table with no freshness column
    configured never raises this alert.
    """
    max_age_seconds = thresholds.freshness_max_hours * 3600.0
    if not metrics.is_stale(max_age_seconds):
        return []
    hours = (metrics.freshness_seconds or 0.0) / 3600.0
    return [
        Alert(
            table_name=metrics.table_name,
            rule="freshness_max_hours",
            severity=AlertSeverity.CRITICAL,
            message=f"table is {hours:.1f}h stale, exceeding the {thresholds.freshness_max_hours}h max",
        )
    ]


def evaluate_row_count_anomaly(history: list[TableMetrics], thresholds: AlertThresholds) -> list[Alert]:
    """Flag the table if its latest row_count is a rolling z-score anomaly.

    Reuses `observability.anomaly.detect_row_count_anomalies`, running it
    with ``thresholds.row_count_zscore_max`` as the z-score cutoff so the
    YAML config is the single source of truth for what counts as
    anomalous, and `latest_anomalies` so only the newest snapshot (not the
    whole history) can raise an alert.
    """
    if not history:
        return []
    results = detect_row_count_anomalies(history, threshold=thresholds.row_count_zscore_max)
    flagged = latest_anomalies(results)
    return [
        Alert(
            table_name=r.table_name,
            rule="row_count_zscore_max",
            severity=AlertSeverity.WARNING,
            message=(
                f"row_count z-score is {r.z_score:.2f}, exceeding the "
                f"{thresholds.row_count_zscore_max} max "
                f"(value={r.value:.0f}, rolling mean={r.rolling_mean:.1f})"
            ),
        )
        for r in flagged
    ]


def evaluate_schema_change(schema_change: Optional[SchemaChange]) -> list[Alert]:
    """Flag any added/removed/retyped column in ``schema_change``.

    Not threshold-based -- any real schema change is worth surfacing as
    a critical alert as-is, so there is no `AlertThresholds` field for
    it (unlike the other rules here).
    """
    if schema_change is None or not schema_change.has_changes:
        return []

    parts = []
    if schema_change.added_columns:
        parts.append(f"added={schema_change.added_columns}")
    if schema_change.removed_columns:
        parts.append(f"removed={schema_change.removed_columns}")
    if schema_change.type_changes:
        type_changes = ", ".join(
            f"{c.column}:{c.old_dtype}->{c.new_dtype}" for c in schema_change.type_changes
        )
        parts.append(f"type_changes=[{type_changes}]")

    return [
        Alert(
            table_name=schema_change.table_name,
            rule="schema_change",
            severity=AlertSeverity.CRITICAL,
            message="schema changed: " + "; ".join(parts),
        )
    ]


def evaluate_drift(drift_result: Optional[DriftResult], *, psi_threshold: float = 0.1) -> list[Alert]:
    """Flag every column whose PSI is at or above ``psi_threshold`` in ``drift_result``.

    A column PSI's own `ColumnDrift.severity` ("moderate" vs
    "significant", from `observability.drift.psi_severity`) decides the
    `AlertSeverity`: only a "significant" shift is raised as critical.
    """
    if drift_result is None:
        return []
    drifted = drift_result.drifted_columns(threshold=psi_threshold)
    return [
        Alert(
            table_name=drift_result.table_name,
            rule="data_drift_psi",
            severity=AlertSeverity.CRITICAL if c.severity == "significant" else AlertSeverity.WARNING,
            message=f"column '{c.column}' {c.metric} PSI is {c.psi:.3f} ({c.severity} shift)",
        )
        for c in drifted
    ]


def evaluate_all(
    *,
    latest_metrics: TableMetrics,
    metrics_history: list[TableMetrics],
    thresholds: AlertThresholds,
    schema_change: Optional[SchemaChange] = None,
    drift_result: Optional[DriftResult] = None,
    drift_psi_threshold: float = 0.1,
) -> list[Alert]:
    """Run every threshold-based rule for one table and combine the results.

    Intended as the single entry point a CLI/scheduler (later roadmap
    items) calls once per table per run, wiring `AppConfig.alert_thresholds`
    together with whatever the profiler, anomaly detector, schema-change
    detector and drift check produced for that table. ``metrics_history``
    must be ordered oldest-first and include ``latest_metrics`` as its
    last element, matching `SnapshotStore.history`'s contract.
    """
    alerts: list[Alert] = []
    alerts.extend(evaluate_null_pct(latest_metrics, thresholds))
    alerts.extend(evaluate_freshness(latest_metrics, thresholds))
    alerts.extend(evaluate_row_count_anomaly(metrics_history, thresholds))
    alerts.extend(evaluate_schema_change(schema_change))
    alerts.extend(evaluate_drift(drift_result, psi_threshold=drift_psi_threshold))
    return alerts
