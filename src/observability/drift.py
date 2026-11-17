"""Data drift check between two snapshots: Population Stability Index (PSI).

Compares two `TableMetrics` snapshots of the same table -- typically the
latest snapshot against a baseline pulled from `SnapshotStore.history` --
and flags columns whose distribution has shifted beyond a configurable
PSI threshold. Unlike `observability.anomaly` (which scans a *sequence* of
snapshots for one metric's rolling z-score), this module compares exactly
two snapshots at a time, the same shape as `observability.schema_change`
comparing two `TableSchema` captures.

PSI needs a binned distribution to compare, but a snapshot only carries a
scalar percentage per column (no raw values), so each percentage is
treated as a two-bucket distribution: for null_pct, bucket 1 is "null"
and bucket 0 is "not null"; for distinct_pct, bucket 1 is "distinct
value" and bucket 0 is "repeated value". This keeps drift detection
working from the aggregate metrics the toolkit already collects, with no
extra querying of the source table. A full Kolmogorov-Smirnov test needs
the underlying per-row samples, which snapshots don't retain, so it is
left out until (if ever) the toolkit stores raw samples.
"""
from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field

from .metrics import TableMetrics

_EPSILON = 1e-6  # avoids log(0) / division-by-zero when a bucket is empty in one snapshot


def psi(expected: list[float], actual: list[float]) -> float:
    """Population Stability Index between two parallel bucketed distributions.

    ``expected`` and ``actual`` are proportions for the same buckets, in
    the same order (each should sum to ~1.0, as produced by
    `_pct_to_buckets`). Each bucket is clamped away from 0 with
    `_EPSILON` so an empty bucket in either snapshot never produces a
    division by zero or an undefined log. Higher PSI means a bigger
    shift; 0.0 means the two distributions are identical.
    """
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same number of buckets")
    total = 0.0
    for expected_pct, actual_pct in zip(expected, actual):
        e = max(expected_pct, _EPSILON)
        a = max(actual_pct, _EPSILON)
        total += (a - e) * math.log(a / e)
    return total


def _pct_to_buckets(pct: float) -> list[float]:
    """Represent a single percentage as a two-bucket distribution [hit, miss]."""
    clamped = min(max(pct, 0.0), 1.0)
    return [clamped, 1.0 - clamped]


def psi_severity(psi_value: float) -> str:
    """Classify a PSI value using the conventional industry thresholds.

    < 0.1 is treated as no meaningful shift, 0.1-0.25 as a moderate
    shift worth watching, and >= 0.25 as a significant shift.
    """
    if psi_value < 0.1:
        return "none"
    if psi_value < 0.25:
        return "moderate"
    return "significant"


class ColumnDrift(BaseModel):
    """PSI drift for one column's metric between a baseline and current snapshot."""

    column: str
    metric: str  # "null_pct" or "distinct_pct"
    baseline_value: float
    current_value: float
    psi: float
    severity: str


class DriftResult(BaseModel):
    """The drift comparison between two consecutive snapshots of one table."""

    table_name: str
    baseline_collected_at: str
    current_collected_at: str
    columns: list[ColumnDrift] = Field(default_factory=list)

    def drifted_columns(self, *, threshold: float = 0.1) -> list[ColumnDrift]:
        """Columns whose PSI is at or above ``threshold`` (default: the conventional 'moderate' cutoff)."""
        return [c for c in self.columns if c.psi >= threshold]

    def has_drift(self, *, threshold: float = 0.1) -> bool:
        """True if any column's metric drifted at or above ``threshold``."""
        return bool(self.drifted_columns(threshold=threshold))


def detect_drift(baseline: TableMetrics, current: TableMetrics) -> DriftResult:
    """Compare ``baseline`` and ``current`` snapshots of the same table.

    Only columns present in both snapshots are compared (a column added
    or removed between snapshots is `observability.schema_change`'s
    concern, not drift's). Raises ValueError if the two snapshots belong
    to different tables -- comparing across tables is almost certainly a
    caller bug, mirroring `schema_change.diff_schemas`.
    """
    if baseline.table_name != current.table_name:
        raise ValueError(
            f"cannot compare drift across different tables: {baseline.table_name!r} vs {current.table_name!r}"
        )

    baseline_cols = {c.name: c for c in baseline.columns}
    current_cols = {c.name: c for c in current.columns}
    shared = sorted(set(baseline_cols) & set(current_cols))

    results: list[ColumnDrift] = []
    for name in shared:
        base_col = baseline_cols[name]
        curr_col = current_cols[name]
        for metric_name, base_value, curr_value in (
            ("null_pct", base_col.null_pct, curr_col.null_pct),
            ("distinct_pct", base_col.distinct_pct, curr_col.distinct_pct),
        ):
            value = psi(_pct_to_buckets(base_value), _pct_to_buckets(curr_value))
            results.append(
                ColumnDrift(
                    column=name,
                    metric=metric_name,
                    baseline_value=base_value,
                    current_value=curr_value,
                    psi=value,
                    severity=psi_severity(value),
                )
            )

    return DriftResult(
        table_name=current.table_name,
        baseline_collected_at=baseline.collected_at.isoformat(),
        current_collected_at=current.collected_at.isoformat(),
        columns=results,
    )


def detect_drift_history(history: list[TableMetrics]) -> list[DriftResult]:
    """Run `detect_drift` over every consecutive pair in ``history``.

    ``history`` must be ordered oldest-first, matching
    `SnapshotStore.history`'s contract. Returns one `DriftResult` per
    transition -- ``len(history) - 1`` results -- so a lone first
    snapshot never produces a result on its own, mirroring
    `schema_change.detect_schema_changes`.
    """
    if len(history) < 2:
        return []
    return [detect_drift(old, new) for old, new in zip(history[:-1], history[1:])]


def latest_drift(results: list[DriftResult], *, threshold: float = 0.1) -> Optional[DriftResult]:
    """Return the most recent transition's DriftResult, if it drifted.

    Convenience for callers (e.g. the alert rule engine, a later roadmap
    item) that only care whether the *newest* snapshot drifted, not the
    whole history -- mirrors `observability.anomaly.latest_anomalies` and
    `observability.schema_change.latest_schema_change`.
    """
    if not results:
        return None
    return results[-1] if results[-1].has_drift(threshold=threshold) else None
