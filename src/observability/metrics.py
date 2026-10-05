"""Metrics model for the data observability toolkit.

Defines the data structures used to represent a single profiling snapshot
of a table: row count, per-column null %, distinct %, min/max, and table
freshness. This module only defines and validates the shape of a snapshot;
the profiler module (next roadmap item) is responsible for populating it by
scanning a live table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ColumnMetrics(BaseModel):
    """Profiling metrics for a single column within one snapshot."""

    name: str
    null_pct: float = Field(..., ge=0.0, le=1.0, description="Fraction of rows where this column is NULL.")
    distinct_pct: float = Field(..., ge=0.0, le=1.0, description="Fraction of rows with a distinct value.")
    min_value: Optional[str] = Field(
        default=None, description="Minimum value, stringified so it is comparable across source dialects."
    )
    max_value: Optional[str] = Field(
        default=None, description="Maximum value, stringified so it is comparable across source dialects."
    )

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("column name must not be blank")
        return value.strip()


class TableMetrics(BaseModel):
    """A single profiling snapshot for one table at a point in time."""

    table_name: str
    row_count: int = Field(..., ge=0)
    columns: list[ColumnMetrics] = Field(default_factory=list)
    freshness_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds between collection time and the most recent value in the "
            "table's freshness column, if one is configured for this table."
        ),
    )
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("table_name")
    @classmethod
    def _table_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("table_name must not be blank")
        return value.strip()

    def column(self, name: str) -> Optional[ColumnMetrics]:
        """Return the metrics for a given column name, or None if absent."""
        return next((c for c in self.columns if c.name == name), None)

    def is_stale(self, max_age_seconds: float) -> bool:
        """True if freshness is known and exceeds the given max age.

        Returns False (not stale) when freshness was never computed, since
        "unknown" should not be treated the same as "known to be stale".
        """
        if self.freshness_seconds is None:
            return False
        return self.freshness_seconds > max_age_seconds

    def to_flat_dict(self) -> dict:
        """Flatten this snapshot into a single-row dict.

        Column-level metrics are namespaced as ``"<column>.<metric>"`` so an
        entire snapshot can be stored or compared as one wide row. Used by
        the historical metric snapshot store (a later roadmap item).
        """
        flat: dict = {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "freshness_seconds": self.freshness_seconds,
            "collected_at": self.collected_at.isoformat(),
        }
        for col in self.columns:
            flat[f"{col.name}.null_pct"] = col.null_pct
            flat[f"{col.name}.distinct_pct"] = col.distinct_pct
            flat[f"{col.name}.min"] = col.min_value
            flat[f"{col.name}.max"] = col.max_value
        return flat
