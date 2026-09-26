"""Configuration loading and validation for the data observability toolkit.

Defines the pydantic models describing a toolkit run (which tables to
profile, where the metrics history is stored, and alert thresholds) and a
small loader that reads a YAML file into a validated `AppConfig` instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ConnectionConfig(BaseModel):
    """How to reach the database holding the tables to observe."""

    dialect: str = Field(..., description="SQLAlchemy dialect, e.g. 'postgresql', 'sqlite', 'duckdb'.")
    url_env_var: Optional[str] = Field(
        default=None,
        description="Name of the environment variable holding the full SQLAlchemy connection URL.",
    )
    url: Optional[str] = Field(
        default=None,
        description="Literal connection URL. Prefer url_env_var for anything with credentials.",
    )

    @field_validator("dialect")
    @classmethod
    def _dialect_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dialect must not be blank")
        return value.strip().lower()


class TableConfig(BaseModel):
    """A single table to profile."""

    name: str = Field(..., description="Fully qualified table name, e.g. 'public.orders'.")
    freshness_column: Optional[str] = Field(
        default=None, description="Timestamp column used to compute freshness metrics."
    )
    connection: str = Field(..., description="Key into AppConfig.connections identifying the source DB.")

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("table name must not be blank")
        return value.strip()


class AlertThresholds(BaseModel):
    """Threshold-based alert rule configuration (rolling z-score based)."""

    null_pct_max: float = Field(default=0.2, ge=0.0, le=1.0)
    row_count_zscore_max: float = Field(default=3.0, gt=0.0)
    freshness_max_hours: float = Field(default=24.0, gt=0.0)


class MetricsStoreConfig(BaseModel):
    """Where historical profiling snapshots are persisted."""

    path: str = Field(default="observability_metrics.db", description="SQLite file path for metric history.")


class AppConfig(BaseModel):
    """Top-level configuration for a toolkit run."""

    connections: dict[str, ConnectionConfig] = Field(default_factory=dict)
    tables: list[TableConfig] = Field(default_factory=list)
    metrics_store: MetricsStoreConfig = Field(default_factory=MetricsStoreConfig)
    alert_thresholds: AlertThresholds = Field(default_factory=AlertThresholds)

    @field_validator("tables")
    @classmethod
    def _tables_reference_known_connection(cls, tables: list[TableConfig], info) -> list[TableConfig]:
        connections = info.data.get("connections", {}) or {}
        unknown = [t.name for t in tables if t.connection not in connections]
        if unknown:
            raise ValueError(
                f"tables reference undefined connections: {unknown} "
                f"(known connections: {sorted(connections)})"
            )
        return tables


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file into an `AppConfig`.

    Raises FileNotFoundError if the path does not exist, and
    pydantic.ValidationError if the content does not match the schema.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return AppConfig.model_validate(raw)
