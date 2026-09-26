"""Unit tests for observability.config."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from observability.config import AppConfig, load_config

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"


def test_load_example_config():
    config = load_config(EXAMPLE_CONFIG)

    assert isinstance(config, AppConfig)
    assert "warehouse" in config.connections
    assert config.connections["warehouse"].dialect == "postgresql"
    assert [t.name for t in config.tables] == ["public.orders", "public.customers"]
    assert config.tables[0].freshness_column == "created_at"
    assert config.metrics_store.path == "observability_metrics.db"
    assert config.alert_thresholds.null_pct_max == pytest.approx(0.2)


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_table_with_unknown_connection_is_rejected():
    raw = {
        "connections": {"warehouse": {"dialect": "postgresql"}},
        "tables": [{"name": "public.orders", "connection": "does_not_exist"}],
    }
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_alert_thresholds_defaults():
    config = AppConfig()
    assert config.alert_thresholds.null_pct_max == 0.2
    assert config.alert_thresholds.row_count_zscore_max == 3.0
    assert config.alert_thresholds.freshness_max_hours == 24.0
