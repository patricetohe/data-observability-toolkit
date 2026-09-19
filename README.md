# Data Observability Toolkit

A lightweight data observability toolkit: profiles tables over time, detects schema changes and data drift, flags anomalies against historical baselines, and surfaces it all on a small dashboard.

## Architecture

```
[Source tables] --> Profiler (row count, nulls %, distinct %, min/max, freshness)
                                    |
                                    v
                    Historical metric store (SQLite)
                                    |
                    Anomaly detection (rolling z-score) + schema-change + drift checks
                                    |
                                    v
                            Streamlit dashboard + alert rules
```

- **src/observability/** — profiler, anomaly detector, schema tracker, alerting engine
- **dashboard/** — Streamlit app
- **config/** — table connections and alert threshold definitions
- **tests/** — unit tests

## Status

Built incrementally — see [ROADMAP.md](ROADMAP.md) for the current backlog.

## Stack

Python, pandas, SQLite (metrics store), Streamlit, pydantic, SQLGlot (lightweight lineage), APScheduler.
