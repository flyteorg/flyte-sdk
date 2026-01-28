# Flyte Connectors

Connect Flyte workflows to external data sources and services.

## Installation

```bash
pip install --pre flyteplugins-connectors[<connector>]
```

## Available Connectors

| Connector | Description |
|-----------|-------------|
| BigQuery | Google Cloud data warehouse |
| Databricks | Databricks job execution |
| Snowflake | Snowflake SQL query execution |

## Testing

```bash
pytest plugins/connectors/tests/ -v
```

## Requirements

- Python 3.10+
- See `pyproject.toml` for full dependencies
