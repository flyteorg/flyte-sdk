<div align="center">

# 🔗 Flyte Connectors Plugin

[![PyPI version](https://badge.fury.io/py/flyteplugins-connectors.svg)](https://badge.fury.io/py/flyteplugins-connectors)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/flyteorg/flyte/workflows/tests/badge.svg)](https://github.com/flyteorg/flyte/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.flyte.org)

**🚀 Seamlessly connect Flyte workflows to external data sources and services**

*Build powerful data pipelines with native integrations to popular cloud services*

</div>

## 🚀 Quick Start

### Installation


```bash
pip install --pre flyteplugins-connectors[bigquery]  # Install BigQuery connector
```



### BigQuery Integration

Execute SQL queries on BigQuery and seamlessly integrate results into your Flyte workflows:

```python
from flyteplugins.connectors.bigquery.task import BigQueryConfig, BigQueryTask
import flyte
from flyte.io import DataFrame

# Configure your BigQuery connection
config = BigQueryConfig(
    ProjectID="your-gcp-project",
    Location="US"  # Optional: specify region
)

# Create a task environment
env = flyte.TaskEnvironment(name="analytics_env")

# Define your BigQuery task
analytics_task = BigQueryTask(
    name="user_analytics",
    inputs={
        "user_id": int,
        "start_date": str,
        "end_date": str
    },
    output_dataframe_type=DataFrame,
    plugin_config=config,
    query_template="""
        SELECT
            user_id,
            COUNT(*) as event_count,
            MAX(timestamp) as last_activity
        FROM events
        WHERE user_id = {{ .user_id }}
          AND DATE(timestamp) BETWEEN '{{ .start_date }}' AND '{{ .end_date }}'
        GROUP BY user_id
    """
)

env.from_task(analytics_task)

# Run your workflow
if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.with_runcontext(mode="remote").run(
        analytics_task,
        user_id=12345,
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    print(f"Workflow URL: {result.url}")
```

## 📚 Available Connectors

| Connector | Status | Description | Use Cases |
|-----------|--------|-------------|-----------|
| 🔷 **BigQuery** | ✅ Stable | Google Cloud data warehouse | Analytics, ML training, reporting |
| 🔗 **More Coming Soon** | 🚧 | Additional connectors in development | - |

## 🧪 Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all connector tests
pytest plugins/connectors/tests/ -v

# Run specific connector tests
pytest plugins/connectors/tests/test_bigquery.py -v

# Run with coverage
pytest plugins/connectors/tests/ --cov=flyteplugins.connectors --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-connector`
3. **Write tests** for your changes
4. **Ensure tests pass**: `pytest plugins/connectors/tests/`
5. **Submit a pull request**

### Adding a New Connector

1. Create your connector module in `src/flyteplugins/connectors/`
2. Implement the `TaskTemplate` interface
3. Add comprehensive tests in `tests/`
4. Update this README with examples
5. Add example usage in `examples/connectors/`

## 🔧 Requirements

- **Python**: 3.10+
- **Flyte**: Latest version
- **Dependencies**: See `pyproject.toml` for full requirements


## 🆘 Support

- **📬 Community**: [Flyte Slack](https://slack.flyte.org/)
- **🐛 Issues**: [GitHub Issues](https://github.com/flyteorg/flyte-sdk/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/flyteorg/flyte/discussions)
- **📚 Documentation**: [docs.flyte.org](https://docs.flyte.org)

---

<div align="center">

**Made with ❤️ by the Flyte Community**

[⭐ Star us on GitHub](https://github.com/flyteorg/flyte) • [🐦 Follow us on Twitter](https://twitter.com/flyteorg) • [💼 LinkedIn](https://linkedin.com/company/flyte-org)

</div>
