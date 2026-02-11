import pathlib

import pytest
from flyte.io import DataFrame
from flyte.models import SerializationContext
from google.cloud import bigquery

from flyteplugins.bigquery.task import BigQueryConfig, BigQueryTask


@pytest.fixture
def serialization_context() -> SerializationContext:
    return SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )


class TestBigQueryConfig:
    def test_bigquery_config_creation_minimal(self):
        config = BigQueryConfig(ProjectID="test-project")
        assert config.ProjectID == "test-project"
        assert config.Location is None
        assert config.QueryJobConfig is None

    def test_bigquery_config_creation_with_location(self):
        config = BigQueryConfig(ProjectID="test-project", Location="US")
        assert config.ProjectID == "test-project"
        assert config.Location == "US"
        assert config.QueryJobConfig is None

    def test_bigquery_config_creation_with_job_config(self):
        job_config = bigquery.QueryJobConfig(use_query_cache=False)
        config = BigQueryConfig(ProjectID="test-project", Location="EU", QueryJobConfig=job_config)
        assert config.ProjectID == "test-project"
        assert config.Location == "EU"
        assert config.QueryJobConfig == job_config


class TestBigQueryTask:
    def test_bigquery_task_creation_minimal(self):
        config = BigQueryConfig(ProjectID="test-project")
        task = BigQueryTask(name="test_task", query_template="SELECT * FROM table", plugin_config=config)

        assert task.name == "test_task"
        assert task.query_template == "SELECT * FROM table"
        assert task.plugin_config == config
        assert task.output_dataframe_type is None
        assert task.task_type == "bigquery_query_job_task"

    def test_bigquery_task_creation_with_inputs(self):
        config = BigQueryConfig(ProjectID="test-project")
        inputs = {"user_id": int, "start_date": str}

        task = BigQueryTask(
            name="test_task",
            query_template="SELECT * FROM table WHERE user_id = {{ .user_id }}",
            plugin_config=config,
            inputs=inputs,
        )

        assert task.name == "test_task"
        assert task.interface.inputs is not None
        assert "user_id" in [name for name, _ in task.interface.inputs.items()]
        assert "start_date" in [name for name, _ in task.interface.inputs.items()]

    def test_bigquery_task_creation_with_output_dataframe(self):
        config = BigQueryConfig(ProjectID="test-project")

        task = BigQueryTask(
            name="test_task",
            query_template="SELECT * FROM table",
            plugin_config=config,
            output_dataframe_type=DataFrame,
        )

        assert task.output_dataframe_type == DataFrame
        assert task.interface.outputs is not None
        assert "results" in task.interface.outputs

    def test_bigquery_task_query_template_normalization(self):
        config = BigQueryConfig(ProjectID="test-project")

        # Test query with multiple spaces, tabs, and newlines
        query_with_whitespace = """
        SELECT
            col1, col2, col3
        FROM table
        WHERE condition = 1
        """

        task = BigQueryTask(name="test_task", query_template=query_with_whitespace, plugin_config=config)

        # Should normalize whitespace to single spaces
        expected = "SELECT col1, col2, col3 FROM table WHERE condition = 1"
        assert task.query_template == expected

    def test_bigquery_task_custom_config_minimal(self, serialization_context):
        config = BigQueryConfig(ProjectID="test-project")
        task = BigQueryTask(name="test_task", query_template="SELECT * FROM table", plugin_config=config)

        custom_config = task.custom_config(serialization_context)

        expected = {"Location": None, "ProjectID": "test-project", "Domain": "test-domain"}
        assert custom_config == expected

    def test_bigquery_task_custom_config_with_location(self, serialization_context):
        config = BigQueryConfig(ProjectID="test-project", Location="US")
        task = BigQueryTask(name="test_task", query_template="SELECT * FROM table", plugin_config=config)

        custom_config = task.custom_config(serialization_context)

        expected = {"Location": "US", "ProjectID": "test-project", "Domain": "test-domain"}
        assert custom_config == expected

    def test_bigquery_task_custom_config_with_job_config(self, serialization_context):
        job_config = bigquery.QueryJobConfig(use_query_cache=False, use_legacy_sql=True, maximum_bytes_billed=1000000)
        config = BigQueryConfig(ProjectID="test-project", Location="EU", QueryJobConfig=job_config)
        task = BigQueryTask(name="test_task", query_template="SELECT * FROM table", plugin_config=config)

        custom_config = task.custom_config(serialization_context)

        # Should include both basic config and job config properties
        assert custom_config["Location"] == "EU"
        assert custom_config["ProjectID"] == "test-project"
        assert custom_config["Domain"] == "test-domain"
        assert custom_config["useQueryCache"] is False
        assert custom_config["useLegacySql"] is True
        assert custom_config["maximumBytesBilled"] == "1000000"

    def test_bigquery_task_sql_method(self, serialization_context):
        config = BigQueryConfig(ProjectID="test-project")
        query = "SELECT * FROM dataset.table WHERE id = {{ .user_id }}"

        task = BigQueryTask(name="test_task", query_template=query, plugin_config=config)

        sql_proto = task.sql(serialization_context)

        assert sql_proto is not None
        assert sql_proto.statement == query
        assert sql_proto.dialect == 1  # ANSI dialect

    def test_bigquery_task_with_complex_inputs_and_outputs(self):
        config = BigQueryConfig(ProjectID="test-project", Location="US")
        inputs = {"user_id": int, "start_date": str, "end_date": str, "limit": int}

        task = BigQueryTask(
            name="analytics_task",
            query_template="""
                SELECT user_id, COUNT(*) as event_count
                FROM events
                WHERE user_id = {{ .user_id }}
                  AND date >= '{{ .start_date }}'
                  AND date <= '{{ .end_date }}'
                GROUP BY user_id
                LIMIT {{ .limit }}
            """,
            plugin_config=config,
            inputs=inputs,
            output_dataframe_type=DataFrame,
        )

        assert task.name == "analytics_task"
        assert task.output_dataframe_type == DataFrame
        assert len([name for name, _ in task.interface.inputs.items()]) == 4
        assert "results" in task.interface.outputs

        # Check that query template was normalized
        assert "SELECT user_id, COUNT(*) as event_count FROM events" in task.query_template
        assert "GROUP BY user_id LIMIT" in task.query_template

    def test_bigquery_task_type_constant(self):
        config = BigQueryConfig(ProjectID="test-project")
        task = BigQueryTask(name="test_task", query_template="SELECT 1", plugin_config=config)

        assert task._TASK_TYPE == "bigquery_query_job_task"
        assert task.task_type == "bigquery_query_job_task"

    def test_bigquery_task_empty_query_template(self):
        config = BigQueryConfig(ProjectID="test-project")

        task = BigQueryTask(name="test_task", query_template="", plugin_config=config)

        assert task.query_template == ""

    def test_bigquery_task_query_template_with_only_whitespace(self):
        config = BigQueryConfig(ProjectID="test-project")

        task = BigQueryTask(name="test_task", query_template="   \n\t   ", plugin_config=config)

        assert task.query_template == ""

    def test_bigquery_task_custom_config_with_google_application_credentials(self, serialization_context):
        """Test that google_application_credentials secret key is correct (no trailing colon)."""
        config = BigQueryConfig(ProjectID="test-project", Location="US")
        secret_name = "my-gcp-credentials-secret"

        task = BigQueryTask(
            name="test_task",
            query_template="SELECT * FROM table",
            plugin_config=config,
            google_application_credentials=secret_name,
        )

        custom_config = task.custom_config(serialization_context)

        # Verify secrets dict exists and has correct key (no trailing colon)
        assert "secrets" in custom_config
        assert "google_application_credentials" in custom_config["secrets"]
        assert "google_application_credentials:" not in custom_config["secrets"]
        assert custom_config["secrets"]["google_application_credentials"] == secret_name

    def test_bigquery_task_custom_config_without_google_application_credentials(self, serialization_context):
        """Test that secrets is not included when google_application_credentials is not set."""
        config = BigQueryConfig(ProjectID="test-project", Location="US")

        task = BigQueryTask(
            name="test_task",
            query_template="SELECT * FROM table",
            plugin_config=config,
        )

        custom_config = task.custom_config(serialization_context)

        # Verify secrets dict is not included when no credentials are specified
        assert "secrets" not in custom_config
