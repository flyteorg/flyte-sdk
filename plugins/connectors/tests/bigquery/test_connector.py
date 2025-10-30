import datetime
from unittest.mock import MagicMock, patch

import pytest
from flyte.io import DataFrame
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.interface_pb2 import Variable, VariableMap
from flyteidl2.core.tasks_pb2 import Sql, TaskTemplate
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from google.protobuf import struct_pb2

from flyteplugins.connectors.bigquery.connector import (
    BigQueryConnector,
    BigQueryMetadata,
    pythonTypeToBigQueryType,
)


def test_type_mapping_complete():
    """Test that all expected Python types are mapped to BigQuery types."""
    assert pythonTypeToBigQueryType[list] == "ARRAY"
    assert pythonTypeToBigQueryType[bool] == "BOOL"
    assert pythonTypeToBigQueryType[bytes] == "BYTES"
    assert pythonTypeToBigQueryType[datetime.datetime] == "DATETIME"
    assert pythonTypeToBigQueryType[float] == "FLOAT64"
    assert pythonTypeToBigQueryType[int] == "INT64"
    assert pythonTypeToBigQueryType[str] == "STRING"


def test_metadata_creation():
    """Test creating BigQueryMetadata instance."""
    metadata = BigQueryMetadata(job_id="test-job-123", project="test-project", location="US")
    assert metadata.job_id == "test-job-123"
    assert metadata.project == "test-project"
    assert metadata.location == "US"


class TestBigQueryConnector:
    @pytest.fixture
    def connector(self):
        """Create a BigQueryConnector instance."""
        return BigQueryConnector()

    @pytest.fixture
    def task_template_minimal(self):
        """Create a minimal task template for testing."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT 1", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["ProjectID"] = "test-project"
        custom["Location"] = "US"
        template.custom.CopyFrom(custom)

        return template

    @pytest.fixture
    def task_template_with_inputs(self):
        """Create a task template with input variables."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT * FROM table WHERE id = @user_id", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        # Add input variables
        int_type = LiteralType()
        int_type.simple = SimpleType.INTEGER
        user_id_var = Variable(type=int_type)

        variables = VariableMap()
        variables.variables["user_id"].CopyFrom(user_id_var)
        template.interface.inputs.CopyFrom(variables)

        custom = struct_pb2.Struct()
        custom["ProjectID"] = "test-project"
        custom["Location"] = "US"
        custom["Domain"] = "test-domain"
        template.custom.CopyFrom(custom)

        return template

    def test_connector_class_attributes(self, connector):
        """Test that the connector has the correct class attributes."""
        assert connector.name == "Bigquery Connector"
        assert connector.task_type_name == "bigquery_query_job_task"
        assert connector.metadata_type == BigQueryMetadata

    @pytest.mark.asyncio
    async def test_create_minimal(self, connector, task_template_minimal):
        """Test creating a BigQuery job without inputs."""
        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_query_job = MagicMock()
            mock_query_job.job_id = "job-123"
            mock_client.query.return_value = mock_query_job

            metadata = await connector.create(task_template_minimal, inputs=None)

            assert metadata.job_id == "job-123"
            assert metadata.project == "test-project"
            assert metadata.location == "US"

            # Verify client was created with correct parameters
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["location"] == "US"
            assert "Flyte/1.0.0" in call_kwargs["client_info"].user_agent

            # Verify query was called without job_config
            mock_client.query.assert_called_once_with("SELECT 1", job_config=None)

    @pytest.mark.asyncio
    async def test_create_with_inputs(self, connector, task_template_with_inputs):
        """Test creating a BigQuery job with input parameters."""
        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_query_job = MagicMock()
            mock_query_job.job_id = "job-456"
            mock_client.query.return_value = mock_query_job

            inputs = {"user_id": 12345}
            metadata = await connector.create(task_template_with_inputs, inputs=inputs)

            assert metadata.job_id == "job-456"
            assert metadata.project == "test-project"
            assert metadata.location == "US"

            # Verify query was called with job_config
            mock_client.query.assert_called_once()
            call_args = mock_client.query.call_args
            assert call_args[0][0] == "SELECT * FROM table WHERE id = @user_id"
            job_config = call_args[1]["job_config"]
            assert job_config is not None
            assert len(job_config.query_parameters) == 1
            assert job_config.query_parameters[0].name == "user_id"
            assert job_config.query_parameters[0].value == 12345

    @pytest.mark.asyncio
    async def test_create_with_domain_in_user_agent(self, connector, task_template_with_inputs):
        """Test that domain is included in user agent when present."""
        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_query_job = MagicMock()
            mock_query_job.job_id = "job-789"
            mock_client.query.return_value = mock_query_job

            await connector.create(task_template_with_inputs, inputs=None)

            call_kwargs = mock_client_class.call_args[1]
            user_agent = call_kwargs["client_info"].user_agent
            assert "Flyte/1.0.0" in user_agent
            assert "GPN:Union;test-domain" in user_agent

    @pytest.mark.asyncio
    async def test_get_succeeded_with_destination(self, connector):
        """Test getting a successful BigQuery job with destination table."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US")

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock successful job with destination
            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "DONE"
            mock_job.destination = MagicMock()
            mock_job.destination.project = "output-project"
            mock_job.destination.dataset_id = "output_dataset"
            mock_job.destination.table_id = "output_table"

            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.connectors.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.SUCCEEDED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.SUCCEEDED
                assert resource.message == "DONE"
                assert len(resource.log_links) == 1
                assert resource.log_links[0].name == "BigQuery Console"
                assert "console.cloud.google.com/bigquery" in resource.log_links[0].uri
                assert "test-project" in resource.log_links[0].uri
                assert "job-123" in resource.log_links[0].uri

                # Verify outputs contain DataFrame with correct URI
                assert resource.outputs is not None
                assert "results" in resource.outputs
                assert isinstance(resource.outputs["results"], DataFrame)
                assert resource.outputs["results"].uri == "bq://output-project:output_dataset.output_table"

    @pytest.mark.asyncio
    async def test_get_succeeded_without_destination(self, connector):
        """Test getting a successful BigQuery job without destination table."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US")

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock successful job without destination
            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "DONE"
            mock_job.destination = None

            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.connectors.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.SUCCEEDED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.SUCCEEDED
                assert resource.message == "DONE"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_get_running(self, connector):
        """Test getting a running BigQuery job."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US")

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "RUNNING"

            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.connectors.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.RUNNING

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.RUNNING
                assert resource.message == "RUNNING"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_get_failed(self, connector):
        """Test getting a failed BigQuery job."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US")

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_job = MagicMock()
            mock_errors = [{"reason": "invalidQuery", "message": "Syntax error in query"}]
            mock_job.errors = mock_errors
            mock_job.state = "DONE"

            mock_client.get_job.return_value = mock_job

            resource = await connector.get(metadata)

            assert resource.phase == TaskExecution.FAILED
            assert "reason" in resource.message
            assert "invalidQuery" in resource.message
            assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_get_pending(self, connector):
        """Test getting a pending BigQuery job."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US")

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "PENDING"

            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.connectors.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.QUEUED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.QUEUED
                assert resource.message == "PENDING"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_delete(self, connector):
        """Test deleting (canceling) a BigQuery job."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US")

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            await connector.delete(metadata)

            mock_client.cancel_job.assert_called_once_with("job-123", "test-project", "US")

    @pytest.mark.asyncio
    async def test_create_with_multiple_input_types(self, connector):
        """Test creating a job with multiple input parameter types."""
        template = TaskTemplate()
        template.sql.CopyFrom(
            Sql(
                statement="SELECT * FROM table WHERE id = @user_id AND name = @name AND active = @active",
                dialect=Sql.Dialect.ANSI,
            )
        )
        template.metadata.runtime.version = "1.0.0"

        # Add multiple input variables with different types
        int_type = LiteralType()
        int_type.simple = SimpleType.INTEGER
        str_type = LiteralType()
        str_type.simple = SimpleType.STRING
        bool_type = LiteralType()
        bool_type.simple = SimpleType.BOOLEAN

        variables = VariableMap()
        variables.variables["user_id"].CopyFrom(Variable(type=int_type))
        variables.variables["name"].CopyFrom(Variable(type=str_type))
        variables.variables["active"].CopyFrom(Variable(type=bool_type))
        template.interface.inputs.CopyFrom(variables)

        custom = struct_pb2.Struct()
        custom["ProjectID"] = "test-project"
        custom["Location"] = "EU"
        template.custom.CopyFrom(custom)

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_query_job = MagicMock()
            mock_query_job.job_id = "job-multi"
            mock_client.query.return_value = mock_query_job

            inputs = {"user_id": 123, "name": "test", "active": True}
            metadata = await connector.create(template, inputs=inputs)

            assert metadata.job_id == "job-multi"
            assert metadata.location == "EU"

            # Verify query parameters
            call_args = mock_client.query.call_args
            job_config = call_args[1]["job_config"]
            assert len(job_config.query_parameters) == 3

            param_dict = {p.name: p for p in job_config.query_parameters}
            assert param_dict["user_id"].value == 123
            assert param_dict["name"].value == "test"
            assert param_dict["active"].value is True

    @pytest.mark.asyncio
    async def test_get_log_link_format(self, connector):
        """Test that the log link is properly formatted."""
        metadata = BigQueryMetadata(job_id="job-abc", project="my-project", location="europe-west1")

        with patch("flyteplugins.connectors.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "RUNNING"
            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.connectors.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.RUNNING

                resource = await connector.get(metadata)

                assert len(resource.log_links) == 1
                log_link = resource.log_links[0]
                assert log_link.name == "BigQuery Console"
                expected_uri = "https://console.cloud.google.com/bigquery?project=my-project&j=bq:europe-west1:job-abc&page=queryresults"
                assert log_link.uri == expected_uri
