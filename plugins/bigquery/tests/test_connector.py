import datetime
import json
from unittest.mock import MagicMock, patch

import pytest
from flyte.io import DataFrame
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.interface_pb2 import Variable, VariableEntry, VariableMap
from flyteidl2.core.tasks_pb2 import Sql, TaskTemplate
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from google.protobuf import struct_pb2

from flyteplugins.bigquery.connector import (
    BigQueryConnector,
    BigQueryMetadata,
    _get_bigquery_client,
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
    metadata = BigQueryMetadata(job_id="test-job-123", project="test-project", location="US", user_agent="Flyte/1.0.0")
    assert metadata.job_id == "test-job-123"
    assert metadata.project == "test-project"
    assert metadata.location == "US"
    assert metadata.user_agent == "Flyte/1.0.0"


class TestBigQueryConnector:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the bigquery client cache before each test."""
        _get_bigquery_client.cache_clear()
        yield
        _get_bigquery_client.cache_clear()

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

        # Add input variables using the new list-based structure
        int_type = LiteralType()
        int_type.simple = SimpleType.INTEGER
        user_id_var = Variable(type=int_type)

        variables = VariableMap()
        var_entry = VariableEntry(key="user_id", value=user_id_var)
        variables.variables.append(var_entry)
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
        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
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
        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
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
        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
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
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US", user_agent="Flyte/1.0.0")

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
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

            with patch("flyteplugins.bigquery.connector.convert_to_flyte_phase") as mock_convert:
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
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US", user_agent="Flyte/1.0.0")

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock successful job without destination
            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "DONE"
            mock_job.destination = None

            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.SUCCEEDED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.SUCCEEDED
                assert resource.message == "DONE"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_get_running(self, connector):
        """Test getting a running BigQuery job."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US", user_agent="Flyte/1.0.0")

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "RUNNING"

            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.RUNNING

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.RUNNING
                assert resource.message == "RUNNING"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_get_failed(self, connector):
        """Test getting a failed BigQuery job."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US", user_agent="Flyte/1.0.0")

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
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
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US", user_agent="Flyte/1.0.0")

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "PENDING"

            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.QUEUED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.QUEUED
                assert resource.message == "PENDING"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_delete(self, connector):
        """Test deleting (canceling) a BigQuery job."""
        metadata = BigQueryMetadata(job_id="job-123", project="test-project", location="US", user_agent="Flyte/1.0.0")

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
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

        # Add multiple input variables with different types using the new list-based structure
        int_type = LiteralType()
        int_type.simple = SimpleType.INTEGER
        str_type = LiteralType()
        str_type.simple = SimpleType.STRING
        bool_type = LiteralType()
        bool_type.simple = SimpleType.BOOLEAN

        variables = VariableMap()
        variables.variables.append(VariableEntry(key="user_id", value=Variable(type=int_type)))
        variables.variables.append(VariableEntry(key="name", value=Variable(type=str_type)))
        variables.variables.append(VariableEntry(key="active", value=Variable(type=bool_type)))
        template.interface.inputs.CopyFrom(variables)

        custom = struct_pb2.Struct()
        custom["ProjectID"] = "test-project"
        custom["Location"] = "EU"
        template.custom.CopyFrom(custom)

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
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
        metadata = BigQueryMetadata(
            job_id="job-abc", project="my-project", location="europe-west1", user_agent="Flyte/1.0.0"
        )

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_job = MagicMock()
            mock_job.errors = None
            mock_job.state = "RUNNING"
            mock_client.get_job.return_value = mock_job

            with patch("flyteplugins.bigquery.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.RUNNING

                resource = await connector.get(metadata)

                assert len(resource.log_links) == 1
                log_link = resource.log_links[0]
                assert log_link.name == "BigQuery Console"
                expected_uri = "https://console.cloud.google.com/bigquery?project=my-project&j=bq:europe-west1:job-abc&page=queryresults"
                assert log_link.uri == expected_uri

    @pytest.mark.asyncio
    async def test_create_with_google_application_credentials(self, connector, task_template_minimal):
        """Test that google_application_credentials JSON string is properly parsed."""
        credentials_dict = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "key-id-123",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIItest\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        credentials_json = json.dumps(credentials_dict)

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
            with patch(
                "flyteplugins.bigquery.connector.service_account.Credentials.from_service_account_info"
            ) as mock_creds:
                mock_credentials = MagicMock()
                mock_creds.return_value = mock_credentials

                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                mock_query_job = MagicMock()
                mock_query_job.job_id = "job-with-creds"
                mock_client.query.return_value = mock_query_job

                metadata = await connector.create(
                    task_template_minimal,
                    inputs=None,
                    google_application_credentials=credentials_json,
                )

                assert metadata.job_id == "job-with-creds"

                # Verify that from_service_account_info was called with a parsed dict, not a string
                mock_creds.assert_called_once()
                call_args = mock_creds.call_args[0][0]
                assert isinstance(call_args, dict)
                assert call_args["type"] == "service_account"
                assert call_args["project_id"] == "test-project"
                assert call_args["client_email"] == "test@test-project.iam.gserviceaccount.com"

                # Verify the credentials were passed to the client
                mock_client_class.assert_called_once()
                assert mock_client_class.call_args[1]["credentials"] == mock_credentials

    @pytest.mark.asyncio
    async def test_create_iterates_variables_with_new_structure(self, connector):
        """Test that the connector correctly iterates over variables using the new iteration pattern.

        This test verifies the change from:
            for name, lt in task_template.interface.inputs.variables.items()
        To:
            for variable in task_template.interface.inputs.variables

        The variables field changed from a map to a repeated field (list), so we now
        iterate directly over the list of Variable objects which have key and value attributes.
        """
        template = TaskTemplate()
        template.sql.CopyFrom(
            Sql(
                statement="SELECT * FROM table WHERE user_id = @user_id AND email = @email",
                dialect=Sql.Dialect.ANSI,
            )
        )
        template.metadata.runtime.version = "2.0.0"

        # Create variables using the new list-based VariableMap structure
        int_type = LiteralType()
        int_type.simple = SimpleType.INTEGER
        str_type = LiteralType()
        str_type.simple = SimpleType.STRING

        variables = VariableMap()
        variables.variables.append(VariableEntry(key="user_id", value=Variable(type=int_type)))
        variables.variables.append(VariableEntry(key="email", value=Variable(type=str_type)))
        template.interface.inputs.CopyFrom(variables)

        custom = struct_pb2.Struct()
        custom["ProjectID"] = "test-project"
        custom["Location"] = "US"
        custom["Domain"] = "test-domain"
        template.custom.CopyFrom(custom)

        with patch("flyteplugins.bigquery.connector.bigquery.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_query_job = MagicMock()
            mock_query_job.job_id = "job-iteration-test"
            mock_client.query.return_value = mock_query_job

            inputs = {"user_id": 42, "email": "test@example.com"}
            metadata = await connector.create(template, inputs=inputs)

            assert metadata.job_id == "job-iteration-test"

            # Verify that the query was called with proper parameters
            call_args = mock_client.query.call_args
            job_config = call_args[1]["job_config"]

            # The new iteration pattern should successfully create query parameters
            assert len(job_config.query_parameters) == 2

            param_dict = {p.name: p for p in job_config.query_parameters}
            assert "user_id" in param_dict
            assert "email" in param_dict
            assert param_dict["user_id"].value == 42
            assert param_dict["email"].value == "test@example.com"

            # Verify parameter types are correctly mapped
            assert param_dict["user_id"].type_ == "INT64"
            assert param_dict["email"].type_ == "STRING"
