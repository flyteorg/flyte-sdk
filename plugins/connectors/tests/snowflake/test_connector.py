from unittest.mock import MagicMock, patch

import pytest
from flyte.io import DataFrame
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.tasks_pb2 import Sql, TaskTemplate
from google.protobuf import struct_pb2

from flyteplugins.connectors.snowflake.connector import (
    SnowflakeConnector,
    SnowflakeJobMetadata,
    _construct_query_link,
)


def test_metadata_creation():
    """Test creating SnowflakeJobMetadata instance."""
    metadata = SnowflakeJobMetadata(
        account="test-account",
        user="test-user",
        database="test-db",
        schema="PUBLIC",
        warehouse="test-warehouse",
        query_id="query-123",
        has_output=True,
    )
    assert metadata.account == "test-account"
    assert metadata.user == "test-user"
    assert metadata.database == "test-db"
    assert metadata.schema == "PUBLIC"
    assert metadata.warehouse == "test-warehouse"
    assert metadata.query_id == "query-123"
    assert metadata.has_output is True


def test_construct_query_link_with_dots():
    """Test constructing query link with account containing dots."""
    link = _construct_query_link("account.us-east-1.aws", "query-123")
    assert "app.snowflake.com/us-east-1/aws/account" in link
    assert "query-123" in link


def test_construct_query_link_simple():
    """Test constructing query link with simple account name."""
    link = _construct_query_link("simple-account", "query-456")
    assert "app.snowflake.com/simple-account" in link
    assert "query-456" in link


class TestSnowflakeConnector:
    @pytest.fixture
    def connector(self):
        """Create a SnowflakeConnector instance."""
        return SnowflakeConnector()

    @pytest.fixture
    def task_template_minimal(self):
        """Create a minimal task template for testing."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT 1", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["account"] = "test-account"
        custom["user"] = "test-user"
        custom["database"] = "test-db"
        custom["schema"] = "PUBLIC"
        custom["warehouse"] = "test-warehouse"
        template.custom.CopyFrom(custom)

        return template

    @pytest.fixture
    def task_template_with_output(self):
        """Create a task template with output location."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT * FROM users", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["account"] = "test-account"
        custom["user"] = "test-user"
        custom["database"] = "test-db"
        custom["schema"] = "PUBLIC"
        custom["warehouse"] = "test-warehouse"
        custom["output_location_prefix"] = "s3://bucket/path/"
        template.custom.CopyFrom(custom)

        return template

    def test_connector_class_attributes(self, connector):
        """Test that the connector has the correct class attributes."""
        assert connector.name == "Snowflake Connector"
        assert connector.task_type_name == "snowflake"
        assert connector.metadata_type == SnowflakeJobMetadata

    @pytest.mark.asyncio
    async def test_create_minimal(self, connector, task_template_minimal):
        """Test creating a Snowflake query without inputs."""
        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.sfqid = "query-123"
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            metadata = await connector.create(task_template_minimal, inputs=None)

            assert metadata.query_id == "query-123"
            assert metadata.account == "test-account"
            assert metadata.user == "test-user"
            assert metadata.database == "test-db"
            assert metadata.schema == "PUBLIC"
            assert metadata.warehouse == "test-warehouse"
            assert metadata.has_output is False

            # Verify connection was created with correct parameters
            mock_get_conn.assert_called_once()
            call_kwargs = mock_get_conn.call_args[1]
            assert call_kwargs["account"] == "test-account"
            assert call_kwargs["user"] == "test-user"
            assert call_kwargs["database"] == "test-db"
            assert call_kwargs["schema"] == "PUBLIC"
            assert call_kwargs["warehouse"] == "test-warehouse"

            # Verify query was executed
            mock_cursor.execute_async.assert_called_once_with("SELECT 1")
            mock_cursor.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_output(self, connector, task_template_with_output):
        """Test creating a Snowflake query with output location."""
        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.sfqid = "query-456"
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            metadata = await connector.create(task_template_with_output, inputs=None)

            assert metadata.query_id == "query-456"
            assert metadata.has_output is True

    @pytest.mark.asyncio
    async def test_create_missing_account(self, connector, task_template_minimal):
        """Test creating a query without account raises error."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT 1", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["user"] = "test-user"
        custom["database"] = "test-db"
        custom["warehouse"] = "test-warehouse"
        template.custom.CopyFrom(custom)

        with pytest.raises(ValueError, match="Missing Snowflake account"):
            await connector.create(template, inputs=None)

    @pytest.mark.asyncio
    async def test_create_missing_required_fields(self, connector):
        """Test creating a query without required fields raises error."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT 1", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["account"] = "test-account"
        custom["user"] = "test-user"
        # Missing database and warehouse
        template.custom.CopyFrom(custom)

        with pytest.raises(ValueError, match="User, database and warehouse must be specified"):
            await connector.create(template, inputs=None)

    @pytest.mark.asyncio
    async def test_get_succeeded_with_output(self, connector):
        """Test getting a successful Snowflake query with output."""
        metadata = SnowflakeJobMetadata(
            account="test-account",
            user="test-user",
            database="test-db",
            schema="PUBLIC",
            warehouse="test-warehouse",
            query_id="query-123",
            has_output=True,
        )

        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            # Mock successful query
            mock_status = MagicMock()
            mock_status.name = "SUCCESS"
            mock_cursor.get_query_status_throw_if_error.return_value = mock_status
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            with patch("flyteplugins.connectors.snowflake.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.SUCCEEDED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.SUCCEEDED
                assert resource.message == "SUCCESS"
                assert len(resource.log_links) == 1
                assert resource.log_links[0].name == "Snowflake Console"
                assert "query-123" in resource.log_links[0].uri

                # Verify outputs contain DataFrame with correct URI
                assert resource.outputs is not None
                assert "results" in resource.outputs
                assert isinstance(resource.outputs["results"], DataFrame)
                assert "snowflake://" in resource.outputs["results"].uri
                assert "test-account" in resource.outputs["results"].uri
                assert "query-123" in resource.outputs["results"].uri

    @pytest.mark.asyncio
    async def test_get_succeeded_without_output(self, connector):
        """Test getting a successful Snowflake query without output."""
        metadata = SnowflakeJobMetadata(
            account="test-account",
            user="test-user",
            database="test-db",
            schema="PUBLIC",
            warehouse="test-warehouse",
            query_id="query-123",
            has_output=False,
        )

        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            mock_status = MagicMock()
            mock_status.name = "SUCCESS"
            mock_cursor.get_query_status_throw_if_error.return_value = mock_status
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            with patch("flyteplugins.connectors.snowflake.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.SUCCEEDED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.SUCCEEDED
                assert resource.message == "SUCCESS"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_get_running(self, connector):
        """Test getting a running Snowflake query."""
        metadata = SnowflakeJobMetadata(
            account="test-account",
            user="test-user",
            database="test-db",
            schema="PUBLIC",
            warehouse="test-warehouse",
            query_id="query-123",
            has_output=False,
        )

        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            mock_status = MagicMock()
            mock_status.name = "RUNNING"
            mock_cursor.get_query_status_throw_if_error.return_value = mock_status
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            with patch("flyteplugins.connectors.snowflake.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.RUNNING

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.RUNNING
                assert resource.message == "RUNNING"
                assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_get_failed(self, connector):
        """Test getting a failed Snowflake query."""
        metadata = SnowflakeJobMetadata(
            account="test-account",
            user="test-user",
            database="test-db",
            schema="PUBLIC",
            warehouse="test-warehouse",
            query_id="query-123",
            has_output=False,
        )

        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            # Mock failed query
            mock_cursor.get_query_status_throw_if_error.side_effect = Exception("Query execution failed")
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            resource = await connector.get(metadata)

            assert resource.phase == TaskExecution.FAILED
            assert "Query execution failed" in resource.message
            assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_delete(self, connector):
        """Test deleting (canceling) a Snowflake query."""
        metadata = SnowflakeJobMetadata(
            account="test-account",
            user="test-user",
            database="test-db",
            schema="PUBLIC",
            warehouse="test-warehouse",
            query_id="query-123",
            has_output=False,
        )

        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            await connector.delete(metadata)

            # Verify cancel query was executed
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0][0]
            assert "SYSTEM$CANCEL_QUERY" in call_args
            assert "query-123" in call_args
            mock_cursor.close.assert_called_once()
            mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_log_link_format(self, connector):
        """Test that the log link is properly formatted."""
        metadata = SnowflakeJobMetadata(
            account="my-account.us-west-2.aws",
            user="test-user",
            database="test-db",
            schema="PUBLIC",
            warehouse="test-warehouse",
            query_id="query-abc",
            has_output=False,
        )

        with patch(
            "flyteplugins.connectors.snowflake.connector._get_snowflake_connection"
        ) as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            mock_status = MagicMock()
            mock_status.name = "RUNNING"
            mock_cursor.get_query_status_throw_if_error.return_value = mock_status
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            with patch("flyteplugins.connectors.snowflake.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.RUNNING

                resource = await connector.get(metadata)

                assert len(resource.log_links) == 1
                log_link = resource.log_links[0]
                assert log_link.name == "Snowflake Console"
                assert "app.snowflake.com" in log_link.uri
                assert "query-abc" in log_link.uri
