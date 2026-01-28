from unittest.mock import MagicMock, patch

import pytest
from flyte.io import DataFrame
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.interface_pb2 import Variable, VariableMap
from flyteidl2.core.tasks_pb2 import Sql, TaskTemplate
from flyteidl2.core.types_pb2 import LiteralType, StructuredDatasetType
from google.protobuf import struct_pb2

from flyteplugins.connectors.snowflake.connector import (
    SnowflakeConnector,
    SnowflakeJobMetadata,
    _construct_query_link,
    _expand_batch_query,
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


def test_construct_query_link_org_account():
    """Test constructing query link with org-account format."""
    link = _construct_query_link("myorg-myaccount", "query-123")
    assert link == "https://app.snowflake.com/myorg/myaccount/#/compute/history/queries/query-123/detail"


def test_construct_query_link_simple():
    """Test constructing query link with simple account name (no hyphen)."""
    link = _construct_query_link("myaccount", "query-456")
    assert link == "https://app.snowflake.com/myaccount/#/compute/history/queries/query-456/detail"


def test_expand_batch_query():
    """Test expanding a parameterized query with list inputs into multi-row VALUES."""
    query = "INSERT INTO t (id, name) VALUES (%(id)s, %(name)s)"
    inputs = {"id": [1, 2], "name": ["Alice", "Bob"]}

    expanded, params = _expand_batch_query(query, inputs)

    assert expanded == ("INSERT INTO t (id, name) VALUES (%(id_0)s, %(name_0)s), (%(id_1)s, %(name_1)s)")
    assert params == {"id_0": 1, "name_0": "Alice", "id_1": 2, "name_1": "Bob"}


def test_expand_batch_query_no_values_clause():
    """Test that _expand_batch_query raises on queries without VALUES."""
    with pytest.raises(ValueError, match="VALUES"):
        _expand_batch_query("SELECT * FROM t", {"id": [1, 2]})


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
        """Create a task template with output variables."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT * FROM users", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["account"] = "test-account"
        custom["user"] = "test-user"
        custom["database"] = "test-db"
        custom["schema"] = "PUBLIC"
        custom["warehouse"] = "test-warehouse"
        template.custom.CopyFrom(custom)

        # Set output variables so has_output is True
        template.interface.outputs.CopyFrom(
            VariableMap(
                variables={"results": Variable(type=LiteralType(structured_dataset_type=StructuredDatasetType()))}
            )
        )

        return template

    def test_connector_class_attributes(self, connector):
        """Test that the connector has the correct class attributes."""
        assert connector.name == "Snowflake Connector"
        assert connector.task_type_name == "snowflake"
        assert connector.metadata_type == SnowflakeJobMetadata

    @pytest.mark.asyncio
    async def test_create_minimal(self, connector, task_template_minimal):
        """Test creating a Snowflake query without inputs."""
        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
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
            mock_cursor.execute_async.assert_called_once_with("SELECT 1", None)
            mock_cursor.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_output(self, connector, task_template_with_output):
        """Test creating a Snowflake query with output location."""
        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.sfqid = "query-456"
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            metadata = await connector.create(task_template_with_output, inputs=None)

            assert metadata.query_id == "query-456"
            assert metadata.has_output is True

    @pytest.mark.asyncio
    async def test_create_batch_inputs(self, connector):
        """Test creating a Snowflake query with batch flag expands to multi-row VALUES."""
        template = TaskTemplate()
        template.sql.CopyFrom(
            Sql(
                statement="INSERT INTO t (id, name) VALUES (%(id)s, %(name)s)",
                dialect=Sql.Dialect.ANSI,
            )
        )
        template.metadata.runtime.version = "1.0.0"
        custom = struct_pb2.Struct()
        custom["account"] = "test-account"
        custom["user"] = "test-user"
        custom["database"] = "test-db"
        custom["schema"] = "PUBLIC"
        custom["warehouse"] = "test-warehouse"
        custom["batch"] = True
        template.custom.CopyFrom(custom)

        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.sfqid = "query-batch-789"
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            batch_inputs = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
            metadata = await connector.create(template, inputs=batch_inputs)

            assert metadata.query_id == "query-batch-789"

            # Verify execute_async was called with expanded multi-row query
            mock_cursor.execute_async.assert_called_once()
            call_args = mock_cursor.execute_async.call_args[0]
            expanded_query = call_args[0]
            assert expanded_query == (
                "INSERT INTO t (id, name) VALUES (%(id_0)s, %(name_0)s), (%(id_1)s, %(name_1)s), (%(id_2)s, %(name_2)s)"
            )
            flat_params = call_args[1]
            assert flat_params == {
                "id_0": 1,
                "name_0": "Alice",
                "id_1": 2,
                "name_1": "Bob",
                "id_2": 3,
                "name_2": "Charlie",
            }

    @pytest.mark.asyncio
    async def test_create_scalar_inputs_uses_execute_async(self, connector, task_template_minimal):
        """Test that scalar (non-list) inputs still use execute_async directly."""
        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.sfqid = "query-scalar-101"
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            scalar_inputs = {"id": 1, "name": "Alice"}
            metadata = await connector.create(task_template_minimal, inputs=scalar_inputs)

            assert metadata.query_id == "query-scalar-101"
            mock_cursor.execute_async.assert_called_once_with("SELECT 1", scalar_inputs)

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

        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()

            # Mock successful query status on the connection
            mock_status = MagicMock()
            mock_status.name = "SUCCESS"
            mock_conn.get_query_status_throw_if_error.return_value = mock_status
            mock_get_conn.return_value = mock_conn

            with patch("flyteplugins.connectors.snowflake.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.SUCCEEDED

                resource = await connector.get(metadata)

                assert resource.phase == TaskExecution.SUCCEEDED
                assert resource.message == "SUCCESS"
                assert len(resource.log_links) == 1
                assert resource.log_links[0].name == "Snowflake Dashboard"
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

        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()

            mock_status = MagicMock()
            mock_status.name = "SUCCESS"
            mock_conn.get_query_status_throw_if_error.return_value = mock_status
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

        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()

            mock_status = MagicMock()
            mock_status.name = "RUNNING"
            mock_conn.get_query_status_throw_if_error.return_value = mock_status
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

        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()

            # Mock failed query on the connection
            mock_conn.get_query_status_throw_if_error.side_effect = Exception("Query execution failed")
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

        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
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
            account="myorg-myaccount",
            user="test-user",
            database="test-db",
            schema="PUBLIC",
            warehouse="test-warehouse",
            query_id="query-abc",
            has_output=False,
        )

        with patch("flyteplugins.connectors.snowflake.connector._get_snowflake_connection") as mock_get_conn:
            mock_conn = MagicMock()

            mock_status = MagicMock()
            mock_status.name = "RUNNING"
            mock_conn.get_query_status_throw_if_error.return_value = mock_status
            mock_get_conn.return_value = mock_conn

            with patch("flyteplugins.connectors.snowflake.connector.convert_to_flyte_phase") as mock_convert:
                mock_convert.return_value = TaskExecution.RUNNING

                resource = await connector.get(metadata)

                assert len(resource.log_links) == 1
                log_link = resource.log_links[0]
                assert log_link.name == "Snowflake Dashboard"
                assert log_link.uri == (
                    "https://app.snowflake.com/myorg/myaccount/#/compute/history/queries/query-abc/detail"
                )
