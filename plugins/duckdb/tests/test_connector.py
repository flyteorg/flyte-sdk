import pathlib
import tempfile

import pytest
from flyte.io import DataFrame
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.interface_pb2 import Variable, VariableEntry, VariableMap
from flyteidl2.core.tasks_pb2 import Sql, TaskTemplate
from flyteidl2.core.types_pb2 import LiteralType, StructuredDatasetType
from google.protobuf import struct_pb2

from flyteplugins.duckdb.connector import (
    DuckDBConnector,
    DuckDBJobMetadata,
)


def _make_output_variable_map():
    return VariableMap(
        variables=[
            VariableEntry(
                key="results",
                value=Variable(type=LiteralType(structured_dataset_type=StructuredDatasetType())),
            )
        ]
    )


def test_metadata_creation():
    """Test creating DuckDBJobMetadata instance."""
    metadata = DuckDBJobMetadata(
        query_id="test-query-123",
        result_uri="/tmp/duckdb_result_test.parquet",
        has_output=True,
    )
    assert metadata.query_id == "test-query-123"
    assert metadata.result_uri == "/tmp/duckdb_result_test.parquet"
    assert metadata.has_output is True


def test_metadata_defaults():
    """Test DuckDBJobMetadata default values."""
    metadata = DuckDBJobMetadata(query_id="test-query-456")
    assert metadata.query_id == "test-query-456"
    assert metadata.result_uri is None
    assert metadata.has_output is False


class TestDuckDBConnector:
    @pytest.fixture
    def connector(self):
        """Create a DuckDBConnector instance."""
        return DuckDBConnector()

    @pytest.fixture
    def task_template_minimal(self):
        """Create a minimal task template for testing."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT 1", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["database_path"] = ":memory:"
        template.custom.CopyFrom(custom)

        return template

    @pytest.fixture
    def task_template_with_output(self):
        """Create a task template with output variables."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT * FROM range(10)", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["database_path"] = ":memory:"
        template.custom.CopyFrom(custom)

        template.interface.outputs.CopyFrom(_make_output_variable_map())

        return template

    @pytest.fixture
    def task_template_with_extensions(self):
        """Create a task template with extensions configured."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT 1", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["database_path"] = ":memory:"

        # Extensions as a list in the Struct
        extensions_list = struct_pb2.ListValue()
        extensions_list.values.add().string_value = "json"
        custom.fields["extensions"].CopyFrom(struct_pb2.Value(list_value=extensions_list))

        template.custom.CopyFrom(custom)

        return template

    def test_connector_class_attributes(self, connector):
        """Test that the connector has the correct class attributes."""
        assert connector.name == "DuckDB Connector"
        assert connector.task_type_name == "duckdb"
        assert connector.metadata_type == DuckDBJobMetadata

    @pytest.mark.asyncio
    async def test_create_minimal(self, connector, task_template_minimal):
        """Test creating a DuckDB query without inputs and without output."""
        metadata = await connector.create(task_template_minimal, inputs=None)

        assert metadata.query_id is not None
        assert metadata.has_output is False
        assert metadata.result_uri is None

    @pytest.mark.asyncio
    async def test_create_with_output(self, connector, task_template_with_output):
        """Test creating a DuckDB query with output produces a parquet file."""
        metadata = await connector.create(task_template_with_output, inputs=None)

        assert metadata.query_id is not None
        assert metadata.has_output is True
        assert metadata.result_uri is not None
        assert metadata.result_uri.endswith(".parquet")
        assert pathlib.Path(metadata.result_uri).exists()

        # Clean up
        pathlib.Path(metadata.result_uri).unlink()

    @pytest.mark.asyncio
    async def test_create_with_inputs(self, connector):
        """Test creating a DuckDB query with parameterized inputs."""
        template = TaskTemplate()
        template.sql.CopyFrom(
            Sql(statement="SELECT * FROM range(10) WHERE range > %(min_val)s", dialect=Sql.Dialect.ANSI)
        )
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["database_path"] = ":memory:"
        template.custom.CopyFrom(custom)

        template.interface.outputs.CopyFrom(_make_output_variable_map())

        metadata = await connector.create(template, inputs={"min_val": 5})

        assert metadata.has_output is True
        assert metadata.result_uri is not None
        assert pathlib.Path(metadata.result_uri).exists()

        # Verify the result has correct data
        import pandas as pd_lib

        df = pd_lib.read_parquet(metadata.result_uri)
        assert len(df) == 4  # values 6, 7, 8, 9
        assert all(df["range"] > 5)

        # Clean up
        pathlib.Path(metadata.result_uri).unlink()

    @pytest.mark.asyncio
    async def test_create_without_output(self, connector):
        """Test creating a DuckDB query that produces no output."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="CREATE TABLE test (id INTEGER)", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["database_path"] = ":memory:"
        template.custom.CopyFrom(custom)

        metadata = await connector.create(template, inputs=None)

        assert metadata.has_output is False
        assert metadata.result_uri is None

    @pytest.mark.asyncio
    async def test_get_succeeded_with_output(self, connector):
        """Test getting a completed DuckDB query with output."""
        metadata = DuckDBJobMetadata(
            query_id="test-123",
            result_uri="/tmp/duckdb_result_test.parquet",
            has_output=True,
        )

        resource = await connector.get(metadata)

        assert resource.phase == TaskExecution.SUCCEEDED
        assert resource.message == "Query completed"
        assert resource.outputs is not None
        assert "results" in resource.outputs
        assert isinstance(resource.outputs["results"], DataFrame)
        assert "duckdb://" in resource.outputs["results"].uri
        assert "/tmp/duckdb_result_test.parquet" in resource.outputs["results"].uri

    @pytest.mark.asyncio
    async def test_get_succeeded_without_output(self, connector):
        """Test getting a completed DuckDB query without output."""
        metadata = DuckDBJobMetadata(
            query_id="test-456",
            has_output=False,
        )

        resource = await connector.get(metadata)

        assert resource.phase == TaskExecution.SUCCEEDED
        assert resource.message == "Query completed"
        assert resource.outputs is None

    @pytest.mark.asyncio
    async def test_delete_cleans_up_file(self, connector):
        """Test that delete removes the temporary result file."""
        # Create a temporary file to simulate a result
        tmp_path = pathlib.Path(tempfile.gettempdir()) / "duckdb_result_delete_test.parquet"
        tmp_path.write_text("test")

        assert tmp_path.exists()

        metadata = DuckDBJobMetadata(
            query_id="test-delete",
            result_uri=str(tmp_path),
            has_output=True,
        )

        await connector.delete(metadata)

        assert not tmp_path.exists()

    @pytest.mark.asyncio
    async def test_delete_no_file(self, connector):
        """Test that delete handles missing files gracefully."""
        metadata = DuckDBJobMetadata(
            query_id="test-no-file",
            result_uri="/tmp/nonexistent_file.parquet",
            has_output=True,
        )

        # Should not raise
        await connector.delete(metadata)

    @pytest.mark.asyncio
    async def test_delete_no_result_uri(self, connector):
        """Test that delete handles None result_uri gracefully."""
        metadata = DuckDBJobMetadata(
            query_id="test-no-uri",
            has_output=False,
        )

        # Should not raise
        await connector.delete(metadata)

    @pytest.mark.asyncio
    async def test_create_with_extensions(self, connector, task_template_with_extensions):
        """Test that extensions are installed and loaded."""
        # json extension is bundled with DuckDB, so this should work without network
        metadata = await connector.create(task_template_with_extensions, inputs=None)
        assert metadata.query_id is not None

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, connector):
        """Test the complete create -> get -> delete flow."""
        template = TaskTemplate()
        template.sql.CopyFrom(Sql(statement="SELECT 42 AS answer", dialect=Sql.Dialect.ANSI))
        template.metadata.runtime.version = "1.0.0"

        custom = struct_pb2.Struct()
        custom["database_path"] = ":memory:"
        template.custom.CopyFrom(custom)

        template.interface.outputs.CopyFrom(_make_output_variable_map())

        # Create
        metadata = await connector.create(template, inputs=None)
        assert metadata.has_output is True
        assert pathlib.Path(metadata.result_uri).exists()

        # Get
        resource = await connector.get(metadata)
        assert resource.phase == TaskExecution.SUCCEEDED
        assert resource.outputs is not None

        # Verify result content
        import pandas as pd_lib

        df = pd_lib.read_parquet(metadata.result_uri)
        assert df["answer"].iloc[0] == 42

        # Delete
        await connector.delete(metadata)
        assert not pathlib.Path(metadata.result_uri).exists()
