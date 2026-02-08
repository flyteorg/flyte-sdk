import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flyte import logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyte.io import DataFrame
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf import json_format

import duckdb

TASK_TYPE = "duckdb"


@dataclass
class DuckDBJobMetadata(ResourceMeta):
    """
    Metadata for a DuckDB query job.

    Attributes:
        query_id: Unique identifier for tracking the query.
        result_uri: Path to the temporary parquet file containing query results.
        has_output: Indicates if the query produces output.
    """

    query_id: str
    result_uri: Optional[str] = None
    has_output: bool = False


class DuckDBConnector(AsyncConnector):
    name = "DuckDB Connector"
    task_type_name = TASK_TYPE
    metadata_type = DuckDBJobMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DuckDBJobMetadata:
        """
        Execute a DuckDB query.

        DuckDB queries run locally and synchronously. The query executes entirely within
        this method, and the result (if any) is written to a temporary parquet file.

        Args:
            task_template: The Flyte task template containing the SQL query and configuration.
            inputs: Optional dictionary of input parameters for parameterized queries.

        Returns:
            A DuckDBJobMetadata object containing the query ID and result file path.
        """
        custom = json_format.MessageToDict(task_template.custom) if task_template.custom else {}

        database_path = custom.get("database_path", ":memory:")
        extensions = custom.get("extensions", [])

        query = task_template.sql.statement
        has_output = task_template.interface.outputs is not None and len(task_template.interface.outputs.variables) > 0
        query_id = str(uuid.uuid4())

        def _execute_query():
            conn = duckdb.connect(database=database_path)
            try:
                for ext in extensions:
                    conn.install_extension(ext)
                    conn.load_extension(ext)

                if inputs:
                    params = list(inputs.values())
                    param_names = list(inputs.keys())
                    positional_query = query
                    for name in param_names:
                        positional_query = positional_query.replace(f"%({name})s", "?")
                    result = conn.execute(positional_query, params)
                else:
                    result = conn.execute(query)

                result_uri = None
                if has_output:
                    df = result.fetchdf()
                    result_uri = os.path.join(tempfile.gettempdir(), f"duckdb_result_{query_id}.parquet")
                    df.to_parquet(result_uri)

                return result_uri
            finally:
                conn.close()

        loop = asyncio.get_running_loop()
        result_uri = await loop.run_in_executor(None, _execute_query)

        logger.info(f"DuckDB query executed with ID: {query_id}")

        return DuckDBJobMetadata(
            query_id=query_id,
            result_uri=result_uri,
            has_output=has_output,
        )

    async def get(
        self,
        resource_meta: DuckDBJobMetadata,
        **kwargs,
    ) -> Resource:
        """
        Get the status of a DuckDB query.

        DuckDB queries complete synchronously in create(), so this always returns SUCCEEDED.

        Args:
            resource_meta: The DuckDBJobMetadata from the create() call.

        Returns:
            A Resource object with SUCCEEDED status and optional outputs.
        """
        outputs = None
        if resource_meta.has_output and resource_meta.result_uri:
            outputs = {"results": DataFrame(uri=f"duckdb://{resource_meta.result_uri}")}

        return Resource(phase=TaskExecution.SUCCEEDED, message="Query completed", outputs=outputs)

    async def delete(
        self,
        resource_meta: DuckDBJobMetadata,
        **kwargs,
    ):
        """
        Clean up temporary result files.

        Args:
            resource_meta: The DuckDBJobMetadata containing the result file path.
        """

        def _cleanup():
            if resource_meta.result_uri and os.path.exists(resource_meta.result_uri):
                os.remove(resource_meta.result_uri)
                logger.info(f"Cleaned up temporary result file: {resource_meta.result_uri}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _cleanup)


ConnectorRegistry.register(DuckDBConnector())
