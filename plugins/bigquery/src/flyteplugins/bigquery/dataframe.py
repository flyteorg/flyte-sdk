import typing

from flyte.io.extend import DataFrameDecoder
from flyteidl2.core import literals_pb2
from google.cloud import bigquery_storage
from google.cloud.bigquery_storage_v1 import types

if typing.TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
else:
    from flyte._utils import lazy_module

    pd = lazy_module("pandas")
    pa = lazy_module("pyarrow")

BIGQUERY = "bq"


def _parse_bigquery_uri(uri: str) -> tuple[str, str, str]:
    """Parse bq://<project>:<dataset>.<table> into its components."""
    if not uri.startswith("bq://"):
        raise ValueError(f"Invalid BigQuery URI {uri!r}. Expected bq://<project>:<dataset>.<table>.")

    try:
        project_id, table_path = uri.removeprefix("bq://").split(":", 1)
        dataset_id, table_id = table_path.split(".", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid BigQuery URI {uri!r}. Expected bq://<project>:<dataset>.<table>.") from exc

    if not project_id or not dataset_id or not table_id:
        raise ValueError(f"Invalid BigQuery URI {uri!r}. Expected bq://<project>:<dataset>.<table>.")

    return project_id, dataset_id, table_id


def _read_from_bq(
    flyte_value: literals_pb2.StructuredDataset,
    current_task_metadata: literals_pb2.StructuredDatasetMetadata,
) -> "pd.DataFrame":
    project_id, dataset_id, table_id = _parse_bigquery_uri(flyte_value.uri)

    read_options = None
    structured_dataset_type = current_task_metadata.structured_dataset_type
    if structured_dataset_type and structured_dataset_type.columns:
        read_options = types.ReadSession.TableReadOptions(
            selected_fields=[column.name for column in structured_dataset_type.columns]
        )

    table = f"projects/{project_id}/datasets/{dataset_id}/tables/{table_id}"
    read_session = types.ReadSession(
        table=table,
        data_format=types.DataFormat.ARROW,
        read_options=read_options,
    )
    client = bigquery_storage.BigQueryReadClient()
    session = client.create_read_session(
        parent=f"projects/{project_id}",
        read_session=read_session,
    )

    frames = [page.to_dataframe() for stream in session.streams for page in client.read_rows(stream.name).rows().pages]
    if frames:
        return pd.concat(frames)

    schema = pa.ipc.read_schema(pa.py_buffer(session.arrow_schema.serialized_schema))
    return schema.empty_table().to_pandas()


class BQToPandasDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(pd.DataFrame, BIGQUERY, "")

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> "pd.DataFrame":
        return _read_from_bq(flyte_value, current_task_metadata)
