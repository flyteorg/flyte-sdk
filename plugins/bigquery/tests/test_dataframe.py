from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest
from flyte.io.extend import DataFrameTransformerEngine
from flyteidl2.core import literals_pb2, types_pb2

from flyteplugins.bigquery.dataframe import (
    BQToPandasDecodingHandler,
    _parse_bigquery_uri,
    _read_from_bq,
)


def _structured_dataset(uri: str = "bq://test-project:test_dataset.test_table"):
    return literals_pb2.StructuredDataset(uri=uri)


def _metadata(*column_names: str):
    return literals_pb2.StructuredDatasetMetadata(
        structured_dataset_type=types_pb2.StructuredDatasetType(
            columns=[types_pb2.StructuredDatasetType.DatasetColumn(name=name) for name in column_names]
        )
    )


def test_bigquery_decoder_is_registered():
    decoder = DataFrameTransformerEngine.get_decoder(pd.DataFrame, "bq", "")
    assert isinstance(decoder, BQToPandasDecodingHandler)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("bq://project:dataset.table", ("project", "dataset", "table")),
        ("bq://my-project:my_dataset.table$20260101", ("my-project", "my_dataset", "table$20260101")),
    ],
)
def test_parse_bigquery_uri(uri, expected):
    assert _parse_bigquery_uri(uri) == expected


@pytest.mark.parametrize("uri", ["gs://bucket/file", "bq://project", "bq://:dataset.table", "bq://project:.table"])
def test_parse_bigquery_uri_rejects_invalid_uri(uri):
    with pytest.raises(ValueError, match="Expected bq://"):
        _parse_bigquery_uri(uri)


def test_read_from_bq():
    first = pd.DataFrame({"name": ["Alice"], "age": [25]})
    second = pd.DataFrame({"name": ["Bob"], "age": [30]})
    pages = [MagicMock(), MagicMock()]
    pages[0].to_dataframe.return_value = first
    pages[1].to_dataframe.return_value = second

    session = MagicMock()
    session.streams = [MagicMock(name="stream-one")]
    reader = MagicMock()
    reader.rows.return_value.pages = pages

    with patch("flyteplugins.bigquery.dataframe.bigquery_storage.BigQueryReadClient") as client_cls:
        client = client_cls.return_value
        client.create_read_session.return_value = session
        client.read_rows.return_value = reader

        result = _read_from_bq(_structured_dataset(), _metadata("name"))

    pd.testing.assert_frame_equal(result.reset_index(drop=True), pd.concat([first, second], ignore_index=True))
    request = client.create_read_session.call_args.kwargs
    assert request["parent"] == "projects/test-project"
    assert request["read_session"].table == "projects/test-project/datasets/test_dataset/tables/test_table"
    assert list(request["read_session"].read_options.selected_fields) == ["name"]


def test_read_from_empty_bq_table():
    schema = pa.schema([("name", pa.string()), ("age", pa.int64())])
    session = MagicMock()
    session.streams = []
    session.arrow_schema.serialized_schema = schema.serialize().to_pybytes()

    with patch("flyteplugins.bigquery.dataframe.bigquery_storage.BigQueryReadClient") as client_cls:
        client_cls.return_value.create_read_session.return_value = session
        result = _read_from_bq(_structured_dataset(), _metadata())

    assert result.empty
    assert list(result.columns) == ["name", "age"]
