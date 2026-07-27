import typing
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("lance")
pytest.importorskip("pyarrow")

import lance
import pyarrow as pa
from flyte.io._dataframe.dataframe import DataFrame, DataFrameTransformerEngine
from flyte.types import TypeEngine

# Import lance handlers to register them
import flyteplugins.lance.df_transformer  # noqa: F401
from flyteplugins.lance.df_transformer import (
    LANCE,
    ArrowToLanceEncodingHandler,
    LanceDatasetToLanceEncodingHandler,
    LanceToArrowDecodingHandler,
    LanceToLanceDatasetDecodingHandler,
    get_lance_storage_options,
)

# Sample data for testing
TEST_TABLE = pa.table(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [1.5, 2.5, 3.5],
    }
)

LANCE_DF = typing.Annotated[DataFrame, LANCE]


# ============================================================================
# Handler registration and properties
# ============================================================================


def test_handler_properties():
    """Encoder/decoder handlers advertise the right type, protocol and format."""
    enc = LanceDatasetToLanceEncodingHandler()
    assert enc.python_type is lance.LanceDataset
    assert enc.protocol is None
    assert enc.supported_format == LANCE

    dec = LanceToLanceDatasetDecodingHandler()
    assert dec.python_type is lance.LanceDataset
    assert dec.protocol is None
    assert dec.supported_format == LANCE

    arrow_enc = ArrowToLanceEncodingHandler()
    assert arrow_enc.python_type is pa.Table
    assert arrow_enc.supported_format == LANCE

    arrow_dec = LanceToArrowDecodingHandler()
    assert arrow_dec.python_type is pa.Table
    assert arrow_dec.supported_format == LANCE


def test_retrieving_encoder():
    """Encoders are retrievable for both lance.LanceDataset and pyarrow.Table."""
    assert DataFrameTransformerEngine.get_encoder(lance.LanceDataset, "file", LANCE) is not None
    assert DataFrameTransformerEngine.get_encoder(pa.Table, "file", LANCE) is not None


def test_retrieving_decoder():
    """Decoders are retrievable for both lance.LanceDataset and pyarrow.Table."""
    assert DataFrameTransformerEngine.get_decoder(lance.LanceDataset, "file", LANCE) is not None
    assert DataFrameTransformerEngine.get_decoder(pa.Table, "file", LANCE) is not None


def test_lance_dataset_default_format():
    """lance.LanceDataset defaults to the "lance" format."""
    assert DataFrameTransformerEngine.DEFAULT_FORMATS.get(lance.LanceDataset) == LANCE


def test_arrow_lance_is_opt_in():
    """pyarrow.Table must NOT default to lance (it stays parquet); lance is opt-in."""
    assert DataFrameTransformerEngine.DEFAULT_FORMATS.get(pa.Table) != LANCE


# ============================================================================
# Annotation extraction
# ============================================================================


def test_annotate_extraction():
    """The "lance" format is extracted from an Annotated[DataFrame, "lance"]."""
    from flyte.io._dataframe.dataframe import extract_cols_and_format

    base_type, _cols, fmt, _schema, _hash = extract_cols_and_format(LANCE_DF)
    assert base_type is DataFrame
    assert fmt == LANCE


def test_types_lance_dataframe_with_format():
    """to_literal_type carries the lance format through an annotation."""
    lt = TypeEngine.to_literal_type(LANCE_DF)
    assert lt.structured_dataset_type is not None
    assert lt.structured_dataset_type.format == LANCE


# ============================================================================
# Round-trip: pyarrow.Table -> lance -> {lance.LanceDataset, pyarrow.Table}
# ============================================================================


@pytest.mark.asyncio
async def test_arrow_to_lance_decode_as_dataset(ctx_with_test_raw_data_path):
    """Encode a pyarrow.Table as lance, decode it back as a streaming LanceDataset."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(LANCE_DF)

    fdf = DataFrame.from_df(val=TEST_TABLE)
    lit = await fdt.to_literal(fdf, python_type=LANCE_DF, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == LANCE
    assert lit.scalar.structured_dataset.uri

    ds = await fdt.to_python_value(lit, expected_python_type=lance.LanceDataset)
    assert isinstance(ds, lance.LanceDataset)
    assert ds.count_rows() == TEST_TABLE.num_rows
    assert ds.to_table().column("name").to_pylist() == ["Alice", "Bob", "Charlie"]


@pytest.mark.asyncio
async def test_arrow_to_lance_decode_as_arrow(ctx_with_test_raw_data_path):
    """Encode a pyarrow.Table as lance, decode it back eagerly as a pyarrow.Table."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(LANCE_DF)

    fdf = DataFrame.from_df(val=TEST_TABLE)
    lit = await fdt.to_literal(fdf, python_type=LANCE_DF, expected=lt)

    tbl = await fdt.to_python_value(lit, expected_python_type=pa.Table)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == TEST_TABLE.num_rows
    assert tbl.column("id").to_pylist() == [1, 2, 3]


@pytest.mark.asyncio
async def test_dataframe_open_lance_dataset(ctx_with_test_raw_data_path):
    """The lazy DataFrame.open(lance.LanceDataset).all() streaming handle works."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(LANCE_DF)

    fdf = DataFrame.from_df(val=TEST_TABLE)
    lit = await fdt.to_literal(fdf, python_type=LANCE_DF, expected=lt)

    restored = await fdt.to_python_value(lit, expected_python_type=DataFrame)
    assert isinstance(restored, DataFrame)
    assert restored.format == LANCE

    ds = await restored.open(lance.LanceDataset).all()
    assert isinstance(ds, lance.LanceDataset)
    assert ds.count_rows() == TEST_TABLE.num_rows
    # Streaming reads: a scanner batch and a random-access take both work.
    batches = list(ds.scanner(columns=["id"], batch_size=2).to_batches())
    assert sum(b.num_rows for b in batches) == TEST_TABLE.num_rows
    taken = ds.take([2, 0], columns=["name"])
    assert taken.column("name").to_pylist() == ["Charlie", "Alice"]


# ============================================================================
# Round-trip: lance.LanceDataset -> lance -> lance.LanceDataset (default format)
# ============================================================================


@pytest.mark.asyncio
async def test_lance_dataset_roundtrip(ctx_with_test_raw_data_path, tmp_path):
    """A raw lance.LanceDataset returned from a task round-trips through lance format."""
    src_uri = str(tmp_path / "src.lance")
    lance.write_dataset(TEST_TABLE, src_uri)
    src_ds = lance.dataset(src_uri)

    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(lance.LanceDataset)

    lit = await fdt.to_literal(src_ds, python_type=lance.LanceDataset, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == LANCE
    # Encoder writes to a Flyte-managed path, not the source path.
    assert lit.scalar.structured_dataset.uri != src_uri

    ds = await fdt.to_python_value(lit, expected_python_type=lance.LanceDataset)
    assert isinstance(ds, lance.LanceDataset)
    assert ds.count_rows() == TEST_TABLE.num_rows
    assert ds.to_table().column("name").to_pylist() == ["Alice", "Bob", "Charlie"]


# ============================================================================
# get_lance_storage_options
# ============================================================================


def test_get_lance_storage_options_none_protocol():
    assert get_lance_storage_options(None) == {}


def test_get_lance_storage_options_empty_protocol():
    assert get_lance_storage_options("") == {}


def test_get_lance_storage_options_unknown_protocol():
    assert get_lance_storage_options("unknown") == {}


def test_get_lance_storage_options_gs():
    assert get_lance_storage_options("gs") == {}
    assert get_lance_storage_options("gcs") == {}


def test_get_lance_storage_options_s3_with_mock():
    mock_s3_config = MagicMock()
    mock_s3_config.access_key_id = "test_access_key"
    mock_s3_config.secret_access_key = "test_secret_key"
    mock_s3_config.region = "us-west-2"
    mock_s3_config.endpoint = "http://localhost:9000"

    from flyte.storage import S3

    with patch("flyte._initialize.get_storage", return_value=mock_s3_config):
        with patch.object(S3, "auto", return_value=mock_s3_config):
            result = get_lance_storage_options("s3")
            assert result["aws_access_key_id"] == "test_access_key"
            assert result["aws_secret_access_key"] == "test_secret_key"
            assert result["aws_region"] == "us-west-2"
            # Lance uses aws_endpoint (not aws_endpoint_url like polars).
            assert result["aws_endpoint"] == "http://localhost:9000"
            assert result["aws_allow_http"] == "true"


def test_get_lance_storage_options_abfs_with_mock():
    mock_abfs_config = MagicMock()
    mock_abfs_config.account_name = "test_account"
    mock_abfs_config.account_key = "test_key"
    mock_abfs_config.tenant_id = "test_tenant"
    mock_abfs_config.client_id = "test_client"
    mock_abfs_config.client_secret = "test_secret"

    from flyte.storage import ABFS

    with patch("flyte._initialize.get_storage", return_value=mock_abfs_config):
        with patch.object(ABFS, "auto", return_value=mock_abfs_config):
            result = get_lance_storage_options("abfs")
            assert result["azure_storage_account_name"] == "test_account"
            assert result["azure_storage_account_key"] == "test_key"
            assert result["azure_storage_tenant_id"] == "test_tenant"
            assert result["azure_storage_client_id"] == "test_client"
            assert result["azure_storage_client_secret"] == "test_secret"
