import typing
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import flyte
import pytest
from flyte.io._dataframe import DataFrame
from flyte.io._dataframe.dataframe import PARQUET, DataFrameTransformerEngine
from flyte.types import TypeEngine

# Import polars handlers to register them
import flyteplugins.polars.df_transformer  # noqa: F401
from flyteplugins.polars.df_transformer import (
    ParquetToPolarsDecodingHandler,
    ParquetToPolarsLazyFrameDecodingHandler,
    PolarsLazyFrameToParquetEncodingHandler,
    PolarsToParquetEncodingHandler,
    get_polars_storage_options,
)

pl = pytest.importorskip("polars")

# Sample data for testing
TEST_DATA = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["NYC", "SF", "LA"]}


@pytest.fixture
def sample_dataframe():
    """Create a sample polars DataFrame for testing."""
    return pl.DataFrame(TEST_DATA)


@pytest.fixture
def sample_lazyframe():
    """Create a sample polars LazyFrame for testing."""
    return pl.LazyFrame(TEST_DATA)


def test_types_polars_dataframe():
    """Test that polars DataFrame types are recognized."""
    pt = pl.DataFrame
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    # Note: format is empty string when no annotation is specified
    # The PARQUET format is set as the default during encoding
    assert lt.structured_dataset_type.format == ""
    assert lt.structured_dataset_type.columns == []


def test_types_polars_lazyframe():
    """Test that polars LazyFrame types are recognized."""
    pt = pl.LazyFrame
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    # Note: format is empty string when no annotation is specified
    # The PARQUET format is set as the default during encoding
    assert lt.structured_dataset_type.format == ""
    assert lt.structured_dataset_type.columns == []


def test_annotate_extraction_dataframe():
    """Test annotation extraction for polars DataFrame."""
    from flyte.io._dataframe.dataframe import extract_cols_and_format

    xyz = typing.Annotated[pl.DataFrame, "myformat"]
    a, b, c, d = extract_cols_and_format(xyz)
    assert a is pl.DataFrame
    assert b is None
    assert c == "myformat"
    assert d is None

    # Without annotation, extract_cols_and_format returns empty string
    a, b, c, d = extract_cols_and_format(pl.DataFrame)
    assert a is pl.DataFrame
    assert b is None
    assert c == ""  # No format annotation
    assert d is None


def test_annotate_extraction_lazyframe():
    """Test annotation extraction for polars LazyFrame."""
    from flyte.io._dataframe.dataframe import extract_cols_and_format

    xyz = typing.Annotated[pl.LazyFrame, "myformat"]
    a, b, c, d = extract_cols_and_format(xyz)
    assert a is pl.LazyFrame
    assert b is None
    assert c == "myformat"
    assert d is None

    # Without annotation, extract_cols_and_format returns empty string
    a, b, c, d = extract_cols_and_format(pl.LazyFrame)
    assert a is pl.LazyFrame
    assert b is None
    assert c == ""  # No format annotation
    assert d is None


def test_retrieving_encoder():
    """Test that encoders can be retrieved for polars types."""
    assert DataFrameTransformerEngine.get_encoder(pl.DataFrame, "file", PARQUET) is not None
    assert DataFrameTransformerEngine.get_encoder(pl.LazyFrame, "file", PARQUET) is not None

    # Asking for a generic means you're okay with any one registered for that
    # type assuming there's just one.
    assert DataFrameTransformerEngine.get_encoder(pl.DataFrame, "file", "") is DataFrameTransformerEngine.get_encoder(
        pl.DataFrame, "file", PARQUET
    )
    assert DataFrameTransformerEngine.get_encoder(pl.LazyFrame, "file", "") is DataFrameTransformerEngine.get_encoder(
        pl.LazyFrame, "file", PARQUET
    )


@pytest.mark.asyncio
async def test_to_literal_dataframe(ctx_with_test_raw_data_path, sample_dataframe):
    """Test encoding polars DataFrame to literal."""
    lt = TypeEngine.to_literal_type(pl.DataFrame)

    fdt = DataFrameTransformerEngine()

    lit = await fdt.to_literal(sample_dataframe, python_type=pl.DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET
    assert lit.scalar.structured_dataset.uri is not None

    # Test that we can decode it back
    restored_df = await fdt.to_python_value(lit, expected_python_type=pl.DataFrame)
    assert isinstance(restored_df, pl.DataFrame)
    assert restored_df.shape == sample_dataframe.shape
    assert restored_df.columns == sample_dataframe.columns


@pytest.mark.asyncio
async def test_to_literal_lazyframe(ctx_with_test_raw_data_path, sample_lazyframe):
    """Test encoding polars LazyFrame to literal."""
    lt = TypeEngine.to_literal_type(pl.LazyFrame)

    fdt = DataFrameTransformerEngine()

    lit = await fdt.to_literal(sample_lazyframe, python_type=pl.LazyFrame, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET
    assert lit.scalar.structured_dataset.uri is not None

    # Test that we can decode it back
    restored_lazy = await fdt.to_python_value(lit, expected_python_type=pl.LazyFrame)
    assert isinstance(restored_lazy, pl.LazyFrame)
    # Collect to compare
    restored_df = restored_lazy.collect()
    original_df = sample_lazyframe.collect()
    assert restored_df.shape == original_df.shape
    assert restored_df.columns == original_df.columns


@pytest.mark.asyncio
async def test_dataframe_roundtrip(ctx_with_test_raw_data_path, sample_dataframe):
    """Test roundtrip encoding/decoding of polars DataFrame."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(pl.DataFrame)

    # Encode
    lit = await fdt.to_literal(sample_dataframe, python_type=pl.DataFrame, expected=lt)

    # Decode
    restored_df = await fdt.to_python_value(lit, expected_python_type=pl.DataFrame)

    # Compare
    assert restored_df.shape == sample_dataframe.shape
    assert restored_df.columns == sample_dataframe.columns
    # Compare data
    for col in sample_dataframe.columns:
        assert restored_df[col].to_list() == sample_dataframe[col].to_list()


@pytest.mark.asyncio
async def test_lazyframe_roundtrip(ctx_with_test_raw_data_path, sample_lazyframe):
    """Test roundtrip encoding/decoding of polars LazyFrame."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(pl.LazyFrame)

    # Encode (lazy frame will be collected)
    lit = await fdt.to_literal(sample_lazyframe, python_type=pl.LazyFrame, expected=lt)

    # Decode back as lazy frame
    restored_lazy = await fdt.to_python_value(lit, expected_python_type=pl.LazyFrame)

    # Collect and compare
    restored_df = restored_lazy.collect()
    original_df = sample_lazyframe.collect()

    assert restored_df.shape == original_df.shape
    assert restored_df.columns == original_df.columns
    # Compare data
    for col in original_df.columns:
        assert restored_df[col].to_list() == original_df[col].to_list()


@pytest.mark.asyncio
async def test_dataframe_through_flyte_dataframe(ctx_with_test_raw_data_path, sample_dataframe):
    """Test using polars DataFrame through Flyte DataFrame wrapper."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(pl.DataFrame)

    # Create Flyte DataFrame from polars DataFrame
    fdf = DataFrame.from_df(val=sample_dataframe)

    # Encode
    lit = await fdt.to_literal(fdf, python_type=pl.DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET

    # Decode back
    restored_df = await fdt.to_python_value(lit, expected_python_type=pl.DataFrame)
    assert isinstance(restored_df, pl.DataFrame)
    assert restored_df.shape == sample_dataframe.shape


@pytest.mark.asyncio
async def test_raw_dataframe_io(ctx_with_test_raw_data_path, sample_dataframe):
    """Test using raw polars DataFrame as task input/output."""
    flyte.init()
    env = flyte.TaskEnvironment(name="test-polars-df")

    @env.task
    async def process_dataframe(df: pl.DataFrame) -> pl.DataFrame:
        # Simple transformation
        return df.with_columns((pl.col("age") + 1).alias("age_plus_one"))

    run = flyte.with_runcontext("local").run(process_dataframe, sample_dataframe)
    result = run.outputs()
    assert isinstance(result, pl.DataFrame)
    assert "age_plus_one" in result.columns
    assert result.shape[0] == sample_dataframe.shape[0]


@pytest.mark.asyncio
async def test_raw_lazyframe_io(ctx_with_test_raw_data_path, sample_lazyframe):
    """Test using raw polars LazyFrame as task input/output."""
    flyte.init()
    env = flyte.TaskEnvironment(name="test-polars-lazy")

    @env.task
    async def process_lazyframe(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Simple transformation - lazy evaluation
        return lf.with_columns((pl.col("age") + 1).alias("age_plus_one"))

    run = flyte.with_runcontext("local").run(process_lazyframe, sample_lazyframe)
    result = run.outputs()
    assert isinstance(result, pl.LazyFrame)
    # Collect to check
    collected = result.collect()
    assert "age_plus_one" in collected.columns
    assert collected.shape[0] == sample_lazyframe.collect().shape[0]


# ============================================================================
# Tests for get_polars_storage_options function
# ============================================================================


def test_get_polars_storage_options_none_protocol():
    """Test that empty dict is returned when protocol is None."""
    result = get_polars_storage_options(None)
    assert result == {}


def test_get_polars_storage_options_empty_protocol():
    """Test that empty dict is returned when protocol is empty string."""
    result = get_polars_storage_options("")
    assert result == {}


def test_get_polars_storage_options_unknown_protocol():
    """Test that empty dict is returned for unknown protocols."""
    result = get_polars_storage_options("unknown")
    assert result == {}


def test_get_polars_storage_options_gs():
    """Test that empty dict is returned for GCS (uses application default credentials)."""
    result = get_polars_storage_options("gs")
    assert result == {}


def test_get_polars_storage_options_s3_with_mock():
    """Test S3 storage options with mocked S3 config."""
    mock_s3_config = MagicMock()
    mock_s3_config.access_key_id = "test_access_key"
    mock_s3_config.secret_access_key = "test_secret_key"
    mock_s3_config.region = "us-west-2"
    mock_s3_config.endpoint = "http://localhost:9000"

    from flyte.storage import S3

    # Patch at the source module where get_storage is defined
    with patch("flyte._initialize.get_storage", return_value=mock_s3_config):
        with patch.object(S3, "auto", return_value=mock_s3_config):
            result = get_polars_storage_options("s3")
            assert result["aws_access_key_id"] == "test_access_key"
            assert result["aws_secret_access_key"] == "test_secret_key"
            assert result["aws_region"] == "us-west-2"
            assert result["aws_endpoint_url"] == "http://localhost:9000"
            assert "aws_skip_signature" not in result


def test_get_polars_storage_options_s3_anonymous():
    """Test S3 storage options with anonymous access."""
    mock_s3_config = MagicMock()
    mock_s3_config.access_key_id = None
    mock_s3_config.secret_access_key = None
    mock_s3_config.region = None
    mock_s3_config.endpoint = None

    from flyte.storage import S3

    # Patch at the source module where get_storage is defined
    with patch("flyte._initialize.get_storage", return_value=mock_s3_config):
        with patch.object(S3, "auto", return_value=mock_s3_config):
            result = get_polars_storage_options("s3", anonymous=True)
            assert result.get("aws_skip_signature") == "true"


def test_get_polars_storage_options_abfs_with_mock():
    """Test Azure Blob storage options with mocked ABFS config."""
    mock_abfs_config = MagicMock()
    mock_abfs_config.account_name = "test_account"
    mock_abfs_config.account_key = "test_key"
    mock_abfs_config.tenant_id = "test_tenant"
    mock_abfs_config.client_id = "test_client"
    mock_abfs_config.client_secret = "test_secret"

    from flyte.storage import ABFS

    # Patch at the source module where get_storage is defined
    with patch("flyte._initialize.get_storage", return_value=mock_abfs_config):
        with patch.object(ABFS, "auto", return_value=mock_abfs_config):
            result = get_polars_storage_options("abfs")
            assert result["azure_storage_account_name"] == "test_account"
            assert result["azure_storage_account_key"] == "test_key"
            assert result["azure_storage_tenant_id"] == "test_tenant"
            assert result["azure_storage_client_id"] == "test_client"
            assert result["azure_storage_client_secret"] == "test_secret"


def test_get_polars_storage_options_abfss():
    """Test that abfss protocol is handled same as abfs."""
    mock_abfs_config = MagicMock()
    mock_abfs_config.account_name = "test_account"
    mock_abfs_config.account_key = None
    mock_abfs_config.tenant_id = None
    mock_abfs_config.client_id = None
    mock_abfs_config.client_secret = None

    from flyte.storage import ABFS

    # Patch at the source module where get_storage is defined
    with patch("flyte._initialize.get_storage", return_value=mock_abfs_config):
        with patch.object(ABFS, "auto", return_value=mock_abfs_config):
            result = get_polars_storage_options("abfss")
            assert result["azure_storage_account_name"] == "test_account"


# ============================================================================
# Tests for handler registration and properties
# ============================================================================


def test_handler_properties():
    """Test that handler properties are correctly set."""
    encoder = PolarsToParquetEncodingHandler()
    assert encoder.python_type is pl.DataFrame
    assert encoder.protocol is None
    assert encoder.supported_format == PARQUET

    decoder = ParquetToPolarsDecodingHandler()
    assert decoder.python_type is pl.DataFrame
    assert decoder.protocol is None
    assert decoder.supported_format == PARQUET


def test_lazyframe_handler_properties():
    """Test that LazyFrame handler properties are correctly set."""
    encoder = PolarsLazyFrameToParquetEncodingHandler()
    assert encoder.python_type is pl.LazyFrame
    assert encoder.protocol is None
    assert encoder.supported_format == PARQUET

    decoder = ParquetToPolarsLazyFrameDecodingHandler()
    assert decoder.python_type is pl.LazyFrame
    assert decoder.protocol is None
    assert decoder.supported_format == PARQUET


def test_decoder_registered():
    """Test that decoder can be retrieved for polars types."""
    assert DataFrameTransformerEngine.get_decoder(pl.DataFrame, "file", PARQUET) is not None
    assert DataFrameTransformerEngine.get_decoder(pl.LazyFrame, "file", PARQUET) is not None


# ============================================================================
# Tests for annotated types with columns
# ============================================================================


def test_types_polars_dataframe_with_columns():
    """Test that polars DataFrame with column annotations is recognized."""
    my_cols = OrderedDict(name=str, age=int, city=str)
    pt = typing.Annotated[pl.DataFrame, my_cols]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert len(lt.structured_dataset_type.columns) == 3
    assert lt.structured_dataset_type.columns[0].name == "name"
    assert lt.structured_dataset_type.columns[1].name == "age"
    assert lt.structured_dataset_type.columns[2].name == "city"


def test_types_polars_dataframe_with_format():
    """Test that polars DataFrame with format annotation is recognized."""
    pt = typing.Annotated[pl.DataFrame, PARQUET]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert lt.structured_dataset_type.format == PARQUET


def test_types_polars_dataframe_with_columns_and_format():
    """Test that polars DataFrame with both columns and format is recognized."""
    my_cols = OrderedDict(name=str, age=int)
    pt = typing.Annotated[pl.DataFrame, my_cols, PARQUET]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert len(lt.structured_dataset_type.columns) == 2
    assert lt.structured_dataset_type.format == PARQUET


# ============================================================================
# Tests for column subsetting during decode
# ============================================================================


@pytest.mark.asyncio
async def test_dataframe_column_subsetting(ctx_with_test_raw_data_path, sample_dataframe):
    """Test that decoding with column annotations subsets the data."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(pl.DataFrame)

    # Encode the full dataframe
    lit = await fdt.to_literal(sample_dataframe, python_type=pl.DataFrame, expected=lt)

    # Decode with column subset annotation
    my_cols = OrderedDict(name=str, age=int)
    annotated_type = typing.Annotated[pl.DataFrame, my_cols]
    restored_df = await fdt.to_python_value(lit, expected_python_type=annotated_type)

    # Should only have the requested columns
    assert isinstance(restored_df, pl.DataFrame)
    assert set(restored_df.columns) == {"name", "age"}


# ============================================================================
# Tests for DataFrame wrapper usage
# ============================================================================


@pytest.mark.asyncio
async def test_flyte_dataframe_open_method(ctx_with_test_raw_data_path, sample_dataframe):
    """Test using DataFrame.open() method with polars."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(DataFrame)

    # Wrap in Flyte DataFrame and encode
    fdf = DataFrame.from_df(val=sample_dataframe)
    lit = await fdt.to_literal(fdf, python_type=DataFrame, expected=lt)

    # Decode as Flyte DataFrame
    restored_fdf = await fdt.to_python_value(lit, expected_python_type=DataFrame)
    assert isinstance(restored_fdf, DataFrame)

    # Use open() method to get polars DataFrame
    polars_df = await restored_fdf.open(pl.DataFrame).all()
    assert isinstance(polars_df, pl.DataFrame)
    assert polars_df.shape == sample_dataframe.shape


# ============================================================================
# Tests for various data types
# ============================================================================


@pytest.fixture
def sample_dataframe_with_various_types():
    """Create a sample polars DataFrame with various data types."""
    import datetime

    return pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "date_col": [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3)],
            "list_col": [[1, 2], [3, 4], [5, 6]],
        }
    )


@pytest.mark.asyncio
async def test_dataframe_with_various_types(ctx_with_test_raw_data_path, sample_dataframe_with_various_types):
    """Test roundtrip with various data types."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(pl.DataFrame)

    # Encode
    lit = await fdt.to_literal(sample_dataframe_with_various_types, python_type=pl.DataFrame, expected=lt)

    # Decode
    restored_df = await fdt.to_python_value(lit, expected_python_type=pl.DataFrame)

    # Compare
    assert restored_df.shape == sample_dataframe_with_various_types.shape
    assert restored_df.columns == sample_dataframe_with_various_types.columns
    # Verify data integrity for simple columns
    assert restored_df["int_col"].to_list() == sample_dataframe_with_various_types["int_col"].to_list()
    assert restored_df["str_col"].to_list() == sample_dataframe_with_various_types["str_col"].to_list()
    assert restored_df["bool_col"].to_list() == sample_dataframe_with_various_types["bool_col"].to_list()


# ============================================================================
# Tests for empty DataFrames
# ============================================================================


@pytest.mark.asyncio
async def test_empty_dataframe(ctx_with_test_raw_data_path):
    """Test roundtrip with empty DataFrame."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(pl.DataFrame)

    # Create empty DataFrame with schema
    empty_df = pl.DataFrame({"name": [], "age": []}).cast({"name": pl.Utf8, "age": pl.Int64})

    # Encode
    lit = await fdt.to_literal(empty_df, python_type=pl.DataFrame, expected=lt)

    # Decode
    restored_df = await fdt.to_python_value(lit, expected_python_type=pl.DataFrame)

    assert isinstance(restored_df, pl.DataFrame)
    assert restored_df.shape[0] == 0
    assert restored_df.columns == ["name", "age"]


@pytest.mark.asyncio
async def test_empty_lazyframe(ctx_with_test_raw_data_path):
    """Test roundtrip with empty LazyFrame."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(pl.LazyFrame)

    # Create empty LazyFrame with schema
    empty_lf = pl.DataFrame({"name": [], "value": []}).cast({"name": pl.Utf8, "value": pl.Float64}).lazy()

    # Encode
    lit = await fdt.to_literal(empty_lf, python_type=pl.LazyFrame, expected=lt)

    # Decode
    restored_lf = await fdt.to_python_value(lit, expected_python_type=pl.LazyFrame)

    assert isinstance(restored_lf, pl.LazyFrame)
    collected = restored_lf.collect()
    assert collected.shape[0] == 0
