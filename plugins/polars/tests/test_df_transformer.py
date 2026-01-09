import typing

import flyte
import pytest
from flyte.io._dataframe import DataFrame
from flyte.io._dataframe.dataframe import PARQUET, DataFrameTransformerEngine
from flyte.types import TypeEngine

# Import polars handlers to register them
import flyteplugins.polars.df_transformer  # noqa: F401

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
    assert lt.structured_dataset_type.format == PARQUET
    assert lt.structured_dataset_type.columns == []


def test_types_polars_lazyframe():
    """Test that polars LazyFrame types are recognized."""
    pt = pl.LazyFrame
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert lt.structured_dataset_type.format == PARQUET
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

    a, b, c, d = extract_cols_and_format(pl.DataFrame)
    assert a is pl.DataFrame
    assert b is None
    assert c == PARQUET
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

    a, b, c, d = extract_cols_and_format(pl.LazyFrame)
    assert a is pl.LazyFrame
    assert b is None
    assert c == PARQUET
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
