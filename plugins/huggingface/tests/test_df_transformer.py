import typing
from collections import OrderedDict

import flyte
import pytest
from flyte.io._dataframe import DataFrame
from flyte.io._dataframe.dataframe import PARQUET, DataFrameTransformerEngine
from flyte.types import TypeEngine

# Import huggingface handlers to register them
import flyteplugins.huggingface.df_transformer  # noqa: F401
from flyteplugins.huggingface.df_transformer import (
    HuggingFaceDatasetToParquetEncodingHandler,
    ParquetToHuggingFaceDatasetDecodingHandler,
)

datasets = pytest.importorskip("datasets")
pd = pytest.importorskip("pandas")

# Sample data for testing
TEST_DATA = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["NYC", "SF", "LA"]}


@pytest.fixture
def sample_dataset():
    """Create a sample HuggingFace Dataset for testing."""
    return datasets.Dataset.from_pandas(pd.DataFrame(TEST_DATA))


# ============================================================================
# Type recognition tests
# ============================================================================


def test_types_huggingface_dataset():
    """Test that HuggingFace Dataset type is recognized."""
    pt = datasets.Dataset
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert lt.structured_dataset_type.format == ""
    assert lt.structured_dataset_type.columns == []


def test_types_dataset_with_columns():
    """Test that HuggingFace Dataset with column annotations is recognized."""
    my_cols = OrderedDict(name=str, age=int, city=str)
    pt = typing.Annotated[datasets.Dataset, my_cols]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert len(lt.structured_dataset_type.columns) == 3
    assert lt.structured_dataset_type.columns[0].name == "name"
    assert lt.structured_dataset_type.columns[1].name == "age"
    assert lt.structured_dataset_type.columns[2].name == "city"


def test_types_dataset_with_format():
    """Test that HuggingFace Dataset with format annotation is recognized."""
    pt = typing.Annotated[datasets.Dataset, PARQUET]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert lt.structured_dataset_type.format == PARQUET


def test_types_dataset_with_columns_and_format():
    """Test that HuggingFace Dataset with both columns and format is recognized."""
    my_cols = OrderedDict(name=str, age=int)
    pt = typing.Annotated[datasets.Dataset, my_cols, PARQUET]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert len(lt.structured_dataset_type.columns) == 2
    assert lt.structured_dataset_type.format == PARQUET


# ============================================================================
# Handler registration tests
# ============================================================================


def test_retrieving_encoder():
    """Test that encoders can be retrieved for HuggingFace Dataset."""
    assert DataFrameTransformerEngine.get_encoder(datasets.Dataset, "file", PARQUET) is not None
    assert DataFrameTransformerEngine.get_encoder(
        datasets.Dataset, "file", ""
    ) is DataFrameTransformerEngine.get_encoder(datasets.Dataset, "file", PARQUET)


def test_decoder_registered():
    """Test that decoder can be retrieved for HuggingFace Dataset."""
    assert DataFrameTransformerEngine.get_decoder(datasets.Dataset, "file", PARQUET) is not None


def test_handler_properties():
    """Test that handler properties are correctly set."""
    encoder = HuggingFaceDatasetToParquetEncodingHandler()
    assert encoder.python_type is datasets.Dataset
    assert encoder.protocol is None
    assert encoder.supported_format == PARQUET

    decoder = ParquetToHuggingFaceDatasetDecodingHandler()
    assert decoder.python_type is datasets.Dataset
    assert decoder.protocol is None
    assert decoder.supported_format == PARQUET


# ============================================================================
# Encode/decode roundtrip tests
# ============================================================================


@pytest.mark.asyncio
async def test_to_literal_dataset(ctx_with_test_raw_data_path, sample_dataset):
    """Test encoding HuggingFace Dataset to literal."""
    lt = TypeEngine.to_literal_type(datasets.Dataset)
    fdt = DataFrameTransformerEngine()

    lit = await fdt.to_literal(sample_dataset, python_type=datasets.Dataset, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET
    assert lit.scalar.structured_dataset.uri is not None

    restored = await fdt.to_python_value(lit, expected_python_type=datasets.Dataset)
    assert isinstance(restored, datasets.Dataset)
    assert len(restored) == len(sample_dataset)
    assert set(restored.column_names) == set(sample_dataset.column_names)


@pytest.mark.asyncio
async def test_dataset_roundtrip(ctx_with_test_raw_data_path, sample_dataset):
    """Test roundtrip encoding/decoding of HuggingFace Dataset."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)

    lit = await fdt.to_literal(sample_dataset, python_type=datasets.Dataset, expected=lt)
    restored = await fdt.to_python_value(lit, expected_python_type=datasets.Dataset)

    assert len(restored) == len(sample_dataset)
    assert restored.column_names == sample_dataset.column_names
    for col in sample_dataset.column_names:
        assert restored[col] == sample_dataset[col]


@pytest.mark.asyncio
async def test_dataset_through_flyte_dataframe(ctx_with_test_raw_data_path, sample_dataset):
    """Test using HuggingFace Dataset through Flyte DataFrame wrapper."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)

    fdf = DataFrame.from_df(val=sample_dataset)

    lit = await fdt.to_literal(fdf, python_type=datasets.Dataset, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET

    restored = await fdt.to_python_value(lit, expected_python_type=datasets.Dataset)
    assert isinstance(restored, datasets.Dataset)
    assert len(restored) == len(sample_dataset)


@pytest.mark.asyncio
async def test_raw_dataset_io(ctx_with_test_raw_data_path, sample_dataset):
    """Test using raw HuggingFace Dataset as task input/output."""
    flyte.init()
    env = flyte.TaskEnvironment(name="test-hf-dataset")

    @env.task
    async def process_dataset(ds: datasets.Dataset) -> datasets.Dataset:
        return ds.select(range(2))

    run = flyte.with_runcontext("local").run(process_dataset, sample_dataset)
    result = run.outputs()[0]
    assert isinstance(result, datasets.Dataset)
    assert len(result) == 2


# ============================================================================
# Column subsetting tests
# ============================================================================


@pytest.mark.asyncio
async def test_dataset_column_subsetting(ctx_with_test_raw_data_path, sample_dataset):
    """Test that decoding with column annotations subsets the data."""
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)

    lit = await fdt.to_literal(sample_dataset, python_type=datasets.Dataset, expected=lt)

    my_cols = OrderedDict(name=str, age=int)
    annotated_type = typing.Annotated[datasets.Dataset, my_cols]
    restored = await fdt.to_python_value(lit, expected_python_type=annotated_type)

    assert isinstance(restored, datasets.Dataset)
    assert set(restored.column_names) == {"name", "age"}


# ============================================================================
# Data type tests
# ============================================================================


@pytest.mark.asyncio
async def test_dataset_with_various_types(ctx_with_test_raw_data_path):
    """Test roundtrip with various data types."""
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )
    ds = datasets.Dataset.from_pandas(df)

    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)

    lit = await fdt.to_literal(ds, python_type=datasets.Dataset, expected=lt)
    restored = await fdt.to_python_value(lit, expected_python_type=datasets.Dataset)

    assert len(restored) == len(ds)
    assert restored["int_col"] == ds["int_col"]
    assert restored["str_col"] == ds["str_col"]
    assert restored["bool_col"] == ds["bool_col"]


@pytest.mark.asyncio
async def test_empty_dataset(ctx_with_test_raw_data_path):
    """Test roundtrip with empty Dataset."""
    empty_ds = datasets.Dataset.from_pandas(pd.DataFrame({"name": [], "age": []}))

    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)

    lit = await fdt.to_literal(empty_ds, python_type=datasets.Dataset, expected=lt)
    restored = await fdt.to_python_value(lit, expected_python_type=datasets.Dataset)

    assert isinstance(restored, datasets.Dataset)
    assert len(restored) == 0
    assert set(restored.column_names) == {"name", "age"}


