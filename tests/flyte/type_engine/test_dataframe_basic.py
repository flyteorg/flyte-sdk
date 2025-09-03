import typing
from collections import OrderedDict

import pytest
from flyteidl.core import types_pb2
from fsspec.utils import get_protocol

import flyte
from flyte.io._dataframe import lazy_import_dataframe_handler
from flyte.io._dataframe.dataframe import (
    PARQUET,
    DataFrame,
    DataFrameEncoder,
    DataFrameTransformerEngine,
    extract_cols_and_format,
)
from flyte.types import TypeEngine

lazy_import_dataframe_handler()

pd = pytest.importorskip("pandas")

# Sample data for testing
TEST_DATA = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["NYC", "SF", "LA"]}


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(TEST_DATA)


def test_protocol():
    assert get_protocol("s3://my-s3-bucket/file") == "s3"
    assert get_protocol("/file") == "file"


def test_types_pandas():
    pt = pd.DataFrame
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert lt.structured_dataset_type.format == ""
    assert lt.structured_dataset_type.columns == []

    pt = typing.Annotated[pd.DataFrame, "csv"]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type.format == "csv"


def test_annotate_extraction():
    xyz = typing.Annotated[pd.DataFrame, "myformat"]
    a, b, c, d = extract_cols_and_format(xyz)
    assert a is pd.DataFrame
    assert b is None
    assert c == "myformat"
    assert d is None

    a, b, c, d = extract_cols_and_format(pd.DataFrame)
    assert a is pd.DataFrame
    assert b is None
    assert c == ""
    assert d is None


def test_types_annotated():
    my_cols = OrderedDict(w=typing.Dict[str, typing.Dict[str, int]], x=typing.List[typing.List[int]], y=int, z=str)
    pt = typing.Annotated[pd.DataFrame, my_cols]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 4
    assert (
        lt.structured_dataset_type.columns[0].literal_type.map_value_type.map_value_type.simple
        == types_pb2.SimpleType.INTEGER
    )
    assert (
        lt.structured_dataset_type.columns[1].literal_type.collection_type.collection_type.simple
        == types_pb2.SimpleType.INTEGER
    )
    assert lt.structured_dataset_type.columns[2].literal_type.simple == types_pb2.SimpleType.INTEGER
    assert lt.structured_dataset_type.columns[3].literal_type.simple == types_pb2.SimpleType.STRING

    pt = typing.Annotated[pd.DataFrame, OrderedDict(a=None)]
    with pytest.raises(AssertionError, match="type None is currently not supported by DataFrame"):
        TypeEngine.to_literal_type(pt)


def test_types_sd():
    my_cols = OrderedDict(w=typing.Dict[str, typing.Dict[str, int]], x=typing.List[typing.List[int]], y=int, z=str)
    pt = DataFrame
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None

    pt = typing.Annotated[DataFrame, my_cols]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 4

    pt = typing.Annotated[DataFrame, my_cols, "csv"]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 4
    assert lt.structured_dataset_type.format == "csv"

    pt = typing.Annotated[DataFrame, {}, "csv"]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 0
    assert lt.structured_dataset_type.format == "csv"


@pytest.mark.asyncio
async def test_passthrough_df_no_io_pure_local(sample_dataframe):
    """
    Take in a DF and return it directly. This is purely local, shouldn't involve Flyte at all, so trivially the same.
    This should NOT trigger the engine or i/o - just pass through.
    """
    env = flyte.TaskEnvironment(name="test-passthrough")

    @env.task
    async def passthrough_task(df: DataFrame) -> DataFrame:
        # Just return the DataFrame directly - no processing
        return df

    input_df = DataFrame.from_existing_remote("s3://test-bucket/doesnotexist.parquet", format="parquet")

    result = await passthrough_task(input_df)
    # Should return the same DataFrame reference
    assert result is input_df
    assert result.uri == input_df.uri
    assert result.format == input_df.format


@pytest.mark.asyncio
async def test_passthrough_df_no_io(sample_dataframe):
    """
    Take in a DF and return it directly.
    This should NOT trigger the engine or i/o - just pass through.
    """
    flyte.init()
    env = flyte.TaskEnvironment(name="test-passthrough")

    @env.task
    async def passthrough_task(df: DataFrame) -> DataFrame:
        # Just return the DataFrame directly - no processing
        return df

    input_df = DataFrame.from_existing_remote("s3://test-bucket/doesnotexist.parquet", format="parquet")

    run = flyte.with_runcontext("local").run(passthrough_task, input_df)
    result = run._outputs
    lit = await TypeEngine.to_literal(result, DataFrame, TypeEngine.to_literal_type(DataFrame))
    assert lit.scalar.structured_dataset.uri == input_df.uri
    assert result.format == input_df.format
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == input_df.format


def test_retrieving():
    class MyDF(pd.DataFrame): ...

    assert DataFrameTransformerEngine.get_encoder(pd.DataFrame, "file", PARQUET) is not None
    # Asking for a generic means you're okay with any one registered for that
    # type assuming there's just one.
    assert DataFrameTransformerEngine.get_encoder(pd.DataFrame, "file", "") is DataFrameTransformerEngine.get_encoder(
        pd.DataFrame, "file", PARQUET
    )

    class TempEncoder(DataFrameEncoder):
        def __init__(self, protocol):
            super().__init__(MyDF, protocol)

        def encode(self): ...

    DataFrameTransformerEngine.register(TempEncoder("gs"), default_for_type=False)
    with pytest.raises(ValueError):
        DataFrameTransformerEngine.register(TempEncoder("gs://"), default_for_type=False)

    with pytest.raises(ValueError, match="Use None instead"):
        e = TempEncoder("")
        e._protocol = ""
        DataFrameTransformerEngine.register(e)

    class TempEncoder:
        pass

    with pytest.raises(TypeError, match="We don't support this type of handler"):
        DataFrameTransformerEngine.register(TempEncoder, default_for_type=False)


@pytest.mark.asyncio
async def test_to_literal(ctx_with_test_raw_data_path, sample_dataframe):
    lt = TypeEngine.to_literal_type(pd.DataFrame)

    fdt = DataFrameTransformerEngine()

    lit = await fdt.to_literal(sample_dataframe, python_type=pd.DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET

    sd_with_literal_and_df = DataFrame.create_from(val=sample_dataframe)
    sd_with_literal_and_df._literal_sd = lit

    with pytest.raises(ValueError, match="Shouldn't have specified both literal"):
        await fdt.to_literal(sd_with_literal_and_df, python_type=DataFrame, expected=lt)

    sd_with_nothing = DataFrame()
    with pytest.raises(ValueError, match="If dataframe is not specified"):
        await fdt.to_literal(sd_with_nothing, python_type=DataFrame, expected=lt)

    sd_with_uri = DataFrame.from_existing_remote(remote_path="s3://some/extant/df.parquet")

    lt = TypeEngine.to_literal_type(typing.Annotated[DataFrame, {}, "new-df-format"])
    lit = await fdt.to_literal(sd_with_uri, python_type=DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.uri == "s3://some/extant/df.parquet"
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == "new-df-format"


@pytest.mark.asyncio
async def test_to_literal_through_df_with_format(ctx_with_test_raw_data_path, sample_dataframe):
    lt = TypeEngine.to_literal_type(typing.Annotated[pd.DataFrame, "csv"])
    fdf = DataFrame.create_from(val=sample_dataframe)

    fdt = DataFrameTransformerEngine()

    lit = await fdt.to_literal(fdf, python_type=pd.DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == "csv"

    # go backwards to get a python value
    py_val = await fdt.to_python_value(lit, expected_python_type=DataFrame)
    restored_df = await py_val.open(pd.DataFrame).all()

    restored_df_2 = await fdt.to_python_value(lit, expected_python_type=pd.DataFrame)
    assert restored_df.equals(restored_df_2)
    assert restored_df.equals(sample_dataframe)


@pytest.mark.asyncio
async def test_raw_df_io_triggers_engine(sample_dataframe, ctx_with_test_raw_data_path):
    """
    Use case 3: Return a raw df, take in a raw df.
    Taking in a raw df and returning it directly should trigger i/o and the engine.
    """
    flyte.init()
    env = flyte.TaskEnvironment(name="test-raw-df")

    @env.task
    async def process_raw_df(df: pd.DataFrame) -> pd.DataFrame:
        return df

    run = flyte.with_runcontext("local").run(process_raw_df, sample_dataframe)
    result = run.outputs()
    assert result.equals(sample_dataframe)
