import pytest

import flyte
from flyte.io._dataframe.dataframe import DataFrame
from flyte.types import TypeEngine

pd = pytest.importorskip("pandas")

# Sample data for testing
TEST_DATA = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["NYC", "SF", "LA"]}


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(TEST_DATA)


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
    result = run._outputs
    assert result.equals(sample_dataframe)
