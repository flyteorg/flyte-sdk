"""Test data file for DataFrame CLI input tests."""

import flyte
import flyte.io

pd = None
try:
    import pandas as pd
except ImportError:
    pass

env = flyte.TaskEnvironment(name="dataframe_inputs")


@env.task
async def process_pd_df(df: "pd.DataFrame") -> int:
    """Task that takes a pd.DataFrame input and returns row count."""
    return len(df)


@env.task
async def process_fdf(df: flyte.io.DataFrame) -> int:
    """Task that takes a flyte.io.DataFrame input and returns row count."""
    import pandas as pd

    pandas_df = await df.open(pd.DataFrame).all()
    return len(pandas_df)


@env.task
async def process_pd_df_return_df(df: "pd.DataFrame") -> "pd.DataFrame":
    """Task that takes a pd.DataFrame input and returns pd.DataFrame."""
    return df


@env.task
async def process_fdf_return_fdf(df: flyte.io.DataFrame) -> flyte.io.DataFrame:
    """Task that takes a flyte.io.DataFrame input and returns flyte.io.DataFrame."""
    return df
