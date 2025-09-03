from typing import Annotated

import numpy as np
import pandas as pd

import flyte.io

# Create task environment with required dependencies
img = flyte.Image.from_debian_base()
img = img.with_pip_packages("pandas", "pyarrow")

env = flyte.TaskEnvironment(
    "dataframe_usage",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

BASIC_EMPLOYEE_DATA = {
    "employee_id": range(1001, 1009),
    "name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah"],
    "department": ["HR", "Engineering", "Engineering", "Marketing", "Finance", "Finance", "HR", "Engineering"],
    "hire_date": pd.to_datetime(
        ["2018-01-15", "2019-03-22", "2020-07-10", "2017-11-01", "2021-06-05", "2018-09-13", "2022-01-07", "2020-12-30"]
    ),
}

ADDL_EMPLOYEE_DATA = {
    "employee_id": range(1001, 1009),
    "salary": [55000, 75000, 72000, 50000, 68000, 70000, np.nan, 80000],
    "bonus_pct": [0.05, 0.10, 0.07, 0.04, np.nan, 0.08, 0.03, 0.09],
    "full_time": [True, True, True, False, True, True, False, True],
    "projects": [
        ["Recruiting", "Onboarding"],
        ["Platform", "API"],
        ["API", "Data Pipeline"],
        ["SEO", "Ads"],
        ["Budget", "Forecasting"],
        ["Auditing"],
        [],
        ["Platform", "Security", "Data Pipeline"],
    ],
}


@env.task
async def create_raw_dataframe() -> pd.DataFrame:
    """
    This task creates a raw pandas DataFrame with basic employee information.
    This is the most basic use-case of how to pass dataframes (of all kinds, not just pandas). Create the dataframe
    as normal, and return it. Note that the output signature is of the dataframe library type.
    Uploading of the actual bits of the dataframe (which for pandas is serialized to parquet) happens at the
    end of the task, the TypeEngine uploads from memory to blob store.
    """
    return pd.DataFrame(BASIC_EMPLOYEE_DATA)


@env.task
async def create_flyte_dataframe() -> flyte.io.DataFrame:  # Annotated[flyte.io.DataFrame, "csv"]:
    """
    This task creates a Flyte DataFrame with compensation and project data.
    Because there's no generic type in Python that means any dataframe type, Flyte ships with its own. The
    flyte.io.DataFrame class is a thin wrapper around the various dataframe libraries (pandas, pyarrow, dask, etc).
    """
    pd_df = pd.DataFrame(ADDL_EMPLOYEE_DATA)

    fdf = flyte.io.DataFrame.create_from(pd_df)
    return fdf


@env.task
async def get_employee_data() -> pd.DataFrame:
    raw_dataframe = await create_raw_dataframe()
    flyte_dataframe = await create_flyte_dataframe()
    downloaded_fdf = await flyte_dataframe.open(pd.DataFrame).all()
    joined_df = raw_dataframe.merge(downloaded_fdf, on="employee_id", how="inner")

    return joined_df


if __name__ == "__main__":
    # Use local execution mode
    flyte.init()
    run = flyte.with_runcontext(mode="local").run(get_employee_data)
    print("Results:", run.outputs())
