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
async def create_flyte_dataframe() -> Annotated[flyte.io.DataFrame, "csv"]:
    """
    This task creates a Flyte DataFrame with compensation and project data.
    Because there's no generic type in Python that means any dataframe type, Flyte ships with its own. The
    flyte.io.DataFrame class is a thin wrapper around the various dataframe libraries (pandas, pyarrow, dask, etc).
    """
    pd_df = pd.DataFrame(ADDL_EMPLOYEE_DATA)

    fdf = flyte.io.DataFrame.from_df(pd_df)
    return fdf


@env.task
async def get_employee_data(raw_dataframe: pd.DataFrame, flyte_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This task takes two dataframes as input. We'll pass one raw pandas dataframe, and one flyte.io.DataFrame.
    Flyte automatically converts the flyte.io.DataFrame to a pandas DataFrame. The actual download and conversion
    happens only when we access the data (in this case, when we do the merge)."""
    joined_df = raw_dataframe.merge(flyte_dataframe, on="employee_id", how="inner")

    return joined_df


if __name__ == "__main__":
    flyte.init_from_config()  # log_level=logging.DEBUG)
    # Get the data sources

    local_df = pd.DataFrame(BASIC_EMPLOYEE_DATA)
    local_fdf = flyte.io.DataFrame.from_local_sync(local_df)
    run2 = flyte.with_runcontext(mode="remote").run(create_flyte_dataframe)
    run2.wait()

    # Pass both to get_employee_data - Flyte auto-converts flyte.io.DataFrame to pd.DataFrame
    run = flyte.with_runcontext(mode="remote").run(
        get_employee_data,
        raw_dataframe=local_fdf,
        flyte_dataframe=run2.outputs()[0],
    )
    print("Results:", run.url)
