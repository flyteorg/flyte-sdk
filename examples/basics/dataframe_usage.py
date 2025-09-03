from typing import Annotated

import pandas as pd
import numpy as np
import flyte.io


# Create task environment with required dependencies
img = flyte.Image.from_debian_base()
img = img.with_pip_packages("pandas", "pyarrow")

env = flyte.TaskEnvironment(
    "dataframe_usage",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

# Simple sample data
SAMPLE_DATA = {
    "employee_id": range(1001, 1009),
    "name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah"],
    "department": ["HR", "Engineering", "Engineering", "Marketing", "Finance", "Finance", "HR", "Engineering"],
    "hire_date": pd.to_datetime([
        "2018-01-15", "2019-03-22", "2020-07-10", "2017-11-01",
        "2021-06-05", "2018-09-13", "2022-01-07", "2020-12-30"
    ]),
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
        ["Platform", "Security", "Data Pipeline"]
    ]
}


@env.task
async def create_raw_dataframe() -> pd.DataFrame:
    """
    This is the most basic use-case of how to pass dataframes (of all kinds, not just pandas). Create the dataframe
    as normal, and return it. Note that the output signature is of the dataframe library type.
    Uploading of the actual bits of the dataframe (which for pandas is serialized to parquet) happens at the
    end of the task, the TypeEngine uploads from memory to blob store.
    """
    return pd.DataFrame(SAMPLE_DATA)


# create a flyte dataframe object, and show what controls can be done (change the format)
@env.task
async def create_flyte_dataframe() -> Annotated[flyte.io.DataFrame, "csv"]:
    """
    Flyte ships with its own dataframe type.
    """
    pd_df = pd.DataFrame(SAMPLE_DATA)

    fdf = flyte.io.DataFrame.create_from(pd_df)
    return fdf


@env.task
async def wrapper_to_raw(df: flyte.io.DataFrame) -> pd.DataFrame:
    """
    Input: flyte.DataFrame wrapper -> Output: raw pd.DataFrame
    DOWNLOAD happens: df.open().all() downloads remote data to pandas
    UPLOAD happens: returning pandas triggers automatic upload
    """
    # DOWNLOAD: wrapper -> raw pandas (I/O occurs)
    pandas_df = await df.open(pd.DataFrame).all()
    print(f"Downloaded pandas dataframe:\n{pandas_df}", flush=True)

    # Process the raw dataframe
    pandas_df["processed"] = True
    print(f"With processed column:\n{pandas_df}", flush=True)

    # UPLOAD: raw pandas is automatically uploaded at task completion
    return pandas_df


@env.task
async def wrapper_passthrough(df: flyte.io.DataFrame) -> flyte.io.DataFrame:
    """
    Input: flyte.DataFrame wrapper -> Output: flyte.DataFrame wrapper
    If you happen to not perform any operations on the DataFrame, no I/O happens.
    """
    print(f"No I/O - just passing wrapper reference: {df.uri}", flush=True)
    return df


@env.task
async def raw_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: raw pd.DataFrame -> Output: raw pd.DataFrame
    DOWNLOAD happens: input pandas is downloaded from remote storage
    UPLOAD happens: output pandas is uploaded at task completion
    """
    # Input df is already downloaded raw pandas
    df["doubled_age"] = df["age"] * 2
    print(f"Pandas dataframe with doubled age:\n{df}", flush=True)

    # Output will be uploaded automatically
    return df


@env.task
async def main_workflow() -> dict:
    """
    Demonstrates when I/O happens vs when it doesn't.
    """
    # Pattern 1: Create raw dataframe (UPLOAD occurs)
    raw_df = await create_raw_dataframe()
    print(f"Created raw dataframe:\n{raw_df}", flush=True)

    # Pattern 2: Create wrapper dataframe (NO I/O)
    wrapped_df_from_existing_data = await create_wrapper_dataframe()
    print("Created wrapper dataframe", flush=True)

    # Pattern 3: Wrapper metadata access (NO I/O)
    metadata = await inspect_wrapper_metadata(wrapped_df_from_existing_data)
    print(f"DataFrame wrapper metadata: {metadata}", flush=True)

    # Pattern 4: Wrapper -> Raw (DOWNLOAD + UPLOAD)
    processed_raw = await wrapper_to_raw(wrapped_df_from_existing_data)

    # Pattern 5: Wrapper -> Wrapper (NO I/O)
    same_wrapper = await wrapper_passthrough(wrapped_df_from_existing_data)

    # Pattern 6: Raw -> Raw (DOWNLOAD + UPLOAD)
    doubled_raw = await raw_to_raw(raw_df)

    return {
        "raw_df_type": str(type(raw_df)),
        "wrapper_df_uri": wrapped_df_from_existing_data.uri,
        "metadata": metadata,
        "processed_type": str(type(processed_raw)),
        "passthrough_uri": same_wrapper.uri,
        "doubled_type": str(type(doubled_raw)),
        "message": "All dataframes uploaded at task completion!",
    }


if __name__ == "__main__":
    # Use local execution mode
    run = flyte.with_runcontext(mode="local").run(main_workflow)

    print("DataFrame wrapper vs raw I/O patterns demonstrated!")
    print("Key takeaway: flyte.DataFrame = wrapper (I/O on demand)")
    print("             pd.DataFrame = raw (automatic I/O)")
    print("Results:", run.outputs())
