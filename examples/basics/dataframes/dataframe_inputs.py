"""Example for passing raw dataframes as inputs to a task.

Prerequisites: make sure to set the following environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_SESSION_TOKEN (if applicable)

You may also set this with `aws sso login`:

```
$ aws sso login --profile $profile
$ eval "$(aws configure export-credentials --profile $profile --format env)"
```

Run this script to run the task as a python script.

```
python dataframe_inputs.py
```

Or run with `flyte run`:

```
# Create the dataframes and write them to disk. This will create a single parquet file and a directory of parquet files.
python create_dataframe.py

# Run the task on a single dataframe
flyte run dataframe_inputs.py process_df --df ./dataframe.parquet
```

# Run the task on a directory of dataframes
flyte run dataframe_inputs.py process_df --df ./dataframe_partitioned

# Run the task that takes in a flyte.io.DataFrame
flyte run dataframe_inputs.py process_fdf_to_df --df ./dataframe.parquet

# Run the task that takes in a flyte.io.DataFrame
flyte run dataframe_inputs.py process_fdf_to_df --df ./dataframe_partitioned
"""

import pandas as pd

import flyte
import flyte.io

# Create task environment with required dependencies
img = flyte.Image.from_debian_base()
img = img.with_pip_packages("pandas", "pyarrow")

env = flyte.TaskEnvironment(
    "dataframe_inputs",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


@env.task
def local_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "category": ["A", "B", "A", "C"],
            "active": [True, False, True, True],
        }
    )


@env.task
def process_df(df: pd.DataFrame) -> pd.DataFrame:
    return df


@env.task
def process_fdf_to_df(df: flyte.io.DataFrame) -> pd.DataFrame:
    return df


@env.task
def process_df_to_fdf(df: pd.DataFrame) -> flyte.io.DataFrame:
    return flyte.io.DataFrame.from_df(df)


@env.task
def process_fdf_to_fdf(df: flyte.io.DataFrame) -> flyte.io.DataFrame:
    return df


if __name__ == "__main__":
    import flyte.storage

    flyte.init_from_config(
        storage=flyte.storage.S3.auto(region="us-east-2"),
    )

    in_mem_dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "category": ["A", "B", "A", "C"],
            "active": [True, False, True, True],
        }
    )
    local_run = flyte.with_runcontext(mode="local").run(local_df)
    local_task_df = local_run.outputs()[0]

    for dataframe in [in_mem_dataframe, local_task_df]:
        run = flyte.with_runcontext(preserve_original_types=True).run(process_df, df=dataframe)
        print(run.url)
        run.wait()
        result = run.outputs()[0]
        assert isinstance(result, pd.DataFrame)
        print(result)

        flyte_dataframe = flyte.io.DataFrame.from_df(dataframe)
        run = flyte.with_runcontext(preserve_original_types=True).run(process_fdf_to_df, df=flyte_dataframe)
        print(run.url)
        run.wait()
        result: pd.DataFrame = run.outputs()[0]
        assert isinstance(result, pd.DataFrame)
        print(result)

        run = flyte.with_runcontext(preserve_original_types=True).run(process_df_to_fdf, df=dataframe)
        print(run.url)
        run.wait()
        result: flyte.io.DataFrame = run.outputs()[0]
        assert isinstance(result, flyte.io.DataFrame)
        print(result)

        run = flyte.with_runcontext(preserve_original_types=True).run(process_fdf_to_fdf, df=flyte_dataframe)
        print(run.url)
        run.wait()
        result: flyte.io.DataFrame = run.outputs()[0]
        assert isinstance(result, flyte.io.DataFrame)
        print(result)
