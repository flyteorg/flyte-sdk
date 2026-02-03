"""Example for passing Polars DataFrames and LazyFrames as inputs to a task.

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
python polars_dataframe_inputs.py
```

Or run with `flyte run`:

```
# Create the dataframes and write them to disk. This will create parquet files.
python create_polars_dataframe.py

# Run the task on a single dataframe
flyte run polars_dataframe_inputs.py process_df --df ./dataframe.parquet

# Run the task on a lazyframe
flyte run polars_dataframe_inputs.py process_lf --lf ./lazyframe.parquet

# Run the task that takes in a flyte.io.DataFrame and returns a polars DataFrame
flyte run polars_dataframe_inputs.py process_fdf_to_df --df ./dataframe.parquet

# Run the task that takes in a flyte.io.DataFrame and returns a polars LazyFrame
flyte run polars_dataframe_inputs.py process_fdf_to_lf --df ./lazyframe.parquet
```
"""

import polars as pl

import flyte
import flyte.io

# Create task environment with required dependencies
img = flyte.Image.from_debian_base(name="flyteplugins-polars-image").with_pip_packages(
    "flyteplugins-polars>=2.0.0b52", "flyte>=2.0.0b52", pre=True
)

env = flyte.TaskEnvironment(
    "polars_dataframe_inputs",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


# =============================================================================
# Polars DataFrame examples
# =============================================================================


@env.task
def process_df(df: pl.DataFrame) -> pl.DataFrame:
    """Process a Polars DataFrame and return it."""
    return df


@env.task
def process_fdf_to_df(df: flyte.io.DataFrame) -> pl.DataFrame:
    """Accept a Flyte DataFrame and return a Polars DataFrame."""
    return df


@env.task
def process_df_to_fdf(df: pl.DataFrame) -> flyte.io.DataFrame:
    """Accept a Polars DataFrame and return a Flyte DataFrame."""
    return flyte.io.DataFrame.from_df(df)


@env.task
def process_fdf_to_fdf(df: flyte.io.DataFrame) -> flyte.io.DataFrame:
    """Accept and return a Flyte DataFrame (passthrough)."""
    return df


# =============================================================================
# Polars LazyFrame examples
# =============================================================================


@env.task
def process_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Process a Polars LazyFrame and return it."""
    return lf


@env.task
def process_fdf_to_lf(df: flyte.io.DataFrame) -> pl.LazyFrame:
    """Accept a Flyte DataFrame and return a Polars LazyFrame."""
    return df


@env.task
def process_lf_to_fdf(lf: pl.LazyFrame) -> flyte.io.DataFrame:
    """Accept a Polars LazyFrame and return a Flyte DataFrame."""
    return flyte.io.DataFrame.from_df(lf)


# =============================================================================
# Mixed DataFrame and LazyFrame examples
# =============================================================================


@env.task
def transform_df_to_lf(df: pl.DataFrame) -> pl.LazyFrame:
    """Accept a Polars DataFrame and return a LazyFrame."""
    return df.lazy()


@env.task
def transform_lf_to_df(lf: pl.LazyFrame) -> pl.DataFrame:
    """Accept a Polars LazyFrame and return a DataFrame by collecting."""
    return lf.collect()


@env.task
def filter_and_aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """Demonstrate filtering and aggregation with Polars DataFrame."""
    return (
        df.filter(pl.col("active"))
        .group_by("category")
        .agg(
            [
                pl.col("salary").mean().alias("avg_salary"),
                pl.col("age").mean().alias("avg_age"),
                pl.len().alias("count"),
            ]
        )
        .sort("category")
    )


@env.task
def lazy_filter_and_aggregate(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Demonstrate filtering and aggregation with Polars LazyFrame."""
    return (
        lf.filter(pl.col("active"))
        .group_by("category")
        .agg(
            [
                pl.col("salary").mean().alias("avg_salary"),
                pl.col("age").mean().alias("avg_age"),
                pl.len().alias("count"),
            ]
        )
        .sort("category")
    )


if __name__ == "__main__":
    import flyte.storage

    flyte.init_from_config(
        storage=flyte.storage.S3.auto(region="us-east-2"),
    )

    # Create sample data
    dataframe = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
            "age": [25, 30, 35, 40, 28, 33],
            "category": ["A", "B", "A", "C", "B", "C"],
            "salary": [55000.0, 75000.0, 72000.0, 50000.0, 68000.0, 70000.0],
            "active": [True, False, True, True, True, False],
        }
    )

    lazyframe = dataframe.lazy()

    # ==========================================================================
    # DataFrame examples
    # ==========================================================================
    print("=" * 60)
    print("Polars DataFrame Examples")
    print("=" * 60)

    # Example 1: Process DataFrame
    print("\n1. Processing Polars DataFrame...")
    run = flyte.run(process_df, df=dataframe)
    print(f"   URL: {run.url}")
    run.wait()
    result = run.outputs()[0]
    print(f"   Result:\n{result}\n")

    # Example 2: Flyte DataFrame to Polars DataFrame
    print("2. Flyte DataFrame -> Polars DataFrame...")
    flyte_dataframe = flyte.io.DataFrame.from_df(dataframe)
    run = flyte.run(process_fdf_to_df, df=flyte_dataframe)
    print(f"   URL: {run.url}")
    run.wait()
    result: pl.DataFrame = run.outputs()[0]
    print(f"   Result:\n{result}\n")

    # Example 3: Polars DataFrame to Flyte DataFrame
    print("3. Polars DataFrame -> Flyte DataFrame...")
    run = flyte.run(process_df_to_fdf, df=dataframe)
    print(f"   URL: {run.url}")
    run.wait()
    result: flyte.io.DataFrame = run.outputs()[0]
    print(f"   Result:\n{result}\n")

    # Example 4: Flyte DataFrame passthrough
    print("4. Flyte DataFrame passthrough...")
    run = flyte.run(process_fdf_to_fdf, df=flyte_dataframe)
    print(f"   URL: {run.url}")
    run.wait()
    result: flyte.io.DataFrame = run.outputs()[0]
    print(f"   Result:\n{result}\n")

    # ==========================================================================
    # LazyFrame examples
    # ==========================================================================
    print("=" * 60)
    print("Polars LazyFrame Examples")
    print("=" * 60)

    # Example 5: Process LazyFrame
    print("\n5. Processing Polars LazyFrame...")
    run = flyte.run(process_lf, lf=lazyframe)
    print(f"   URL: {run.url}")
    run.wait()
    result = run.outputs()[0]
    print(f"   Result (collected):\n{result.collect()}\n")

    # Example 6: Flyte DataFrame to Polars LazyFrame
    print("6. Flyte DataFrame -> Polars LazyFrame...")
    flyte_dataframe_for_lf = flyte.io.DataFrame.from_df(lazyframe)
    run = flyte.run(process_fdf_to_lf, df=flyte_dataframe_for_lf)
    print(f"   URL: {run.url}")
    run.wait()
    result: pl.LazyFrame = run.outputs()[0]
    print(f"   Result (collected):\n{result.collect()}\n")

    # Example 7: Polars LazyFrame to Flyte DataFrame
    print("7. Polars LazyFrame -> Flyte DataFrame...")
    run = flyte.run(process_lf_to_fdf, lf=lazyframe)
    print(f"   URL: {run.url}")
    run.wait()
    result: flyte.io.DataFrame = run.outputs()[0]
    print(f"   Result:\n{result}\n")

    # ==========================================================================
    # Transformation examples
    # ==========================================================================
    print("=" * 60)
    print("DataFrame <-> LazyFrame Transformation Examples")
    print("=" * 60)

    # Example 8: DataFrame to LazyFrame
    print("\n8. Transform DataFrame -> LazyFrame...")
    run = flyte.run(transform_df_to_lf, df=dataframe)
    print(f"   URL: {run.url}")
    run.wait()
    result = run.outputs()[0]
    print(f"   Result (collected):\n{result.collect()}\n")

    # Example 9: LazyFrame to DataFrame
    print("9. Transform LazyFrame -> DataFrame...")
    run = flyte.run(transform_lf_to_df, lf=lazyframe)
    print(f"   URL: {run.url}")
    run.wait()
    result = run.outputs()[0]
    print(f"   Result:\n{result}\n")

    # ==========================================================================
    # Aggregation examples
    # ==========================================================================
    print("=" * 60)
    print("Aggregation Examples")
    print("=" * 60)

    # Example 10: Filter and aggregate DataFrame
    print("\n10. Filter and aggregate DataFrame...")
    run = flyte.run(filter_and_aggregate, df=dataframe)
    print(f"   URL: {run.url}")
    run.wait()
    result = run.outputs()[0]
    print(f"   Result:\n{result}\n")

    # Example 11: Filter and aggregate LazyFrame
    print("11. Filter and aggregate LazyFrame...")
    run = flyte.run(lazy_filter_and_aggregate, lf=lazyframe)
    print(f"   URL: {run.url}")
    run.wait()
    result = run.outputs()[0]
    print(f"   Result (collected):\n{result.collect()}\n")

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
