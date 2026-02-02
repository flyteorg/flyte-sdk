"""
Example demonstrating Polars DataFrame and LazyFrame usage in Flyte.

This example shows:
1. Using polars.DataFrame as task input/output
2. Using polars.LazyFrame as task input/output
3. Performing data transformations with both eager and lazy evaluation
"""

import polars as pl

import flyte

# Create task environment with required dependencies
img = flyte.Image.from_debian_base(name="flyteplugins-polars-image").with_pip_packages(
    "flyteplugins-polars>=2.0.0b52", "flyte>=2.0.0b52", pre=True
)

env = flyte.TaskEnvironment(
    "polars_dataframe",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

# Sample employee data
EMPLOYEE_DATA = {
    "employee_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah"],
    "department": ["HR", "Engineering", "Engineering", "Marketing", "Finance", "Finance", "HR", "Engineering"],
    "salary": [55000, 75000, 72000, 50000, 68000, 70000, 52000, 80000],
    "years_experience": [3, 5, 4, 2, 6, 5, 1, 7],
}


@env.task
async def create_polars_dataframe() -> pl.DataFrame:
    """
    This task creates a raw polars DataFrame with employee information.
    Polars DataFrames use eager evaluation - operations are executed immediately.
    """
    return pl.DataFrame(EMPLOYEE_DATA)


@env.task
async def process_dataframe_eager(df: pl.DataFrame) -> pl.DataFrame:
    """
    This task takes a polars DataFrame as input and performs eager transformations.
    Eager evaluation means operations are executed immediately.
    """
    # Filter employees with salary > 60000
    filtered = df.filter(pl.col("salary") > 60000)

    # Add a computed column
    result = filtered.with_columns((pl.col("salary") / pl.col("years_experience")).alias("salary_per_year"))

    # Sort by salary descending
    result = result.sort("salary", descending=True)

    return result


@env.task
async def create_polars_lazyframe() -> pl.LazyFrame:
    """
    This task creates a polars LazyFrame.
    LazyFrames use lazy evaluation - operations are optimized and executed only when collected.
    This is more efficient for complex queries and large datasets.
    """
    return pl.LazyFrame(EMPLOYEE_DATA)


@env.task
async def process_lazyframe(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    This task takes a polars LazyFrame as input and performs lazy transformations.
    The operations are not executed until the LazyFrame is collected.
    This allows Polars to optimize the query plan.
    """
    print(lf.collect())
    # Build a query plan (not executed yet)
    result = (
        lf.filter(pl.col("salary") > 60000)
        .with_columns((pl.col("salary") / pl.col("years_experience")).alias("salary_per_year"))
        .sort("salary", descending=True)
        .select(["name", "department", "salary", "salary_per_year"])
    )

    # Return the LazyFrame - execution happens when collected
    return result


@env.task
async def combine_dataframes(df1: pl.DataFrame, df2: pl.LazyFrame) -> pl.DataFrame:
    """
    This task demonstrates mixing polars DataFrame and LazyFrame.
    The LazyFrame will be collected (evaluated) when used in operations with the DataFrame.
    """
    # Collect the lazy frame to eager
    df2_eager = df2.collect()

    # Join the two dataframes
    # Note: This is a simple example - in practice you'd join on a common key
    # For demonstration, we'll just concatenate them
    combined = pl.concat([df1, df2_eager])

    return combined


@env.task
async def aggregate_with_lazyframe(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    This task demonstrates using LazyFrame for efficient aggregations.
    The aggregation is part of the query plan and optimized by Polars.
    """
    # Build an optimized aggregation query
    aggregated = (
        lf.group_by("department")
        .agg(
            [
                pl.col("salary").mean().alias("avg_salary"),
                pl.col("salary").max().alias("max_salary"),
                pl.col("salary").min().alias("min_salary"),
                pl.len().alias("employee_count"),
            ]
        )
        .sort("avg_salary", descending=True)
    )

    # Collect to execute the query
    return aggregated.collect()


if __name__ == "__main__":
    import logging

    import flyte.storage

    # make sure to set the following environment variables:
    # - AWS_ACCESS_KEY_ID
    # - AWS_SECRET_ACCESS_KEY
    # - AWS_SESSION_TOKEN (if applicable)
    #
    # You may also set this with `aws sso login`:
    # $ aws sso login --profile $profile
    # $ eval "$(aws configure export-credentials --profile $profile --format env)"

    flyte.init_from_config(
        log_level=logging.DEBUG,
        storage=flyte.storage.S3.auto(region="us-east-2"),
    )

    # Example 1: Using polars DataFrame (eager evaluation)
    print("Example 1: Polars DataFrame (eager evaluation)")
    df_run = flyte.run(create_polars_dataframe)
    print(df_run.url)
    print(df_run.wait())
    df_result: pl.DataFrame = df_run.outputs()[0]
    print(f"Created DataFrame: {df_result}")

    processed_df_run = flyte.run(process_dataframe_eager, df=df_result)
    print(processed_df_run.url)
    print(processed_df_run.wait())
    processed_result = processed_df_run.outputs()[0]
    print(f"\nProcessed DataFrame: {processed_result}")

    # Example 2: Using polars LazyFrame (lazy evaluation)
    print("\n\nExample 2: Polars LazyFrame (lazy evaluation)")
    lf_run = flyte.run(create_polars_lazyframe)
    print(lf_run.url)
    print(lf_run.wait())
    lf_result = lf_run.outputs()[0]
    print(f"Created LazyFrame: {lf_result}")

    processed_lf_run = flyte.run(process_lazyframe, lf=lf_result)
    print(processed_lf_run.url)
    print(processed_lf_run.wait())
    processed_lf_result = processed_lf_run.outputs()[0]
    print(f"Processed LazyFrame: {processed_lf_result}")

    # Example 3: Aggregation with LazyFrame
    print("\n\nExample 3: Aggregation with LazyFrame")
    agg_run = flyte.run(aggregate_with_lazyframe, lf=lf_result)
    print(agg_run.url)
    print(agg_run.wait())
    agg_output = agg_run.outputs()[0]
    print(f"Aggregated results by department: {agg_output}")

    # Example 4: Combining DataFrame and LazyFrame
    print("\n\nExample 4: Combining DataFrame and LazyFrame")
    combined_run = flyte.run(combine_dataframes, df1=df_result, df2=lf_result)
    print(combined_run.url)
    print(combined_run.wait())
    combined_output = combined_run.outputs()[0]
    print(f"Combined DataFrame: {combined_output}")
