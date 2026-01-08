"""
Example demonstrating Polars DataFrame and LazyFrame usage in Flyte.

This example shows:
1. Using polars.DataFrame as task input/output
2. Using polars.LazyFrame as task input/output
3. Performing data transformations with both eager and lazy evaluation
"""

import flyte
import polars as pl

# Create task environment with required dependencies
img = flyte.Image.from_debian_base()
img = img.with_pip_packages("polars", "flyteplugins-polars")

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
    result = filtered.with_columns(
        (pl.col("salary") / pl.col("years_experience")).alias("salary_per_year")
    )

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
                pl.count().alias("employee_count"),
            ]
        )
        .sort("avg_salary", descending=True)
    )

    # Collect to execute the query
    return aggregated.collect()


if __name__ == "__main__":
    flyte.init_from_config()

    # Example 1: Using polars DataFrame (eager evaluation)
    print("Example 1: Polars DataFrame (eager evaluation)")
    df_task = flyte.with_runcontext(mode="local").run(create_polars_dataframe)
    df_result = df_task.outputs()
    print(f"Created DataFrame with shape: {df_result.shape}")
    print(df_result.head())

    processed_df = flyte.with_runcontext(mode="local").run(process_dataframe_eager, df=df_result)
    processed_result = processed_df.outputs()
    print(f"\nProcessed DataFrame with shape: {processed_result.shape}")
    print(processed_result)

    # Example 2: Using polars LazyFrame (lazy evaluation)
    print("\n\nExample 2: Polars LazyFrame (lazy evaluation)")
    lf_task = flyte.with_runcontext(mode="local").run(create_polars_lazyframe)
    lf_result = lf_task.outputs()
    print(f"Created LazyFrame (not yet evaluated)")

    processed_lf = flyte.with_runcontext(mode="local").run(process_lazyframe, lf=lf_result)
    processed_lf_result = processed_lf.outputs()
    print(f"Processed LazyFrame (still not evaluated)")
    # Collect to see the results
    collected = processed_lf_result.collect()
    print(f"After collection, shape: {collected.shape}")
    print(collected)

    # Example 3: Aggregation with LazyFrame
    print("\n\nExample 3: Aggregation with LazyFrame")
    agg_result = flyte.with_runcontext(mode="local").run(aggregate_with_lazyframe, lf=lf_result)
    agg_output = agg_result.outputs()
    print("Aggregated results by department:")
    print(agg_output)

    # Example 4: Combining DataFrame and LazyFrame
    print("\n\nExample 4: Combining DataFrame and LazyFrame")
    combined_result = flyte.with_runcontext(mode="local").run(
        combine_dataframes, df1=df_result, df2=lf_result
    )
    combined_output = combined_result.outputs()
    print(f"Combined DataFrame shape: {combined_output.shape}")
    print(combined_output.head())
