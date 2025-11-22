# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyiceberg",
#     "pyarrow",
#     "pandas",
#     "duckdb"
# ]
# ///
"""
PyIceberg Parallel Batch Aggregation Example

This script demonstrates how to:
1. Read data from an Iceberg table using PyIceberg
2. Split the data into partitions
3. Use asyncio to run parallel batch aggregations on each partition
4. Combine results from all partitions

This pattern is designed to eventually be converted to Flyte tasks where:
- The main function reads the table and identifies partitions
- Each partition's aggregation becomes an async Flyte task
- Results are aggregated in parallel
"""

import asyncio
import logging
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.catalog import load_catalog
from pyiceberg.table import Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def aggregate_partition(
    partition_data: pa.Table,
    partition_id: int,
    group_by_column: str,
    agg_column: str,
    agg_function: str = "sum"
) -> pd.DataFrame:
    """
    Perform aggregation on a single partition of data.

    This function will eventually become a Flyte task that runs independently
    for each partition, allowing parallel execution across the cluster.

    Args:
        partition_data: PyArrow table containing the partition's data
        partition_id: Identifier for this partition
        group_by_column: Column name to group by
        agg_column: Column name to aggregate
        agg_function: Aggregation function (sum, mean, count, etc.)

    Returns:
        Pandas DataFrame with aggregation results for this partition
    """
    logger.info(f"Processing partition {partition_id} with {len(partition_data)} rows")

    # Simulate some processing time (in real scenarios, this could be heavy computation)
    await asyncio.sleep(0.1)

    # Convert to pandas for easy aggregation
    df = partition_data.to_pandas()

    # Perform aggregation based on the specified function
    if agg_function == "sum":
        result = df.groupby(group_by_column)[agg_column].sum().reset_index()
    elif agg_function == "mean":
        result = df.groupby(group_by_column)[agg_column].mean().reset_index()
    elif agg_function == "count":
        result = df.groupby(group_by_column)[agg_column].count().reset_index()
    elif agg_function == "max":
        result = df.groupby(group_by_column)[agg_column].max().reset_index()
    elif agg_function == "min":
        result = df.groupby(group_by_column)[agg_column].min().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_function}")

    # Add partition ID for tracking
    result["partition_id"] = partition_id

    logger.info(f"Partition {partition_id} completed: {len(result)} groups")
    return result


def split_into_batches(table: pa.Table, num_batches: int) -> List[pa.Table]:
    """
    Split a PyArrow table into multiple batches for parallel processing.

    In production, you would typically partition by actual Iceberg partitions,
    time ranges, or other logical divisions of your data.

    Args:
        table: PyArrow table to split
        num_batches: Number of batches to create

    Returns:
        List of PyArrow tables (batches)
    """
    total_rows = len(table)
    batch_size = max(1, total_rows // num_batches)

    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        # Last batch gets any remaining rows
        end_idx = total_rows if i == num_batches - 1 else (i + 1) * batch_size

        if start_idx < total_rows:
            batch = table.slice(start_idx, end_idx - start_idx)
            batches.append(batch)

    logger.info(f"Split {total_rows} rows into {len(batches)} batches")
    return batches


async def process_table_parallel(
    table_data: pa.Table,
    group_by_column: str,
    agg_column: str,
    agg_function: str = "sum",
    num_partitions: int = 4
) -> pd.DataFrame:
    """
    Main orchestrator that reads table data and processes partitions in parallel.

    This function demonstrates the pattern that will be used in Flyte where:
    1. Read the Iceberg table
    2. Identify partitions to process
    3. Launch parallel async tasks (will become Flyte tasks) for each partition
    4. Collect and combine results

    Args:
        table_data: PyArrow table containing all data
        group_by_column: Column to group by for aggregation
        agg_column: Column to aggregate
        agg_function: Aggregation function to apply
        num_partitions: Number of parallel partitions to process

    Returns:
        Combined pandas DataFrame with aggregated results from all partitions
    """
    logger.info(f"Starting parallel processing with {num_partitions} partitions")

    # Split data into batches (partitions)
    batches = split_into_batches(table_data, num_partitions)

    # Create async tasks for each partition
    # In Flyte, each of these would be a separate task execution
    tasks = [
        aggregate_partition(
            partition_data=batch,
            partition_id=i,
            group_by_column=group_by_column,
            agg_column=agg_column,
            agg_function=agg_function
        )
        for i, batch in enumerate(batches)
    ]

    # Run all tasks in parallel and wait for results
    logger.info(f"Launching {len(tasks)} parallel aggregation tasks")
    partition_results = await asyncio.gather(*tasks)

    # Combine results from all partitions
    combined_df = pd.concat(partition_results, ignore_index=True)

    # Final aggregation across all partitions
    # (since each partition processed independently, we need to combine)
    final_result = combined_df.groupby(group_by_column).agg({
        agg_column: agg_function,
        "partition_id": "count"  # Track how many partitions contributed
    }).reset_index()
    final_result.rename(columns={"partition_id": "num_partitions"}, inplace=True)

    logger.info(f"Processing complete. Final result has {len(final_result)} groups")
    return final_result


def create_sample_data() -> pa.Table:
    """
    Create sample data for demonstration purposes.

    In production, this would be replaced by reading from an actual Iceberg table.

    Returns:
        PyArrow table with sample data
    """
    data = {
        "category": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"] * 100,
        "value": list(range(1, 1001)),
        "region": ["US", "EU", "ASIA"] * 333 + ["US"],
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="h")
    }
    df = pd.DataFrame(data)
    return pa.Table.from_pandas(df)


async def main():
    """
    Main entry point demonstrating the full workflow.

    This shows how to:
    1. Load data from an Iceberg table (simulated with sample data)
    2. Process it in parallel batches
    3. Combine and display results
    """
    logger.info("=== PyIceberg Parallel Aggregation Demo ===")

    # In production, you would load from a real Iceberg table like this:
    # catalog = load_catalog("my_catalog", **catalog_config)
    # table = catalog.load_table("my_namespace.my_table")
    # table_data = table.scan().to_arrow()

    # For this demo, we'll use sample data
    logger.info("Creating sample data...")
    table_data = create_sample_data()
    logger.info(f"Sample data created: {len(table_data)} rows, {len(table_data.column_names)} columns")
    logger.info(f"Columns: {table_data.column_names}")

    # Example 1: Sum aggregation
    logger.info("\n--- Example 1: Sum by category ---")
    result_sum = await process_table_parallel(
        table_data=table_data,
        group_by_column="category",
        agg_column="value",
        agg_function="sum",
        num_partitions=4
    )
    print("\nSum results:")
    print(result_sum)

    # Example 2: Mean aggregation
    logger.info("\n--- Example 2: Mean by region ---")
    result_mean = await process_table_parallel(
        table_data=table_data,
        group_by_column="region",
        agg_column="value",
        agg_function="mean",
        num_partitions=3
    )
    print("\nMean results:")
    print(result_mean)

    # Example 3: Count aggregation
    logger.info("\n--- Example 3: Count by category ---")
    result_count = await process_table_parallel(
        table_data=table_data,
        group_by_column="category",
        agg_column="value",
        agg_function="count",
        num_partitions=4
    )
    print("\nCount results:")
    print(result_count)

    logger.info("\n=== Demo Complete ===")
    logger.info("Next steps: Convert aggregate_partition() to a Flyte task")
    logger.info("and process_table_parallel() to a Flyte workflow that launches")
    logger.info("parallel tasks for each partition.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())