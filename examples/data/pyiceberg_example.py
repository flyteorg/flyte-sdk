# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyiceberg",
#     "pyarrow",
#     "pandas",
#     "unionai-reuse>=0.1.7",
# ]
# ///
"""
PyIceberg Parallel Batch Aggregation Example using Flyte

This script demonstrates how to:
1. Read data from an Iceberg table using PyIceberg
2. Split the data into partitions for parallel processing
3. Use Flyte tasks with ReusePolicy to maximize CPU utilization
4. Use flyte.io.DataFrame to minimize data copies by passing references
5. Combine results from all partitions

Key patterns demonstrated:
- flyte.TaskEnvironment with ReusePolicy for efficient resource utilization
- flyte.io.DataFrame for passing dataframe references (metadata only, not data copies)
- asyncio.gather for parallel processing of chunks
- Async tasks for concurrent execution
"""

import asyncio
import logging
from typing import List, Literal

import pandas as pd

import flyte
import flyte.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define image using uv script dependencies (avoids duplicating dependency list)
image = flyte.Image.from_uv_script(__file__, name="pyiceberg")

# Define reusable environment for parallel processing tasks
# ReusePolicy keeps workers alive to avoid cold start overhead
processing_env = flyte.TaskEnvironment(
    name="pyiceberg_processing",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    reusable=flyte.ReusePolicy(
        replicas=(2, 8),  # Scale between 2-8 replicas based on load
        idle_ttl=300,  # Keep workers alive for 5 minutes after idle
    ),
    image=image,
    cache=flyte.Cache("auto", "1.0"),
)

# Non-reusable environment for orchestration tasks
orchestrator_env = processing_env.clone_with(
    name="pyiceberg_orchestrator",
    reusable=None,
    depends_on=[processing_env],
)


AggFunction = Literal["sum", "mean", "count", "max", "min"]


@processing_env.task
async def aggregate_partition(
    partition_data: flyte.io.DataFrame,
    partition_id: int,
    group_by_column: str,
    agg_column: str,
    agg_function: AggFunction = "sum",
) -> flyte.io.DataFrame:
    """
    Perform aggregation on a single partition of data.

    This task runs on reusable workers, maximizing CPU utilization across
    multiple partitions processed in parallel.

    Args:
        partition_data: Flyte DataFrame containing the partition's data (reference only)
        partition_id: Identifier for this partition
        group_by_column: Column name to group by
        agg_column: Column name to aggregate
        agg_function: Aggregation function (sum, mean, count, max, min)

    Returns:
        Flyte DataFrame with aggregation results for this partition
    """
    # Convert from flyte.io.DataFrame to pandas - data is fetched only here
    df: pd.DataFrame = await partition_data.open(pd.DataFrame).all()
    logger.info(f"Processing partition {partition_id} with {len(df)} rows")

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

    # Return as flyte.io.DataFrame - only reference is passed to next task
    return flyte.io.DataFrame.from_df(result)


@processing_env.task
async def split_into_chunks(
    table_data: flyte.io.DataFrame,
    num_chunks: int,
) -> List[flyte.io.DataFrame]:
    """
    Split a DataFrame into multiple chunks for parallel processing.

    Args:
        table_data: Flyte DataFrame to split
        num_chunks: Number of chunks to create

    Returns:
        List of Flyte DataFrames (references only, minimizing data copies)
    """
    # Convert to pandas DataFrame for slicing
    df: pd.DataFrame = await table_data.open(pd.DataFrame).all()
    total_rows = len(df)
    chunk_size = max(1, total_rows // num_chunks)

    chunks: List[flyte.io.DataFrame] = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        # Last chunk gets any remaining rows
        end_idx = total_rows if i == num_chunks - 1 else (i + 1) * chunk_size

        if start_idx < total_rows:
            chunk = df.iloc[start_idx:end_idx]
            # Convert each chunk to flyte.io.DataFrame - stores reference
            chunks.append(flyte.io.DataFrame.from_df(chunk))

    logger.info(f"Split {total_rows} rows into {len(chunks)} chunks")
    return chunks


@processing_env.task
async def combine_results(
    partition_results: List[flyte.io.DataFrame],
    group_by_column: str,
    agg_column: str,
    agg_function: AggFunction,
) -> flyte.io.DataFrame:
    """
    Combine results from all partitions into a final aggregated result.

    Args:
        partition_results: List of Flyte DataFrames with partition results
        group_by_column: Column name that was grouped by
        agg_column: Column name that was aggregated
        agg_function: Aggregation function used

    Returns:
        Final combined Flyte DataFrame with aggregated results
    """
    # Fetch and combine all partition results
    dfs = [await result.open(pd.DataFrame).all() for result in partition_results]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Final aggregation across all partitions
    # (since each partition processed independently, we need to re-aggregate)
    if agg_function in ("sum", "count"):
        final_result = (
            combined_df.groupby(group_by_column).agg({agg_column: "sum", "partition_id": "count"}).reset_index()
        )
    elif agg_function == "mean":
        # For mean, we need weighted average (simplified: just re-average)
        final_result = (
            combined_df.groupby(group_by_column).agg({agg_column: "mean", "partition_id": "count"}).reset_index()
        )
    elif agg_function == "max":
        final_result = (
            combined_df.groupby(group_by_column).agg({agg_column: "max", "partition_id": "count"}).reset_index()
        )
    elif agg_function == "min":
        final_result = (
            combined_df.groupby(group_by_column).agg({agg_column: "min", "partition_id": "count"}).reset_index()
        )
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_function}")

    final_result.rename(columns={"partition_id": "num_partitions"}, inplace=True)

    logger.info(f"Combined results: {len(final_result)} groups from {len(partition_results)} partitions")
    return flyte.io.DataFrame.from_df(final_result)


@orchestrator_env.task
async def process_table_parallel(
    table_data: flyte.io.DataFrame,
    group_by_column: str,
    agg_column: str,
    agg_function: AggFunction = "sum",
    num_partitions: int = 4,
) -> flyte.io.DataFrame:
    """
    Main orchestrator that splits data and processes partitions in parallel.

    This task:
    1. Splits the input data into chunks
    2. Launches parallel tasks for each chunk using asyncio.gather
    3. Combines and returns the final aggregated results

    Args:
        table_data: Flyte DataFrame containing all data (reference only)
        group_by_column: Column to group by for aggregation
        agg_column: Column to aggregate
        agg_function: Aggregation function to apply
        num_partitions: Number of parallel partitions to process

    Returns:
        Combined Flyte DataFrame with aggregated results from all partitions
    """
    logger.info(f"Starting parallel processing with {num_partitions} partitions")

    # Split data into chunks
    chunks = await split_into_chunks(table_data, num_partitions)

    # Process all chunks in parallel using asyncio.gather
    # Each chunk is processed by a separate reusable worker
    partition_coros = [
        aggregate_partition(
            partition_data=chunk,
            partition_id=i,
            group_by_column=group_by_column,
            agg_column=agg_column,
            agg_function=agg_function,
        )
        for i, chunk in enumerate(chunks)
    ]

    # Run all partitions in parallel
    partition_results = await asyncio.gather(*partition_coros)

    logger.info(f"All {len(partition_results)} partitions completed")

    # Combine results
    return await combine_results(
        partition_results,
        group_by_column,
        agg_column,
        agg_function,
    )


@orchestrator_env.task
async def create_sample_data() -> flyte.io.DataFrame:
    """
    Create sample data for demonstration purposes.

    In production, this would be replaced by reading from an actual Iceberg table:
        catalog = load_catalog("my_catalog", **catalog_config)
        table = catalog.load_table("my_namespace.my_table")
        table_data = table.scan().to_arrow()

    Returns:
        Flyte DataFrame with sample data (reference only)
    """
    data = {
        "category": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"] * 100,
        "value": list(range(1, 1001)),
        "region": ["US", "EU", "ASIA"] * 333 + ["US"],
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="h"),
    }
    df = pd.DataFrame(data)
    logger.info(f"Created sample data: {len(df)} rows")

    # Return as flyte.io.DataFrame - only metadata/reference is passed
    return flyte.io.DataFrame.from_df(df)


@orchestrator_env.task
async def main() -> flyte.io.DataFrame:
    """
    Main entry point demonstrating the full workflow.

    This orchestrates:
    1. Loading data (simulated with sample data)
    2. Processing it in parallel batches across reusable workers
    3. Returning combined results
    """
    logger.info("=== PyIceberg Parallel Aggregation with Flyte ===")

    # Create sample data (in production: read from Iceberg table)
    table_data = await create_sample_data()

    # Process with parallel aggregation
    result = await process_table_parallel(
        table_data=table_data,
        group_by_column="category",
        agg_column="value",
        agg_function="sum",
        num_partitions=4,
    )

    logger.info("=== Processing Complete ===")
    return result


if __name__ == "__main__":
    # Initialize Flyte connection
    flyte.init_from_config()

    # Run the main workflow remotely
    run = flyte.run(main)
    print(f"Run URL: {run.url}")

    # Wait for completion and print results
    run.wait()
    outputs = run.outputs()
    if outputs:
        result_df = outputs.to_pandas()
        print("\nFinal Results:")
        print(result_df)
