# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyiceberg",
#     "pyarrow",
#     "unionai-reuse>=0.1.7",
#     "flyte>=2.0.0b35",
# ]
# ///
"""
PyIceberg Parallel Batch Aggregation Example using Flyte

This script demonstrates how to:
1. Read data from an Iceberg table using PyIceberg WITHOUT loading the entire table
2. Use scan().plan_files() to get parquet file paths instead of .to_arrow()
3. Pass file paths (strings) between tasks for zero-copy data passing
4. Each worker reads only its assigned parquet files directly
5. Use Flyte tasks with ReusePolicy to maximize CPU utilization
6. Combine results from all partitions

Key optimization: Instead of loading the entire table with table.scan().to_arrow(),
use table.scan().plan_files() to get file paths, distribute them across workers,
and let each worker read only its assigned files. This achieves:
- Zero-copy data passing (only file paths are serialized)
- No full table load into memory
- True parallel file processing

Key patterns demonstrated:
- PyIceberg scan().plan_files() for efficient file-level parallelism
- Zero-copy data passing by passing file paths instead of data
- flyte.TaskEnvironment with ReusePolicy for efficient resource utilization
- asyncio.gather for parallel processing of parquet files
- Async tasks for concurrent execution
"""

import asyncio
import logging
from typing import List, Literal

import pyarrow as pa
import pyarrow.parquet as pq

import flyte
import flyte.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define image using uv script dependencies (avoids duplicating dependency list)
image = flyte.Image.from_uv_script(__file__, name="pyiceberg", pre=True)

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
    cache=flyte.Cache("auto", "4.0"),
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
    file_paths: List[str],
    partition_id: int,
    group_by_column: str,
    agg_column: str,
    agg_function: AggFunction = "sum",
) -> flyte.io.DataFrame:
    """
    Perform aggregation on a single partition of data by reading parquet files.

    This task runs on reusable workers, maximizing CPU utilization across
    multiple partitions processed in parallel.

    Args:
        file_paths: List of parquet file paths to process for this partition
        partition_id: Identifier for this partition
        group_by_column: Column name to group by
        agg_column: Column name to aggregate
        agg_function: Aggregation function (sum, mean, count, max, min)

    Returns:
        Flyte DataFrame with aggregation results for this partition
    """
    # Read parquet files directly - only load files assigned to this partition
    tables = [pq.read_table(f) for f in file_paths]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    logger.info(f"Processing partition {partition_id} with {table.num_rows} rows from {len(file_paths)} files")

    # Perform aggregation using PyArrow compute
    grouped = table.group_by(group_by_column)
    if agg_function == "sum":
        result = grouped.aggregate([(agg_column, "sum")])
    elif agg_function == "mean":
        result = grouped.aggregate([(agg_column, "mean")])
    elif agg_function == "count":
        result = grouped.aggregate([(agg_column, "count")])
    elif agg_function == "max":
        result = grouped.aggregate([(agg_column, "max")])
    elif agg_function == "min":
        result = grouped.aggregate([(agg_column, "min")])
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_function}")

    # Add partition ID for tracking
    partition_ids = pa.array([partition_id] * result.num_rows)
    result = result.append_column("partition_id", partition_ids)

    logger.info(f"Partition {partition_id} completed: {result.num_rows} groups")

    # Return as flyte.io.DataFrame - only reference is passed to next task
    return flyte.io.DataFrame.from_df(result)


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
    # Fetch and combine all partition results using PyArrow
    tables = [await result.open(pa.Table).all() for result in partition_results]
    combined_table = pa.concat_tables(tables)

    # The aggregated column name from PyArrow includes the function suffix
    agg_col_name = f"{agg_column}_{agg_function}"

    # Final aggregation across all partitions
    # (since each partition processed independently, we need to re-aggregate)
    grouped = combined_table.group_by(group_by_column)
    if agg_function in ("sum", "count"):
        final_result = grouped.aggregate([(agg_col_name, "sum"), ("partition_id", "count")])
    elif agg_function == "mean":
        # For mean, we need weighted average (simplified: just re-average)
        final_result = grouped.aggregate([(agg_col_name, "mean"), ("partition_id", "count")])
    elif agg_function == "max":
        final_result = grouped.aggregate([(agg_col_name, "max"), ("partition_id", "count")])
    elif agg_function == "min":
        final_result = grouped.aggregate([(agg_col_name, "min"), ("partition_id", "count")])
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_function}")

    # Rename columns for clarity
    final_result = final_result.rename_columns([group_by_column, agg_column, "num_partitions"])

    logger.info(f"Combined results: {final_result.num_rows} groups from {len(partition_results)} partitions")
    return flyte.io.DataFrame.from_df(final_result)


@orchestrator_env.task
async def process_table_parallel(
    file_paths: List[str],
    group_by_column: str,
    agg_column: str,
    agg_function: AggFunction = "sum",
    num_partitions: int = 4,
) -> flyte.io.DataFrame:
    """
    Main orchestrator that distributes parquet files across partitions for parallel processing.

    Optimized for Iceberg tables: instead of loading the entire table with .to_arrow(),
    use scan().plan_files() to get file paths, then process files in parallel.

    Zero-copy data passing: only file paths (strings) are passed between tasks.
    Each worker reads parquet files directly when needed.

    This task:
    1. Distributes parquet file paths across partitions (zero data copying!)
    2. Launches parallel tasks for each partition using asyncio.gather
    3. Combines and returns the final aggregated results

    Args:
        file_paths: List of parquet file paths from Iceberg table scan().plan_files()
        group_by_column: Column to group by for aggregation
        agg_column: Column to aggregate
        agg_function: Aggregation function to apply
        num_partitions: Number of parallel partitions to process

    Returns:
        Combined Flyte DataFrame with aggregated results from all partitions
    """
    logger.info(f"Starting parallel processing with {num_partitions} partitions for {len(file_paths)} files")

    # Distribute file paths across partitions (round-robin)
    partition_files = [[] for _ in range(num_partitions)]
    for idx, file_path in enumerate(file_paths):
        partition_idx = idx % num_partitions
        partition_files[partition_idx].append(file_path)

    # Launch tasks for each partition - just pass file paths!
    partition_coros = []
    for i, files in enumerate(partition_files):
        if files:  # Only process non-empty partitions
            partition_coros.append(
                aggregate_partition(
                    file_paths=files,
                    partition_id=i,
                    group_by_column=group_by_column,
                    agg_column=agg_column,
                    agg_function=agg_function,
                )
            )

    logger.info(f"Launched {len(partition_coros)} partition tasks")

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
async def load_iceberg_table(catalog_config: dict, table_name: str) -> List[str]:
    """
    Load an Iceberg table and return file paths for parallel processing.

    This uses scan().plan_files() to get parquet file paths WITHOUT loading the entire table.

    Args:
        catalog_config: Catalog configuration (e.g., {"uri": "thrift://...", "type": "hive"})
        table_name: Fully qualified table name (e.g., "database.table_name")

    Returns:
        List of parquet file paths from the Iceberg table
    """
    from pyiceberg.catalog import load_catalog

    # Load catalog and table
    catalog = load_catalog("my_catalog", **catalog_config)
    table = catalog.load_table(table_name)

    # Get file paths using scan - does NOT load data!
    file_paths = []
    scan = table.scan()
    for task in scan.plan_files():
        file_paths.append(task.file.file_path)

    logger.info(f"Found {len(file_paths)} parquet files in Iceberg table {table_name}")
    return file_paths


@orchestrator_env.task
async def create_sample_data(rows: int, num_files: int = 4) -> List[str]:
    """
    Create sample data for demonstration purposes and return file paths.

    In production, use load_iceberg_table() instead.

    Args:
        rows: Total number of rows to generate
        num_files: Number of parquet files to create

    Returns:
        List of parquet file paths
    """
    import datetime
    import os
    import tempfile

    # Create sample data using PyArrow - all columns sized to match rows
    category_pattern = ["A", "B", "C"]
    region_pattern = ["US", "EU", "ASIA"]

    # Generate columns with proper length
    categories = [category_pattern[i % len(category_pattern)] for i in range(rows)]
    values = list(range(1, rows + 1))
    regions = [region_pattern[i % len(region_pattern)] for i in range(rows)]

    # Generate timestamps
    base_time = datetime.datetime(2024, 1, 1)
    timestamps = [base_time + datetime.timedelta(hours=i) for i in range(rows)]

    table = pa.table(
        {
            "category": categories,
            "value": values,
            "region": regions,
            "timestamp": pa.array(timestamps, type=pa.timestamp("us")),
        }
    )
    logger.info(f"Created sample data: {table.num_rows} rows")

    # Write to temporary parquet files (simulating Iceberg table files)
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    rows_per_file = rows // num_files

    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = rows if i == num_files - 1 else (i + 1) * rows_per_file
        file_table = table.slice(start_idx, end_idx - start_idx)

        file_path = os.path.join(temp_dir, f"part_{i}.parquet")
        pq.write_table(file_table, file_path)
        uploaded_file = await flyte.io.File.from_local(file_path)
        file_paths.append(uploaded_file.path)

    logger.info(f"Wrote {len(file_paths)} parquet files to {temp_dir}")
    # upload files
    return file_paths


@orchestrator_env.task
async def main(rows: int = 1000) -> flyte.io.DataFrame:
    """
    Main entry point demonstrating the full workflow.

    This orchestrates:
    1. Loading data file paths (simulated with sample data, or use load_iceberg_table)
    2. Processing files in parallel batches across reusable workers
    3. Returning combined results

    In production, replace create_sample_data with load_iceberg_table:
        file_paths = await load_iceberg_table(
            catalog_config={"uri": "thrift://localhost:9083", "type": "hive"},
            table_name="database.table_name"
        )
    """
    logger.info("=== PyIceberg Parallel Aggregation with Flyte ===")

    # Get file paths (in production: use load_iceberg_table)
    file_paths = await create_sample_data(rows, num_files=8)

    # Process with parallel aggregation - files are distributed across workers
    result = await process_table_parallel(
        file_paths=file_paths,
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
    run = flyte.run(main, 10000)
    print(f"Run URL: {run.url}")

    # Wait for completion and print results
    run.wait()
    outputs = run.outputs()
    if outputs:
        print(f"\nFinal Results: {outputs[0]}")
