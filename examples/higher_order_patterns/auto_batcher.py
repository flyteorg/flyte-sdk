"""
Auto Batcher Pattern

A higher-order function that automatically batches large datasets and processes them
in parallel at both the batch level and item level. This pattern is useful for
processing large arrays efficiently by:
1. Splitting data into manageable batches
2. Processing batches in parallel
3. Processing items within each batch in parallel
4. Collecting and flattening results

Usage:
    @env.task
    async def process_item(item: str) -> dict:
        # Process individual item
        return {"item": item, "result": process(item)}

    # Process 1000 items in batches of 50
    results = await batch_process(
        process_item,
        data=large_dataset,
        batch_size=50
    )
"""

import asyncio
from typing import List, TypeVar, Callable, Any, Optional
import flyte

T = TypeVar('T')
R = TypeVar('R')


def create_batches(data: List[T], batch_size: int) -> List[List[T]]:
    """Split data into batches of specified size."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


async def batch_map_reduce(
    map_fn: Callable[[T], R],
    reduce_fn: Callable[[List[R]], Any],
    data: List[T],
    batch_size: int = 10,
    max_concurrent_batches: Optional[int] = None
) -> Any:
    """
    Map-reduce pattern with batching.

    Args:
        map_fn: Function to apply to each item (map phase)
        reduce_fn: Function to reduce batch results
        data: List of items to process
        batch_size: Number of items per batch
        max_concurrent_batches: Maximum number of batches to process concurrently

    Returns:
        Final reduced result
    """
    if not data:
        return reduce_fn([])

    batches = create_batches(data, batch_size)
    print(f"Map-reduce processing {len(data)} items in {len(batches)} batches")

    async def process_and_reduce_batch(batch: List[T], batch_idx: int) -> Any:
        """Process a batch and apply local reduction."""
        print(f"Map-reduce batch {batch_idx + 1}/{len(batches)}")

        with flyte.group(f"map-reduce-batch-{batch_idx + 1}"):
            # Map phase: process all items in batch
            batch_tasks = [map_fn(item) for item in batch]
            mapped_results = await asyncio.gather(*batch_tasks)

            # Local reduce phase
            batch_result = reduce_fn(mapped_results)

        return batch_result

    # Process batches with optional concurrency limit
    if max_concurrent_batches and max_concurrent_batches < len(batches):
        batch_results = []
        for i in range(0, len(batches), max_concurrent_batches):
            batch_chunk = batches[i:i + max_concurrent_batches]
            chunk_tasks = [
                process_and_reduce_batch(batch, i + j)
                for j, batch in enumerate(batch_chunk)
            ]
            chunk_results = await asyncio.gather(*chunk_tasks)
            batch_results.extend(chunk_results)
    else:
        # Process all batches concurrently
        batch_tasks = [
            process_and_reduce_batch(batch, i)
            for i, batch in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

    # Final reduce phase
    final_result = reduce_fn(batch_results)
    print(f"Completed map-reduce processing")
    return final_result