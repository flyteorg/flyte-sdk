"""
Micro-batching pattern for Flyte V2 that balances parallelism with action limits.

Key design decisions:
1. Use micro-batches (e.g., 100-1000 items) instead of individual items
2. Each micro-batch runs in a reusable environment
3. Within each batch, process items concurrently using asyncio
4. This gives us ~1000-10000 tasks instead of 1M tasks

Tracing strategies for fault tolerance:
- @flyte.trace creates checkpoints that enable workflow resumption
- Per-item tracing: Maximum fault tolerance, more checkpoint overhead
- Per-phase tracing: Balanced approach, fewer checkpoints
- Choose based on: operation cost, failure rate, and idempotency

"""

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import List

import flyte
from flyte.remote import Run

# Configuration
NUMBER_OF_INPUTS = 1_000_000
BATCH_SIZE = 1000  # Adjust this to stay under 50k actions limit
# With 1M inputs and batch_size=1000, we get 1000 tasks (well under 50k)
# With batch_size=100, we'd get 10k tasks (still safe)


reusable_image = flyte.Image.from_debian_base(
).with_pip_packages("unionai-reuse>=0.1.9", "flyte>=2.0.0b37")


worker_reuse_policy = flyte.ReusePolicy(
    replicas=(5, 20),  # Auto-scale between 5-20 replicas
    concurrency=10,     # Each replica handles 10 concurrent batches
    scaledown_ttl=timedelta(minutes=5),
    idle_ttl=timedelta(minutes=10),
)

batch_processor_env = flyte.TaskEnvironment(
    name="batch_processor_env",
    resources=flyte.Resources(memory="2Gi", cpu="1"),
    reusable=worker_reuse_policy,
    image=reusable_image,
)

# Orchestrator environment (driver task)
orchestrator_env = flyte.TaskEnvironment(
    name="orchestrator_env",
    depends_on=[batch_processor_env],
    resources=flyte.Resources(memory="4Gi", cpu="1"),
    image=reusable_image,
)


@batch_processor_env.task
async def process_batch(batch_start: int, batch_end: int) -> List[int]:
    """
    Process a batch of items concurrently within a single task.
    
    This task:
    1. Runs in a reusable environment (reduces startup overhead)
    2. Processes items concurrently within the batch
    3. Handles individual item failures gracefully
    """
    
    @flyte.trace
    async def submit_operation(x: int) -> int:
        """
        Traced submit operation - creates a checkpoint when submission succeeds.
        
        If workflow fails after this, re-running will skip already-submitted items.
        This is especially valuable for non-idempotent operations.
        """
        # Simulate submit operation
        await asyncio.sleep(0.01)  # Replace with actual submit logic
        submit_result = x
        return submit_result
    
    @flyte.trace
    async def wait_operation(submit_result: int) -> int:
        """
        Traced wait operation - creates a checkpoint when polling completes.
        
        If workflow fails during long polling, re-running will skip completed polls.
        """
        # Simulate wait/polling operation
        await asyncio.sleep(0.01)  # Replace with actual wait logic
        wait_result = submit_result * 2
        return wait_result
    
    async def submit_and_wait(x: int) -> int:
        """Process a single item with traced checkpoints."""
        try:
            submit_result = await submit_operation(x)
            wait_result = await wait_operation(submit_result)
            return wait_result
        except Exception as e:
            print(f"Error processing item {x}: {e}")
            return -1  # Or handle errors as needed
    
    # Process all items in this batch concurrently
    items = range(batch_start, batch_end)
    results = await asyncio.gather(
        *(submit_and_wait(x) for x in items),
        return_exceptions=True
    )
    
    # Convert exceptions to error values if needed
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Exception for item {batch_start + i}: {result}")
            processed_results.append(-1)
        else:
            processed_results.append(result)
    
    return processed_results


@orchestrator_env.task
async def microbatch_workflow(
    total_items: int = NUMBER_OF_INPUTS,
    batch_size: int = BATCH_SIZE
) -> List[int]:
    """
    Orchestrate the micro-batching workflow.
    
    Creates batches and launches them in parallel, with each batch
    running in a reusable environment.
    """
    
    # Calculate batch ranges
    batches = []
    for start in range(0, total_items, batch_size):
        end = min(start + batch_size, total_items)
        batches.append((start, end))
    
    print(f"Processing {total_items} items in {len(batches)} batches of size ~{batch_size}")
    print(f"Estimated task count: {len(batches)} (well under 50k limit)")
    
    # Process all batches in parallel
    # Each batch is a separate task that runs in a reusable environment
    batch_results = await asyncio.gather(
        *(process_batch(batch_start=start, batch_end=end) for start, end in batches),
        return_exceptions=True
    )
    
    # Flatten results
    all_results = []
    for batch_result in batch_results:
        if isinstance(batch_result, Exception):
            print(f"Batch failed: {batch_result}")
            continue
        all_results.extend(batch_result)
    
    print(f"Processed {len(all_results)} items successfully")
    return all_results


@orchestrator_env.task
async def main_workflow() -> None:
    """Main entry point for the workflow."""
    results = await microbatch_workflow()
    print(f"Workflow completed. Total results: {len(results)}")



# Alternative: Two-level batching for extreme scale
# Use this if you need even more control or have >1M items

@orchestrator_env.task
async def two_level_microbatch_workflow(
    total_items: int = NUMBER_OF_INPUTS,
    l1_batch_size: int = 10000,  # Level 1: coarse batches
    l2_batch_size: int = 1000,   # Level 2: fine batches
) -> List[int]:
    """
    Two-level batching: Split into coarse batches, then each coarse batch
    splits into fine batches. This keeps task count low while maintaining parallelism.
    
    Example: 1M items -> 100 L1 batches -> each spawns 10 L2 batches = 1000 tasks total
    """
    
    async def process_coarse_batch(coarse_start: int, coarse_end: int) -> List[int]:
        """Process a coarse batch by splitting into fine batches."""
        fine_batches = []
        for start in range(coarse_start, coarse_end, l2_batch_size):
            end = min(start + l2_batch_size, coarse_end)
            fine_batches.append((start, end))
        
        # Process fine batches in parallel
        results = await asyncio.gather(
            *(process_batch(batch_start=start, batch_end=end) for start, end in fine_batches)
        )
        
        # Flatten results
        return [item for batch in results for item in batch]
    
    # Create coarse batches
    coarse_batches = []
    for start in range(0, total_items, l1_batch_size):
        end = min(start + l1_batch_size, total_items)
        coarse_batches.append((start, end))
    
    print(f"Two-level batching: {len(coarse_batches)} coarse batches")
    print(f"Each coarse batch will spawn ~{l1_batch_size // l2_batch_size} fine batches")
    
    # Process coarse batches
    coarse_results = await asyncio.gather(
        *(process_coarse_batch(start, end) for start, end in coarse_batches)
    )
    
    # Flatten results
    all_results = [item for batch in coarse_results for item in batch]
    print(f"Two-level workflow completed. Total results: {len(all_results)}")
    return all_results


if __name__ == "__main__":
    flyte_config = Path(__file__).resolve().parent.parent / ".flyte" / "config.yaml"
    flyte.init_from_config(str(flyte_config))
    
    # Run the single-level micro-batching workflow with per-item tracing
    print("Starting micro-batching workflow with per-item tracing...")
    print("This provides maximum fault tolerance with checkpoints per item.\n")
    r: Run = flyte.with_runcontext(log_format="json").run(main_workflow)
    
    # Alternative options:
    
  
    # Option 2: Two-level batching for extreme scale (>10M items)
    # print("Starting two-level micro-batching workflow...")
    # r: Run = flyte.with_runcontext(log_format="json").run(two_level_microbatch_workflow)
    
    print(f"Run name: {r.name}")
    print(f"Run URL: {r.url}")
    r.wait()

