"""
Micro-batching pattern with Flyte Traces for fault tolerance.

This example demonstrates how to process large numbers of items by:
1. Breaking work into batches to avoid hitting action limits
2. Using @flyte.trace to checkpoint progress for recoverability
3. Processing batches in parallel for efficiency

Key benefits:
- Automatic checkpointing at batch boundaries
- Resume from last successful batch on failure
- No re-execution of completed work
"""

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import flyte
from flyte.remote import Run

# Configuration
NUMBER_OF_INPUTS = 10_000
BATCH_SIZE = 1000

# Shared image for all tasks
image = flyte.Image.from_debian_base().with_pip_packages("flyte>=2.0.0b37", "unionai-reuse>=0.1.3")

# Batch processor with reusable containers
batch_env = flyte.TaskEnvironment(
    name="batch_processor",
    resources=flyte.Resources(memory="2Gi", cpu="1"),
    reusable=flyte.ReusePolicy(
        replicas=(3, 10),
        concurrency=5,
        idle_ttl=timedelta(minutes=5),
    ),
    image=image,
)

# Main orchestrator
orchestrator_env = flyte.TaskEnvironment(
    name="orchestrator",
    depends_on=[batch_env],
    resources=flyte.Resources(memory="4Gi", cpu="1"),
    image=image,
)


# Helper functions for external service calls
async def submit_to_service(request_id: int) -> str:
    """
    Submit request to external service.
    
    Replace this with your actual service call:
    async with httpx.AsyncClient() as client:
        response = await client.post("https://your-service.com/api/submit",
                                     json={"request_id": request_id})
        return response.json()["job_id"]
    """
    await asyncio.sleep(0.01)  # Simulate API call
    job_id = f"job_{request_id}"
    return job_id


async def poll_job_status(job_id: str, request_id: int) -> int:
    """
    Poll for job completion.
    
    Replace this with your actual polling logic:
    async with httpx.AsyncClient() as client:
        for _ in range(max_attempts):
            response = await client.get(f"https://your-service.com/api/status/{job_id}")
            status = response.json()
            if status["state"] == "completed":
                return status["result"]
            await asyncio.sleep(1)
    """
    await asyncio.sleep(0.05)  # Simulate polling
    return request_id * 2


@batch_env.task
async def process_batch(batch_start: int, batch_end: int) -> List[int]:
    """
    Process a batch of items with two traced phases for checkpointing.
    
    The @flyte.trace decorators create checkpoints, so if this task fails
    and is retried, it will resume from the last completed phase.
    """

    @flyte.trace
    async def submit_phase(items: List[int]) -> Dict[int, str]:
        """Submit all items and checkpoint the job IDs."""
        job_ids = await asyncio.gather(
            *(submit_to_service(request_id=x) for x in items),
            return_exceptions=True
        )
        
        job_mapping = {}
        for request_id, job_id in zip(items, job_ids):
            if isinstance(job_id, Exception):
                print(f"[ERROR] Submit failed for {request_id}: {job_id}")
                job_mapping[request_id] = None
            else:
                job_mapping[request_id] = job_id
        
        return job_mapping

    @flyte.trace
    async def wait_phase(job_mapping: Dict[int, str]) -> List[int]:
        """Wait for all jobs and checkpoint the results."""
        results = await asyncio.gather(
            *(
                poll_job_status(job_id=job_id, request_id=request_id)
                if job_id is not None
                else asyncio.sleep(0)
                for request_id, job_id in job_mapping.items()
            ),
            return_exceptions=True
        )
        
        processed_results = []
        for request_id, result in zip(job_mapping.keys(), results):
            if isinstance(result, Exception):
                print(f"[ERROR] Wait failed for {request_id}: {result}")
                processed_results.append(-1)
            else:
                processed_results.append(result)
        
        return processed_results

    # Execute both phases - each creates a checkpoint
    items = list(range(batch_start, batch_end))
    job_mapping = await submit_phase(items)
    results = await wait_phase(job_mapping)
    
    print(f"Batch {batch_start}-{batch_end}: {len([r for r in results if r != -1])}/{len(results)} successful")
    return results


@orchestrator_env.task
async def microbatch_workflow(
    total_items: int = NUMBER_OF_INPUTS,
    batch_size: int = BATCH_SIZE,
) -> List[int]:
    """
    Main workflow that processes items in batches.
    
    Each batch is processed in parallel, and progress is checkpointed
    at the batch level, allowing resumption on failure.
    """
    # Calculate batches
    batches = [(start, min(start + batch_size, total_items)) 
               for start in range(0, total_items, batch_size)]

    print(f"Processing {total_items} items in {len(batches)} batches of size {batch_size}")

    # Process all batches in parallel
    batch_results = await asyncio.gather(
        *(process_batch(batch_start=start, batch_end=end) for start, end in batches),
        return_exceptions=True
    )

    # Collect results
    all_results = []
    failed_batches = 0
    
    for i, batch_result in enumerate(batch_results):
        if isinstance(batch_result, Exception):
            print(f"[ERROR] Batch {i} failed: {batch_result}")
            failed_batches += 1
        else:
            all_results.extend(batch_result)

    success_count = len([r for r in all_results if r != -1])
    print(f"\n{'=' * 60}")
    print(f"Workflow completed:")
    print(f"  Total items: {total_items}")
    print(f"  Successful: {success_count}")
    print(f"  Failed batches: {failed_batches}")
    print(f"  Success rate: {success_count / total_items * 100:.1f}%")
    print(f"{'=' * 60}\n")

    return all_results


if __name__ == "__main__":
    flyte.init_from_config()

    print("Starting micro-batching workflow with fault tolerance...")
    print("Progress is checkpointed and can resume from last successful batch.\n")

    r: Run = flyte.run(microbatch_workflow)

    print(f"\nRun: {r.name}")
    print(f"URL: {r.url}")
    print("\nView trace details in the Flyte UI execution graph.")

    r.wait()
    print("\n✅ Workflow completed!")
