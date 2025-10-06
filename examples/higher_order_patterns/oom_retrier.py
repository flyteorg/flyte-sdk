"""
OOM Retrier Pattern

A higher-order function that automatically retries a Flyte task with increasing memory
allocation when it encounters Out of Memory (OOM) errors. This pattern is useful for
tasks that may have variable memory requirements or when you're unsure of the optimal
memory allocation.

Usage:
    @env.task
    async def memory_intensive_task(data: List[str]) -> List[str]:
        # Process large data that might cause OOM
        result = process_large_dataset(data)
        return result

    # Use the retrier
    result = await retry_with_memory(
        memory_intensive_task,
        data,
        initial_memory="500Mi",
        increment="200Mi",
        max_memory="2Gi"
    )
"""

import flyte
import flyte.errors


def parse_memory(mem_str: str) -> int:
    """Convert a memory string like '250Mi' or '1Gi' to Mi as int."""
    if mem_str.endswith("Gi"):
        return int(float(mem_str[:-2]) * 1024)
    if mem_str.endswith("Mi"):
        return int(mem_str[:-2])
    raise ValueError(f"Unsupported memory format: {mem_str}")


def format_memory(mem_mi: int) -> str:
    """Convert Mi int to a memory string like '250Mi'."""
    return f"{mem_mi}Mi"


async def retry_with_memory(
    task_fn,
    *args,
    initial_memory: str = "250Mi",
    increment: str = "200Mi",
    max_memory: str = "4Gi",
    cpu: int = 1,
    **kwargs,
):
    """
    Retry a Flyte task with increasing memory allocation on OOM errors.

    Args:
        task_fn: The Flyte task function to execute
        *args: Positional arguments to pass to the task
        initial_memory: Starting memory allocation (e.g., "250Mi", "1Gi")
        increment: Memory increment on each retry (e.g., "200Mi", "512Mi")
        max_memory: Maximum memory to attempt (e.g., "4Gi", "8Gi")
        cpu: CPU allocation (default: 1)
        **kwargs: Keyword arguments to pass to the task

    Returns:
        The result of the successful task execution

    Raises:
        RuntimeError: If task fails with OOM even at maximum memory
        Other exceptions: Re-raises any non-OOM exceptions from the task
    """
    current_memory_mi = parse_memory(initial_memory)
    increment_mi = parse_memory(increment)
    max_memory_mi = parse_memory(max_memory)

    attempt = 1

    while current_memory_mi <= max_memory_mi:
        mem_str = format_memory(current_memory_mi)
        print(f"Attempt {attempt}: Running task with memory: {mem_str}")

        try:
            result = await task_fn.override(resources=flyte.Resources(cpu=cpu, memory=mem_str))(*args, **kwargs)
            print(f"Success with memory: {mem_str}")
            return result

        except flyte.errors.OOMError as e:
            print(f"OOMError with memory {mem_str}: {e}")
            if current_memory_mi + increment_mi > max_memory_mi:
                break
            current_memory_mi += increment_mi
            attempt += 1

    raise RuntimeError(
        f"Task failed with OOM even after retrying up to {format_memory(max_memory_mi)} across {attempt} attempts"
    )
