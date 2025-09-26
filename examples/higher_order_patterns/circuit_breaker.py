"""
Circuit Breaker Pattern

A simple higher-order function that executes tasks in parallel and fails fast
when too many failures occur.

Usage:
    @env.task
    async def process_item(item: str) -> str:
        # Task that might fail
        return f"processed_{item}"

    results = await circuit_breaker_execute(
        process_item,
        ["item1", "item2", "item3"],
        max_failures=2
    )
"""

import asyncio
from typing import List, TypeVar, Callable, Optional

T = TypeVar('T')
R = TypeVar('R')


class CircuitBreakerError(Exception):
    """Raised when too many failures occur."""
    pass


async def circuit_breaker_execute(
    task_fn: Callable[[T], R],
    items: List[T],
    max_failures: int = 3
) -> List[Optional[R]]:
    """
    Execute tasks in parallel with circuit breaker protection.

    Args:
        task_fn: The task function to execute
        items: List of items to process
        max_failures: Maximum number of failures before circuit opens

    Returns:
        List of results, with None for failed items

    Raises:
        CircuitBreakerError: If failures exceed max_failures
    """
    if not items:
        return []

    print(f"Circuit breaker processing {len(items)} items (max failures: {max_failures})")

    # Start all tasks
    tasks = [asyncio.create_task(task_fn(item)) for item in items]
    results = [None] * len(items)
    failures = 0
    pending = set(tasks)

    # Process results as they complete
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            # Find the index of this task
            task_index = tasks.index(task)

            if task.exception():
                failures += 1
                print(f"✗ Task {task_index + 1} failed: {task.exception()}")

                # Check if we've exceeded failure threshold
                if failures > max_failures:
                    print(f"🔴 Circuit breaker opened! {failures} failures exceed limit of {max_failures}")
                    # Cancel remaining tasks
                    for remaining_task in pending:
                        remaining_task.cancel()

                    raise CircuitBreakerError(f"Circuit breaker opened: {failures} failures exceed limit of {max_failures}")
            else:
                results[task_index] = task.result()
                print(f"✓ Task {task_index + 1} succeeded")

    print(f"✅ Completed with {failures} failures (within limit of {max_failures})")
    return results