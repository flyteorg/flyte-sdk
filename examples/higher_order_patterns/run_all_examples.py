"""
Script to run all higher-order function pattern examples in the higher_order_patterns directory.

This script demonstrates:
1. Auto Batcher Pattern - batch processing with parallel execution
2. Fallback Runner Pattern - primary/fallback task execution
3. OOM Retrier Pattern - automatic memory scaling on OOM errors

Each example is run independently and shows the pattern in action.
"""

import asyncio
import random
import time

import flyte
import flyte.errors

# Set up the environment with a reusable image
reusable_image = flyte.Image.from_debian_base(name="higher_order_patterns").with_pip_packages("unionai-reuse>=0.1.3")

env = flyte.TaskEnvironment(
    name="higher_order_cpu", resources=flyte.Resources(cpu="1", memory="512Mi"), image=reusable_image
)

# Import our higher-order patterns
from auto_batcher import batch_map_reduce
from circuit_breaker import circuit_breaker_execute
from fallback_runner import run_with_fallback
from oom_retrier import retry_with_memory

# === AUTO BATCHER EXAMPLE ===


@env.task
async def process_text_item(item: str) -> dict:
    """Process a single text item - simulates some computation."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return {"item": item, "length": len(item), "processed_at": time.time(), "uppercase": item.upper()}


def sum_all(items) -> int:
    """Simple reduce function that sums numbers."""
    return sum(items)


@env.task
async def get_length(text: str) -> int:
    """Simple task that returns text length."""
    await asyncio.sleep(0.1)
    return len(text)


@env.task
async def auto_batcher_example() -> dict:
    """Demonstrate the auto batcher pattern."""
    print("=== AUTO BATCHER EXAMPLE ===")

    # Create simple data
    data = ["hello", "world", "test", "batch", "processing"]

    # Example: Batch map-reduce - map to lengths, reduce by summing
    print("Running batch_map_reduce example...")
    total_length = await batch_map_reduce(map_fn=get_length, reduce_fn=sum_all, data=data, batch_size=2)

    return {"total_characters": total_length, "words_processed": len(data), "pattern": "auto_batcher"}


# === FALLBACK RUNNER EXAMPLE ===


class APIError(Exception):
    """Simulated API error."""


class CustomTimeoutError(Exception):
    """Simulated timeout error."""


@env.task
async def unreliable_api_task(data: str) -> dict:
    """Primary task that randomly fails to simulate API issues."""
    await asyncio.sleep(0.1)

    # Randomly fail to demonstrate fallback
    if random.random() < 0.7:  # 70% failure rate
        if random.random() < 0.5:
            raise APIError(f"API failed for {data}")
        else:
            raise CustomTimeoutError(f"API timeout for {data}")

    return {"result": f"api_processed_{data}", "source": "primary_api"}


@env.task
async def local_fallback_task(data: str) -> dict:
    """Fallback task that processes locally."""
    await asyncio.sleep(0.05)  # Faster local processing
    return {"result": f"local_processed_{data}", "source": "local_fallback"}


@env.task
async def fallback_runner_example() -> dict:
    """Demonstrate the fallback runner pattern."""
    print("=== FALLBACK RUNNER EXAMPLE ===")

    test_data = ["item_1", "item_2", "item_3", "item_4", "item_5"]
    results = []

    for item in test_data:
        result = await run_with_fallback(
            primary_task=unreliable_api_task,
            fallback_task=local_fallback_task,
            data=item,
            fallback_exceptions=[APIError, CustomTimeoutError],
            log_failures=True,
        )
        results.append(result)

    # Count how many used fallback
    fallback_count = sum(1 for r in results if r["source"] == "local_fallback")
    primary_count = len(results) - fallback_count

    return {
        "total_processed": len(results),
        "primary_successes": primary_count,
        "fallback_uses": fallback_count,
        "pattern": "fallback_runner",
    }


# === OOM RETRIER EXAMPLE ===


@env.task
async def memory_hungry_task(message: str) -> dict:
    """Task that allocates memory and may OOM."""
    print(f"Processing: {message}")

    # Simple memory allocation - fixed size that will likely OOM on small memory
    large_list = [0] * 100000000  # ~800MB
    print(f"Allocated list with {len(large_list)} items")

    return {"message": message, "success": True}


@env.task
async def oom_retrier_example() -> dict:
    """Demonstrate the OOM retrier pattern."""
    print("=== OOM RETRIER EXAMPLE ===")

    try:
        result = await retry_with_memory(
            memory_hungry_task, "large dataset processing", initial_memory="250Mi", increment="300Mi", max_memory="1Gi"
        )

        return {"result": result, "pattern": "oom_retrier"}
    except Exception as e:
        print(f"Failed even with max memory: {e}")
        return {"error": str(e), "pattern": "oom_retrier"}


# === CIRCUIT BREAKER EXAMPLE ===


@env.task
async def flaky_service_task(item: str) -> str:
    """Task that randomly fails to simulate unreliable service."""
    await asyncio.sleep(0.1)

    # 40% failure rate
    if random.random() < 0.4:
        raise Exception(f"Service failed for {item}")

    return f"processed_{item}"


@env.task
async def circuit_breaker_example() -> dict:
    """Demonstrate the circuit breaker pattern."""
    print("=== CIRCUIT BREAKER EXAMPLE ===")

    # Test data
    items = [f"item_{i}" for i in range(10)]

    results = await circuit_breaker_execute(flaky_service_task, items, max_failures=3)
    print(f"Results: {results}")

    successful_count = sum(1 for r in results if r is not None)

    return {
        "items_processed": len(items),
        "successful": successful_count,
        "failed": len(items) - successful_count,
        "pattern": "circuit_breaker",
    }


# === MAIN ORCHESTRATOR ===


@env.task
async def run_all_higher_order_patterns() -> dict:
    """Main task that runs all higher-order pattern examples."""
    print("🚀 Running all higher-order function pattern examples...")

    # Run all examples
    # auto_batcher_result = await auto_batcher_example()
    # fallback_result = await fallback_runner_example()
    # oom_result = await oom_retrier_example()
    circuit_breaker_result = await circuit_breaker_example()

    summary = {
        "completed_patterns": 4,
        # "auto_batcher": auto_batcher_result,
        # "fallback_runner": fallback_result,
        # "oom_retrier": oom_result,
        "circuit_breaker": circuit_breaker_result,
        "status": "completed",
    }

    print("✅ All higher-order pattern examples completed!")
    print(f"Summary: {summary}")

    return summary


if __name__ == "__main__":
    # Initialize Flyte with config
    flyte.init_from_config("../config.yaml")

    print("Starting higher-order patterns demonstration...")
    run = flyte.run(run_all_higher_order_patterns)
    print(f"Execution URL: {run.url}")
