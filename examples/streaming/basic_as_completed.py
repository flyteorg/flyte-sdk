import asyncio
import random

import flyte

reusable_image = flyte.Image.from_debian_base(name="new").with_pip_packages("unionai-reuse>=0.1.3")

env = flyte.TaskEnvironment(
    name="single_cpu",
    resources=flyte.Resources(cpu="1"),
    reusable=flyte.ReusePolicy(replicas=20, idle_ttl=30),
    image=reusable_image
)


@env.task
async def process_item(item: str) -> str:
    print(f"Processing {item}", flush=True)
    # Simulate varying processing times
    query_delay = random.uniform(1, 5)
    await asyncio.sleep(query_delay)
    return f"processed_{item}"


@env.task
async def reduce_batch(items: list[str]) -> str:
    print(f"Reducing batch of {len(items)} items")
    return f"reduced_batch_of_{len(items)}_items"


@env.task
async def streaming_reduce_processing() -> str:
    input_items = [f"item_{i}" for i in range(100)]

    # Start all tasks immediately
    tasks = [asyncio.create_task(process_item(item)) for item in input_items]

    batch_size = 10
    accumulated_values = []
    reducers = []

    print(f"Started {len(tasks)} tasks, will reduce in batches of {batch_size}")

    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        result = await task
        accumulated_values.append(result)

        # When we have enough results, start a reduce operation
        if len(accumulated_values) >= batch_size:
            print(f"Triggering reduce for batch of {len(accumulated_values)}")
            reducer_task = asyncio.create_task(reduce_batch(accumulated_values.copy()))
            reducers.append(reducer_task)
            accumulated_values.clear()

    # Handle any remaining stragglers
    if accumulated_values:
        print(f"Handling final batch of {len(accumulated_values)} stragglers")

        reducer_task = asyncio.create_task(reduce_batch(accumulated_values))
        reducers.append(reducer_task)

    # Wait for all reduce operations to complete
    reduced_results = await asyncio.gather(*reducers)

    # Final reduce step to combine all batch results
    final_result = await reduce_batch(reduced_results)

    print(f"Completed {len(reducers)} reduce operations, final result: {final_result}")
    return final_result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(streaming_reduce_processing)
    print(run.url)