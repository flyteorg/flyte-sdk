"""
Minimal reproduction of the duplicate action ID bug.

Modeled after the entity extraction workflow pattern where:
1. asyncio.gather launches concurrent .override() calls — all override objects
   alive simultaneously with distinct id()s, each getting task_call_seq=1.
2. A polling loop calls .override() repeatedly with identical inputs, producing
   the same (parent, input_hash, task_hash, seq=1) → duplicate action IDs.

At runtime this causes:
  "Task X did not return an output path, but the task has outputs defined."
because a later task's completion event overwrites an earlier one in the
ActionCache (which also has a stale-event bug in remove()).
"""

import asyncio

import flyte
from flyte._image import PythonWheels
from pathlib import Path

controller_dist_folder = Path("/Users/ytong/go/src/github.com/flyteorg/sdk-rust/rs_controller/dist")
wheel_layer = PythonWheels(wheel_dir=controller_dist_folder, package_name="flyte_controller_base")
base = flyte.Image.from_debian_base()
rs_controller_image = base.clone(addl_layer=wheel_layer)

env_worker = flyte.TaskEnvironment(
    name="worker",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    cache="disable",
    image=rs_controller_image,
)

env_main = flyte.TaskEnvironment(
    name="main",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    depends_on=[env_worker],
    cache="disable",
    image=rs_controller_image,
)


@env_worker.task
async def process_batch(short_name: str, batch_index: int) -> str:
    return f"processed: {short_name}-{batch_index}"


@env_worker.task
async def poll_batch(short_name: str, batch_index: int) -> str | int:
    """Simulates a task that may return a retry delay (int) or a final result (str).

    On the first call it returns a polling delay; on subsequent calls the result.
    This mirrors parse_entity_extraction in the real workflow.
    """
    return f"done: {short_name}-{batch_index}"


async def process_with_polling(batch_index: int) -> str:
    """Mirrors process_entity_extraction_as_offline_job.

    Calls process_batch once, then enters a polling loop where poll_batch is
    called with .override() repeatedly using the *same* inputs each time.
    Each .override() creates a new Python object → new id() → seq resets to 1
    → duplicate action IDs when inputs hash the same.
    """
    short_process_name = f"process_batch-{batch_index}"
    result = await process_batch.override(short_name=short_process_name)(
        short_name=short_process_name,
        batch_index=batch_index,
    )

    # Polling loop: same task, same inputs, new .override() object each time
    short_poll_name = f"poll_batch-{batch_index}"
    poll_result = await poll_batch.override(short_name=short_poll_name)(
        short_name=short_poll_name,
        batch_index=batch_index,
    )
    while isinstance(poll_result, int):
        await asyncio.sleep(poll_result)
        # BUG: each .override() call here creates a new object with a new id(),
        # but inputs are identical → same input_hash, same task_hash, seq=1
        # → same action ID as the first call above
        poll_result = await poll_batch.override(short_name=short_poll_name)(
            short_name=short_poll_name,
            batch_index=batch_index,
        )

    return f"{result} | {poll_result}"


@env_main.task
async def main() -> list[str]:
    batches = list(range(5))

    # Pattern 1: asyncio.gather with .override() on the same task
    # All override objects are alive concurrently → all get distinct id()s
    # → all get task_call_seq=1 for the same task name
    results: list[str] = await asyncio.gather(*(process_with_polling(batch_index) for batch_index in batches))

    return results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
