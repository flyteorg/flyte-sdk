"""
Example demonstrating a potential race condition when calling a task
repeatedly in a while loop with the same short_name override.

This pattern can trigger a RuntimeSystemError:
"Task {name} did not return an output path, but the task has outputs defined."

The issue occurs when:
1. A task returns a Union[int, str] where int means "poll again"
2. The task is called in a while loop with .override(short_name=same_name)
3. The controller may confuse outputs from different iterations
"""

import asyncio
import random
from typing import Union

import flyte
import flyte.errors

env = flyte.TaskEnvironment(
    "action_while_loop_env",
    resources=flyte.Resources(cpu=1, memory="200Mi"),
    cache="disable",
)


@env.task
async def poll_job_status(job_id: str, attempt: int) -> Union[int, str]:
    """
    Simulates polling a job status.

    Returns:
        int: Number of seconds to wait before polling again (job still running)
        str: Final result when job is complete
    """
    print(f"Polling job {job_id}, attempt {attempt}")

    # Simulate job completion after a few attempts
    if attempt >= 50:
        return f"job_{job_id}_completed"

    # Return wait time (in seconds) to indicate job is still running
    return random.randint(1, 3)


async def process_job_with_polling(job_id: str) -> str | None:
    """
    Process a job by polling its status in a while loop.

    This is the pattern that can trigger the race condition:
    - Same short_name is used for each call in the while loop
    - The controller may see stale output paths from previous iterations
    """
    short_name = f"poll_status_{job_id}"
    attempt = 0

    # Initial poll
    result = await poll_job_status.override(short_name=short_name)(
        job_id=job_id, attempt=attempt
    )

    # Keep polling while result is an int (wait time)
    try:
        while isinstance(result, int):
            wait_seconds = result
            print(f"Job {job_id} still running, waiting {wait_seconds}s before retry")
            await asyncio.sleep(wait_seconds)

            attempt += 1
            # POTENTIAL RACE CONDITION: Using same short_name for different calls
            result = await poll_job_status.override(short_name=short_name)(
                job_id=job_id, attempt=attempt
            )
    except flyte.errors.RuntimeSystemError as e:
        print(f"Job {job_id} failed with error: {e}")
        raise

    print(f"Job {job_id} completed")
    return result


@env.task
async def main(num_jobs: int = 3) -> list[str]:
    """
    Run multiple jobs concurrently, each with its own polling loop.
    """
    with flyte.group("process_jobs"):
        results = await asyncio.gather(
            *[process_job_with_polling(f"job_{i}") for i in range(num_jobs)]
        )
    return list(results)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, num_jobs=3)
    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
