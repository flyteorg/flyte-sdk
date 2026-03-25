"""
Demonstrates NonRecoverableError — a way to signal that a task failure is terminal
and should not be retried, even if retries remain.

Expected behavior:
    x=-5  -> NonRecoverableError raised, task fails on attempt #1 with no retries consumed
    x=42  -> task succeeds normally on the first attempt
"""

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="non_recoverable", resources=flyte.Resources(cpu=1, memory="250Mi"))


@env.task(retries=3)
async def validate_and_process(x: int) -> str:
    """
    Task with 3 retries. Raises NonRecoverableError for negative input — the task fails
    immediately without consuming any retry attempts. Check the run in the UI: you will
    see only 1 attempt even though retries=3.
    """
    if x < 0:
        raise flyte.errors.NonRecoverableError(
            f"Input x={x} is negative. Negative inputs are always invalid — retrying would not help."
        )
    return f"processed({x})"


if __name__ == "__main__":
    flyte.init_from_config()
    # Run validate_and_process with invalid input — observe in the UI that only
    # 1 attempt is made despite retries=3.
    run = flyte.run(validate_and_process, x=-5)
    print(run.url)
