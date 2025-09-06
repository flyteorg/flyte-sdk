import asyncio
import os
import typing

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="controller_stressor")


@env.task
async def stressor_task(x: int, y: int, z: int = 10, a: int = 20, b: int = 30) -> int:
    """
    A task that simulates a stressor by performing a computation.
    """
    # Simulate some computation
    result = (x + y + z + a + b) * 2
    return result


def get_attempt_number() -> int:
    """
    Get the current attempt number.
    This is a placeholder function to simulate getting the attempt number.
    In a real scenario, this would be replaced with actual logic to retrieve the attempt number.
    """
    return int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))


@env.task(retries=10)
async def main() -> typing.List[int]:
    """
    Main task that runs the stressor_task with different parameters.
    """
    with flyte.group("parallel-group"):
        results = []
        for i in range(5):
            print("Running stressor_task with parameters:", i, i + 1, i + 2, i + 3, i + 4, flush=True)
            result = asyncio.create_task(stressor_task(x=i, y=i + 1, z=i + 2, a=i + 3, b=i + 4))
            results.append(result)
            await asyncio.sleep(5)
            if get_attempt_number() == 0 or (get_attempt_number() == 1 and i == 4):
                raise flyte.errors.RuntimeSystemError(
                    "simulated", f"Simulated failure on attempt {get_attempt_number()}"
                )
        # Use attempt number to fail or continue
        print("Waiting for parallel tasks to complete", flush=True)
        _ = await asyncio.gather(*results)

    if get_attempt_number() == 2:
        raise flyte.errors.RuntimeSystemError("simulated", f"Simulated failure on attempt {get_attempt_number()}")
    # Collect results
    # Run more in sequential and fail after n
    with flyte.group("sequential-group"):
        vals = []
        for i in range(100):
            print("Running stressor_task sequentially with parameters:", i, i + 1, i + 2, i + 3, i + 4, flush=True)
            v = await stressor_task(x=i, y=i + 1, z=i + 2, a=i + 3, b=i + 4)
            vals.append(v)
            if i == 30 and  get_attempt_number() == 3:
                raise flyte.errors.RuntimeSystemError(
                    "simulated", f"Simulated failure on attempt {get_attempt_number()} at iteration {i}"
                )
            if i == 90 and get_attempt_number() == 4:
                raise flyte.errors.RuntimeSystemError("simulated",
                    f"Simulated failure on attempt {get_attempt_number()} at iteration {i}")

        print("All Done with sequential tasks", flush=True)
        return vals


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    r = flyte.run(main)
    print(r.url)
