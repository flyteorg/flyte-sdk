import os

import flyte
import flyte.errors

env = flyte.TaskEnvironment(
    name="crash_recovery_trace",
    resources=flyte.Resources(memory="250Mi", cpu=1),
)


def get_attempt_number() -> int:
    """
    Get the current attempt number.
    This is a placeholder function to simulate getting the attempt number.
    In a real scenario, this would be replaced with actual logic to retrieve the attempt number.
    """
    return int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))


@flyte.trace
async def run(x: int) -> int:
    """
    A simple task that returns the input value.
    """
    return x


@env.task(retries=10)
async def main():
    print("Running crash_recovery_trace task", flush=True)
    with flyte.group("sequential-trace-group"):
        vals = []
        for i in range(11):
            print(f"Running crasher {i}", flush=True)
            v = await run(x=i)
            vals.append(v)
            if i == get_attempt_number() and i < 9:
                raise flyte.errors.RuntimeSystemError(
                    "simulated", f"Simulated failure on attempt {get_attempt_number()} at iteration {i}"
                )

        print("All Done with sequential tasks", flush=True)
        return vals
