"""Negative test: max_action_concurrency=1 must be rejected at submission.

A cap of 1 is a guaranteed deadlock (the root action holds the run's only slot
while awaiting its children), so CreateRun rejects it with InvalidArgument before
the run is ever admitted.

Run:   python examples/concurrency/reject_limit_one.py
Pass:  submission raises (InvalidArgument); the run never starts.
"""

import asyncio

import flyte

env = flyte.TaskEnvironment(name="conc_reject_one")

# --- knobs ---
N = 4
SLEEP_S = 2


@env.task
async def step(i: int) -> int:
    await asyncio.sleep(SLEEP_S)
    return i


@env.task
async def main(n: int = N) -> list[int]:
    tasks = [step(i) for i in range(n)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    try:
        run = flyte.with_runcontext(max_action_concurrency=1).run(main, n=N)
        print("UNEXPECTED: submission accepted:", run.url)
        print("FAIL — max_action_concurrency=1 should be rejected (InvalidArgument)")
    except Exception as e:
        print("OK — submission rejected as expected:")
        print(f"  {type(e).__name__}: {e}")
