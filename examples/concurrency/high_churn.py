"""High-churn fan-out: many SHORT actions under a tight cap.

Stresses the accounting that matters most — the reserve↔snapshot turnover. Slots
free faster than the 1s snapshot settles, so correctness rides on the in-memory
reserve delta (reserved - watermark), not the periodic scan. Longer sleeps only
make the cap easier to hold; short actions are where accounting bugs surface.

Run:   python examples/concurrency/high_churn.py
Watch: the /debug poll loop in README (LIMIT=3) + run_concurrency_tracked_runs gauge.
Pass:  per-run active never exceeds 3; throughput ~= 3 actions/sec (cap, not
       cap/duration); gauge returns to 0.
"""

import asyncio

import flyte

env = flyte.TaskEnvironment(name="conc_high_churn")

N = 20
SLEEP_S = 0.2


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
    run = flyte.with_runcontext(max_action_concurrency=3).run(main, n=N)
    print(run.url)
    run.wait()
