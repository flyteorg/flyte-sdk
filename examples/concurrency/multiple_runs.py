"""Several concurrency-limited runs in flight at once, each with its own cap.

Stresses independent per-run budgets and the copy-on-write limited-set churn
(markLimited/unmarkLimited + LimitedRunKeys) when more than one run is tracked.

Run:   python examples/concurrency/multiple_runs.py
Watch: /debug poll loop — each run key's active must stay <= ITS OWN cap (2, 3, 5);
       run_concurrency_tracked_runs == 3 while running, then 0.
Pass:  no run exceeds its own limit; gauge hits 3 then drains to 0.

NOTE: worker capacity must exceed the SUM of caps (2+3+5=10) plus the 3 roots,
or worker saturation will mask the per-run caps.
"""

import asyncio

import flyte

env = flyte.TaskEnvironment(name="conc_multiple_runs")

# --- knobs ---
N = 12
CAPS = (2, 3, 5)
SLEEP_S = 30


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
    runs = []
    for cap in CAPS:
        r = flyte.with_runcontext(max_action_concurrency=cap).run(main, n=N)
        print(f"cap={cap}: {r.url}")
        runs.append(r)
    for r in runs:
        r.wait()
