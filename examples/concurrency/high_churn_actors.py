"""High-churn fan-out under a tight cap, on REUSABLE containers (actors).

Same intent as high_churn.py — a wide fan-out of many SHORT actions under a
per-run cap — but the child task runs in a reusable environment, so actions land
on warm, already-running containers instead of one fresh pod each. That removes
pod spin-up from the picture and isolates the thing under test: the leasor's
run-level cap accounting (the reserve↔snapshot turnover) at high churn. With
warm actors, slots free as fast as the work does, so correctness rides on the
in-memory reserve delta, not the periodic scan.

Run:   python examples/concurrency/high_churn_actors.py
Watch: poll /debug/leasor/queues and read the `run_concurrency` map for this run
       (CAP=3); also the run_concurrency_tracked_runs gauge.
Pass:  per-run active never exceeds CAP at any poll; run_concurrency_tracked_runs
       returns to 0 after the run finishes.
Note:  actors keep containers warm, so any cap excursion you see is leasor
       accounting, not pod startup latency.
"""

import asyncio

import flyte

env = flyte.TaskEnvironment(
    name="conc_high_churn_actors",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    reusable=flyte.ReusePolicy(
        replicas=(1, 4),
        idle_ttl=300,
        concurrency=100,
        scaledown_ttl=60,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse"),
)

CAP = 3
N = 300
SLEEP_S = 60  # default 0.2


@env.task
async def step(i: int) -> int:
    await asyncio.sleep(SLEEP_S)
    return i


# Driver runs in a non-reusable clone so the fan-out parent isn't itself an actor.
@env.clone_with(name="conc_high_churn_actors_main", reusable=None, depends_on=[env]).task
async def main(n: int = N) -> list[int]:
    tasks = [step(i) for i in range(n)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(max_action_concurrency=CAP).run(main, n=N)
    print(run.url)
    run.wait()
