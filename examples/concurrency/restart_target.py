"""Long-running capped run — the target for restart and abort chaos.

Long sleeps keep the run mid-flight at its cap for minutes, giving a wide window
to kill leasor (or abort the run) and verify the cap survives.

Restart chaos (the interesting path for this PR — boot-seed rehydration):
    kubectl delete pod -l app=leasor                      # single restart at cap
    while true; do kubectl delete pod -l app=leasor; sleep 15; done   # restart loop

  After each restart: active must stay <= CAP (RunSpec reloaded + boot
  CountActiveByRun seed ran before the scheduler started), the run still
  completes, and run_concurrency_tracked_runs returns to 0. Watch for
  over-dispatch on the first post-restart tick (the startup grace window).

Abort chaos: abort this run mid-flight (UI / `flyte abort <run-name>`); confirm
  the cascade completes and the gauge drains to 0.

Run:   python examples/concurrency/restart_target.py
Watch: /debug poll loop (LIMIT=3); the run name printed below is the abort handle.
"""

import asyncio

import flyte

env = flyte.TaskEnvironment(name="conc_restart_target")

# --- knobs ---
N = 12
CAP = 3
SLEEP_S = 120  # long, so the run sits at cap for minutes


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
    run = flyte.with_runcontext(max_action_concurrency=CAP).run(main, n=N)
    print("run name (abort handle):", run.name)
    print(run.url)
    run.wait()
