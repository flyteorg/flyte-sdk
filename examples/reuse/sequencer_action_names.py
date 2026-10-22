"""
Do sub-action names stay stable across retries when the parent runs in a reusable container?

Every call to a task gets a deterministic action name:

    name = hash(parent action name, inputs hash, task identity, call sequence [, group])

The call sequence disambiguates identical calls, so in

    @env.task
    async def main():
        await foo()  # seq 1
        await foo()  # seq 2
        await foo()  # seq 3

the three foo() calls get three distinct but *reproducible* names. A retry of `main` (or a recovery
of the run) re-executes the body, re-derives the same three names, and the engine hands back the
already-completed children instead of running them again.

The sequence counters live on the controller, which is a process-wide singleton. In a regular
container every attempt is a fresh process, so the counters always start at zero. In a reusable
container the process (and the controller) outlives the action, so if the counters are not scoped
to the action *attempt*, attempt 2 continues counting at seq 4, 5, 6 -> new names -> foo runs again.

This example proves which behavior you get:

* `main` runs in a reusable environment with a single replica, so every attempt lands in the same
  process. It calls `foo()` three times with identical inputs, then fails its first attempt.
* `foo` returns its own action name, and the driver lists the children of the run afterwards.

Expected (sequencer scoped per action attempt): 3 foo actions per run, identical names across runs.
Bug (sequencer leaks across attempts):         6 foo actions per run; the retry minted 3 new names.

Usage:
    python sequencer_action_names.py                 # user-error retry
    python sequencer_action_names.py --mode timeout  # timeout retry (attempt is cancelled)
"""

import argparse
import asyncio
from typing import List

import flyte
import flyte.remote
import flyte.storage

actor_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse")

N_CALLS = 3
TIMEOUT_S = 60

# foo lives in a plain environment so it never competes with `main` for the single reusable slot.
leaf_env = flyte.TaskEnvironment(
    name="seq_leaf",
    resources=flyte.Resources(memory="250Mi", cpu=1),
    image=actor_image,
)

# replicas=1 pins every attempt of `main` to the same long-lived process (and controller).
reusable_env = flyte.TaskEnvironment(
    name="seq_reusable",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    reusable=flyte.ReusePolicy(replicas=1, idle_ttl=600),
    image=actor_image,
    depends_on=[leaf_env],
)


@leaf_env.task
async def foo() -> str:
    return flyte.ctx().action.name


async def _is_first_attempt() -> bool:
    """flyte.ctx().attempt_number reads FLYTE_ATTEMPT_NUMBER, which is set once per process and so stays 0
    in a reused process. A marker in the run's storage survives the attempt (and the process, if the
    worker tears it down after a timeout)."""
    marker = flyte.storage.join(flyte.ctx().run_base_dir, "seq-repro-first-attempt")
    if await flyte.storage.exists(marker):
        return False
    await flyte.storage.put_stream(b"1", to_path=marker)
    return True


@reusable_env.task(retries=1, timeout=TIMEOUT_S)
async def main(mode: str = "error") -> List[str]:
    first_attempt = await _is_first_attempt()
    # Identical calls: the only thing distinguishing them is the call sequence.
    names = [await foo() for _ in range(N_CALLS)]
    print(f"first_attempt={first_attempt} foo action names={names}", flush=True)

    if first_attempt and mode != "none":
        if mode == "timeout":
            await asyncio.sleep(TIMEOUT_S * 2)
        raise RuntimeError("failing the first attempt on purpose to force a retry")
    return names


def foo_children(run: flyte.remote.Run) -> List[str]:
    """Names of every foo action created under the root action of the run (all attempts)."""
    return [
        a.name
        for a in flyte.remote.Action.listall(for_run_name=run.name, parent_name="a0")
        if (a.task_name or "").endswith("foo")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "error", "timeout"], default="error")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    flyte.init_from_config()

    runs = [flyte.run(main, mode=args.mode) for _ in range(args.runs)]
    for i, run in enumerate(runs):
        print(f"run {i}: {run.url}", flush=True)

    results = []
    for i, run in enumerate(runs):
        run.wait(quiet=True)
        children = foo_children(run)
        print(f"run {i}: phase={run.phase}")
        print(f"run {i}: {len(children)} foo actions: {sorted(children)}")
        results.append(children)

    print()
    within_run_ok = all(len(c) == N_CALLS for c in results)  # trivially true for --mode none
    across_runs_ok = all(sorted(c) == sorted(results[0]) for c in results)
    print(f"retry reused the first attempt's children (expect {N_CALLS} per run): {within_run_ok}")
    print(f"identical foo action names across runs:                    {across_runs_ok}")
    if not within_run_ok:
        print("BUG: the retry minted new foo names -> the call sequencer leaked across attempts in the reused process")
