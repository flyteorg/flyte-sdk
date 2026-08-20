"""
Every way to re-launch a prior run: rerun, recover, forced replay, and changed inputs.

`flyte.rerun` re-launches an *existing* run — it fetches that run's task spec and inputs from
the platform, so no local code is involved and nothing you edit locally is picked up. Two
behaviours live behind one verb:

* **rerun** (`recover=False`, the default) — a whole new run with the same inputs. Every action
  executes again, subject to global caching.
* **recover** (`recover=True`) — a whole new run with the same inputs, but every action that
  already succeeded in the source run is reused as-is; only what failed or never ran executes.
  Recovered actions land in the `RECOVERED` phase, which is terminal and success-equivalent.

Recovery is durability against *intermittent* failures — a flaky dependency, a node going away,
a credential that had expired. It is deliberately NOT a way to patch a run: replaying with new
code or new inputs is `flyte fork`, reserved for `pip install flyteplugins-union`. So
`recover=True` combined with changed inputs raises (demonstrated in `main()` below).

The pipeline here fans out four `prep` tasks and joins them in `flaky_join`, which fails unless
`FLAKY_OK=1` is present in the run's environment. That gives a run with four succeeded actions
and one failed one — exactly the shape recovery is for. The seed run fails; recovering it with
`FLAKY_OK=1` reuses the four `prep` results and re-executes only `flaky_join`.

Run the whole tour (remote only — rerun and recover are not supported locally):

    uv run python examples/rerun/rerun_and_recover.py

--------------------------------------------------------------------------------------------
Equivalent `flyte` CLI commands
--------------------------------------------------------------------------------------------
The script prints the seed run's name; substitute it for <RUN> below. `flyte rerun` covers the
same ground as `flyte.rerun`, except for changing inputs (no CLI flag for that yet — use the
Python API).

    # Seed run: launches local code, fails at flaky_join.
    flyte run examples/rerun/rerun_and_recover.py pipeline --n 4

    # 1. Pure rerun — everything executes again, subject to caching.
    flyte rerun <RUN>

    # 2. Recover — reuse the succeeded preps, re-run only flaky_join, with the env var it needs.
    flyte rerun <RUN> --recover -e FLAKY_OK=1

    # 3. Recover, but force one already-succeeded action to execute anyway. Action names are
    #    deterministic hashes, not positions, so list them first and copy the one you want:
    flyte get action <RUN>
    flyte rerun <RUN> --recover -e FLAKY_OK=1 --force-rerun-action <ACTION>

    # Repeatable — a listed parent re-enqueues its children, so list those too to force a subtree:
    flyte rerun <RUN> --recover -e FLAKY_OK=1 --force-rerun-action <A1> --force-rerun-action <A2>

    # Useful extras on any of the above: name the new run, stream its logs, retarget it.
    flyte rerun <RUN> --recover -e FLAKY_OK=1 --name recovered-1 --follow
    flyte rerun <RUN> -p my-project -d development

    # Inspect what recovery actually did — reused actions show up as RECOVERED:
    flyte get action <NEW_RUN>
"""

from __future__ import annotations

import asyncio
import os
from typing import List

import flyte
from flyte.remote import Action

env = flyte.TaskEnvironment(name="rerun_demo", resources=flyte.Resources(cpu=1, memory="500Mi"))

#: Env var the join step checks. Absent -> the seed run fails; `-e FLAKY_OK=1` -> it passes.
FLAKY_OK = "FLAKY_OK"


@env.task
async def prep(i: int) -> int:
    """A step that always succeeds. Slow enough that you can see recovery skip it."""
    await asyncio.sleep(5)
    print(f"prep({i}) ran")
    return i * i


@env.task
async def flaky_join(values: List[int]) -> int:
    """Stands in for a step with an intermittent, environment-shaped failure."""
    if os.environ.get(FLAKY_OK) != "1":
        raise RuntimeError(f"flaky_join failed: {FLAKY_OK}=1 was not set on this run")
    total = sum(values)
    print(f"flaky_join ran, total={total}")
    return total


@env.task
async def pipeline(n: int = 4) -> int:
    """Fan out `n` preps, then join them in the step that can fail."""
    values = await asyncio.gather(*[prep(i) for i in range(n)])
    return await flaky_join(list(values))


def summarize(run_name: str) -> None:
    """Print each action's phase. RECOVERED == reused from the source run, never re-executed."""
    for action in Action.listall(for_run_name=run_name):
        print(f"    {action.name:<28} {action.task_name or '-':<28} {action.phase}")


def main() -> None:
    flyte.init_from_config()

    # --- The seed run: local code, fails at flaky_join because FLAKY_OK is unset. -------------
    seed = flyte.run(pipeline, n=4)
    print(f"seed run: {seed.name}\n  {seed.url}")
    seed.wait()
    print(f"  finished in phase: {seed.phase}")
    summarize(seed.name)

    # --- 1. Pure rerun: a whole new run, same inputs, everything executes again. --------------
    #     CLI: flyte rerun <RUN>
    plain = flyte.rerun(seed.name)
    print(f"\n1. rerun -> {plain.name}\n  {plain.url}")

    # --- 2. Recover: reuse the succeeded preps, re-run only flaky_join. -----------------------
    #     The env var is what makes the retry actually pass — recovery reuses the source run's
    #     code and inputs, so the environment is the only lever left.
    #     CLI: flyte rerun <RUN> --recover -e FLAKY_OK=1
    recovered = flyte.with_runcontext(env_vars={FLAKY_OK: "1"}).rerun(seed.name, recover=True)
    print(f"\n2. rerun(recover=True) -> {recovered.name}\n  {recovered.url}")
    recovered.wait()
    print(f"  finished in phase: {recovered.phase}  (preps below should read RECOVERED)")
    summarize(recovered.name)

    # --- 3. Recover, but force one already-succeeded action to execute anyway. ----------------
    #     Names are deterministic hashes, so look one up rather than guessing.
    #     CLI: flyte get action <RUN>
    #          flyte rerun <RUN> --recover -e FLAKY_OK=1 --force-rerun-action <ACTION>
    a_prep = next(
        (a.name for a in Action.listall(for_run_name=seed.name) if a.task_name and "prep" in a.task_name),
        None,
    )
    if a_prep is None:
        print("\n3. skipped: no prep action found on the seed run")
    else:
        forced = flyte.with_runcontext(env_vars={FLAKY_OK: "1"}).rerun(
            seed.name,
            recover=True,
            force_rerun_actions=[a_prep],
        )
        print(f"\n3. rerun(recover=True, force_rerun_actions=[{a_prep!r}]) -> {forced.name}\n  {forced.url}")
        forced.wait()
        print(f"  finished in phase: {forced.phase}  ({a_prep} should have re-executed, not RECOVERED)")
        summarize(forced.name)

    # --- 4. Rerun with different inputs: same code, new parameters. ---------------------------
    #     Keyword arguments are converted against the interface fetched from the platform.
    #     No CLI equivalent yet — this one is Python-only.
    changed = flyte.with_runcontext(env_vars={FLAKY_OK: "1"}).rerun(seed.name, inputs={"n": 6})
    print(f"\n4. rerun(inputs={{'n': 6}}) -> {changed.name}\n  {changed.url}")

    # --- The boundary: changing inputs *while* recovering is fork, not rerun. -----------------
    #     Reserved for `pip install flyteplugins-union`; the SDK refuses it outright.
    try:
        flyte.rerun(seed.name, recover=True, n=6)
    except ValueError as e:
        print(f"\n5. recover + changed inputs is rejected, as designed:\n   {e}")


if __name__ == "__main__":
    main()
