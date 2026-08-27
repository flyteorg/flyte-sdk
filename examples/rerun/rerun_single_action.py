"""
Re-run a single action out of a prior run, instead of the whole thing.

`rerun` normally re-launches a whole run: it sources the task spec and inputs from the run's
root action, `a0`. Point `action_name` at a *child* action instead and the new run is rooted at
that action's task, executed with the exact inputs it received inside the original run.

That is the "iterate on one failing step" loop. A leaf task blew up halfway through an
expensive pipeline; you want to poke at just that task with just those inputs, without dragging
its parents along and without hand-copying arguments out of the UI.

Two things to know:

* **Action names are deterministic hashes, not positions.** They are derived from the parent
  action, the task identity, the input hash, and the call sequence — stable across runs of the
  same workflow, but not guessable. List them with `flyte get action <run>` or
  `flyte.remote.Action.listall(for_run_name=...)`, which is what this example does.
* **It cannot be combined with recovery.** Recovery reuses succeeded actions from the source run
  by matching those same names, and a run rooted at a single action has a different action tree,
  so the reuse set would not line up. `recover=True` with a sub-action raises; the CLI rejects
  `--action-name --recover` outright. Both are demonstrated at the end of `main()`.

The pipeline below fans out `analyze` over several shards. One shard (`BAD_SHARD`) always fails,
so the seed run leaves you with a failed leaf worth re-running on its own.

    uv run python examples/rerun/rerun_single_action.py

--------------------------------------------------------------------------------------------
Equivalent `flyte` CLI commands
--------------------------------------------------------------------------------------------
Substitute the seed run's name (printed by the script) for <RUN>.

    # Seed run: launches local code, fails in one analyze shard.
    flyte run examples/rerun/rerun_single_action.py fan_out --shards 4

    # Find the action you want — names, task names and phases for the whole run.
    flyte get action <RUN>

    # Re-run just that action: a new run rooted at its task, with the inputs it received.
    flyte rerun <RUN> --action-name <ACTION>

    # Composes with the usual options — name it, watch it.
    flyte rerun <RUN> --action-name <ACTION> --name just-that-step --follow

    # Rejected: re-running one action is always a plain re-execution.
    flyte rerun <RUN> --action-name <ACTION> --recover
"""

from __future__ import annotations

import asyncio

import flyte
from flyte.models import ActionPhase
from flyte.remote import Action

env = flyte.TaskEnvironment(name="rerun_action_demo", resources=flyte.Resources(cpu=1, memory="500Mi"))

#: The shard whose `analyze` call always fails, giving us a failed leaf to re-run on its own.
BAD_SHARD = 2


@env.task
async def analyze(shard: int) -> int:
    """One unit of work. Fails for a single shard, succeeds for the rest."""
    await asyncio.sleep(2)
    if shard == BAD_SHARD:
        raise RuntimeError(f"analyze(shard={shard}) failed — this is the action worth re-running alone")
    print(f"analyze(shard={shard}) ran")
    return shard * 10


@env.task
async def fan_out(shards: int = 4) -> int:
    """Fan `analyze` out over shards. One of them takes the whole run down with it."""
    results = await asyncio.gather(*[analyze(s) for s in range(shards)])
    return sum(results)


def find_failed_action(run_name: str) -> str | None:
    """The action we want to re-run: the failed `analyze` leaf, found by phase, not position."""
    for action in Action.listall(for_run_name=run_name, in_phase=[ActionPhase.FAILED]):
        # Skip the root action — re-running a0 is just a whole-run rerun.
        if action.name != "a0" and action.task_name and "analyze" in action.task_name:
            return action.name
    return None


def main() -> None:
    flyte.init_from_config()

    # --- The seed run: one analyze shard fails, taking the run with it. -----------------------
    seed = flyte.run(fan_out, shards=4)
    print(f"seed run: {seed.name}\n  {seed.url}")
    seed.wait()
    print(f"  finished in phase: {seed.phase}")

    print("\nactions in the seed run:")
    for action in Action.listall(for_run_name=seed.name):
        print(f"    {action.name:<28} {action.task_name or '-':<28} {action.phase}")

    target = find_failed_action(seed.name)
    if target is None:
        print("\nno failed analyze action found — nothing to re-run in isolation")
        return

    # --- Re-run just that action, with the exact inputs it got inside the seed run. -----------
    #     CLI: flyte rerun <RUN> --action-name <ACTION>
    only = flyte.rerun(seed.name, action_name=target)
    print(f"\nrerun(action_name={target!r}) -> {only.name}\n  {only.url}")
    only.wait()
    print(f"  finished in phase: {only.phase}  (this run's a0 IS the analyze task)")
    for action in Action.listall(for_run_name=only.name):
        print(f"    {action.name:<28} {action.task_name or '-':<28} {action.phase}")

    # --- The boundary: a single action is always a plain re-execution. ------------------------
    #     CLI: flyte rerun <RUN> --action-name <ACTION> --recover   ->  usage error
    try:
        flyte.rerun(seed.name, action_name=target, recover=True)
    except ValueError as e:
        print(f"\naction_name + recover is rejected, as designed:\n   {e}")


if __name__ == "__main__":
    main()
