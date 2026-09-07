"""Fire a deployed trigger from Python with `flyte.run`.

A trigger is a saved launch configuration for a task: inputs, env vars, queue,
notifications. Scheduled and artifact triggers fire on their own; a trigger
with no automation (see `manual.py`) only fires when something asks for it.
Either kind can be fired on demand by fetching it and passing it to
`flyte.run`, exactly like a task:

    trigger = flyte.remote.Trigger.get(name="full-report", task_name=...)
    run = flyte.run(trigger)            # everything the trigger was deployed with
    run = flyte.run(trigger, days=3)    # override one input, keep the rest

The run is created *as* the trigger: the platform records the trigger as its
origin (same as a scheduled fire), and the run carries the trigger's inputs,
env vars, queue and notification rules. Keyword arguments override individual
inputs; anything left out keeps the value the trigger was deployed with.
`flyte.with_runcontext(...)` layers further overrides (env vars, queue, run
name, ...) on top of the trigger's run spec.

Try it, after deploying `manual.py`:

    flyte deploy examples/triggers/manual.py env
    python examples/triggers/programmatic.py
"""

import flyte
import flyte.remote

TASK_NAME = "manual_trigger_example.report_on_demand"


def fire_as_deployed() -> flyte.remote.Run:
    """Fire `full-report` with exactly what it was deployed with (region="all", days=30)."""
    trigger = flyte.remote.Trigger.get(name="full-report", task_name=TASK_NAME)
    return flyte.run(trigger)


def fire_with_overrides() -> flyte.remote.Run:
    """Override one input; `region` keeps the trigger's value, `days` becomes 3."""
    trigger = flyte.remote.Trigger.get(name="full-report", task_name=TASK_NAME)
    return flyte.run(trigger, days=3)


def fire_with_runcontext() -> flyte.remote.Run:
    """Layer run-level overrides on top of the trigger's run spec.

    The trigger's own env vars and notification rules are kept; `EXTRA_FLAG` is added and
    the run gets a fixed name. Anything `with_runcontext` sets wins over the trigger's value.
    """
    trigger = flyte.remote.Trigger.get(name="quick-report", task_name=TASK_NAME)
    return flyte.with_runcontext(env_vars={"EXTRA_FLAG": "1"}, name="quick-report-from-python").run(trigger)


def fire_every_trigger_on_task() -> list[flyte.remote.Run]:
    """Triggers from `listall()` can be fired too; their details are fetched on demand.

    Scheduled triggers (`nightly` here) fire just fine off-schedule: the platform stamps the
    run start time, and any `flyte.TriggerTime` input is filled from it.
    """
    runs = []
    for trigger in flyte.remote.Trigger.listall(task_name=TASK_NAME):
        runs.append(flyte.run(trigger))
    return runs


if __name__ == "__main__":
    flyte.init_from_config()

    run = fire_with_overrides()
    print(f"fired full-report with days=3: {run.url}")
    run.wait()
    print(f"phase={run.phase} inputs={run.inputs()} outputs={run.outputs()}")

    run = fire_with_runcontext()
    print(f"fired quick-report with run-context overrides: {run.url}")
