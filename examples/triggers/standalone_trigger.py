"""
Exercises the STANDALONE DeployTrigger path.

`remote.Trigger.create` sends a `trigger.TriggerSpec` *directly* in a `DeployTriggerRequest`
(backend handler: `trigger_service.go:DeployTrigger` / `flyte2/runs/service/trigger_service.go`).
This is a different code path from `@env.task(triggers=...)` + `flyte.deploy(env)`, which goes
through `DeployTask` with an embedded `TaskTriggerSpec` (`task_service.go:buildTriggerModels`).

So running this in addition to `flyte deploy ... env` covers the second entry point that writes
the `TriggerSpec` oneof (offloaded inputs) — the one the embedded-trigger deploy never hits.

Prereqs:
  1. flyte2 devbox running, API reachable at localhost:30080.
  2. The target task is ALREADY registered. `remote.Trigger.create` fetches it by name, so deploy
     it first, e.g.:
       flyte --config .flyte/config-oss-local.yaml deploy examples/triggers/basic_cached.py env

Run:
  python examples/triggers/standalone_trigger.py
"""

import flyte
from flyte.remote import Trigger

CONFIG = ".flyte/config-local.yaml"

# task_name = "<TaskEnvironment name>.<function name>", and it must already be registered.
#   - cached task (cache=ignored_inputs="start_time")  -> exercises the cache-ignore "don't fold" path
#   - non-cached custom_task (from basic.py)            -> exercises the "do fold" path
# Swap TASK_NAME / re-deploy the corresponding env to test each.
TASK_NAME = "cached_example_task.cached_task"
# TASK_NAME = "example_task.custom_task"   # needs: flyte deploy examples/triggers/basic.py env

TRIGGER_NAME = "standalone_minutely_"  # distinct from the embedded trigger names

flyte.init_from_config(CONFIG)

# minutely(...) binds the scheduled time to the "start_time" input via flyte.TriggerTime.
# remote.Trigger.create offloads the inputs (DataProxy UploadInputs) and stores the resulting
# OffloadedInputData on the TriggerSpec -> this is the standalone DeployTrigger write path.
Trigger.create(
    trigger=flyte.Trigger.minutely("start_time", name=TRIGGER_NAME),
    task_name=TASK_NAME,
)

print(f"Created standalone trigger '{TRIGGER_NAME}' on task '{TASK_NAME}'.")
