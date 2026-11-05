"""Tasks that orchestrate other tasks.

`@flyte.orchestrator.task` turns a function that only calls other tasks into a
task that Flyte runs without a pod of its own, so it takes up none of the task
pod resources a regular task does. The tasks it calls run as usual.

```python
import flyte
import flyte.orchestrator
import flyte.remote

add = flyte.remote.Task.get("math.add", auto_version="latest")
multiply = flyte.remote.Task.get("math.multiply", auto_version="latest")


@flyte.orchestrator.task
def pipeline(x: int, y: int) -> int:
    return multiply(add(x, y), 2)


run = flyte.run(pipeline, x=3, y=4)
```

It may only call tasks. For an orchestrator that also calls traces or durable
operations, use `env.orchestrator.task` on a `flyte.TaskEnvironment`, which runs
in the environment's image.
"""

from ._task import OrchestratorTaskTemplate, task

__all__ = [
    "OrchestratorTaskTemplate",
    "task",
]
