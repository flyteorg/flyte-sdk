"""Human-in-the-Loop (HITL) plugin for Flyte.

This plugin provides an event-based API for pausing workflows and waiting for human input.

## Basic usage:

```python
import flyte
import flyteplugins.hitl as hitl

task_env = flyte.TaskEnvironment(
    name="my-hitl-workflow",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[hitl.env],
)


@task_env.task(report=True)
async def main() -> int:
    # Create an event (this serves the app if not already running)
    event = await hitl.new_event.aio(
        "integer_input_event",
        data_type=int,
        scope="run",
        prompt="What should I add to x?",
    )
    y = await event.wait.aio()
    return y
```

## Features:

- Event-based API for human-in-the-loop workflows
- Web form for human input
- Programmatic API for automated input
- Support for int, float, str, and bool data types
- Crash-resilient polling with object storage
"""

from ._event import Event, EventScope, event_task_env, new_event

__all__ = [
    "Event",
    "EventScope",
    "env",
    "new_event",
]

__version__ = "0.1.0"

# Expose the task environment as `env` for user convenience
env = event_task_env
