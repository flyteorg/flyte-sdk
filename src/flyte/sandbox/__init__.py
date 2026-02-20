"""Sandboxed tasks powered by Monty (Pydantic's Rust-based sandboxed Python interpreter).

.. warning:: Experimental feature: alpha — APIs may change without notice.

Sandboxed tasks are:
- **Side-effect free**: No filesystem, network, or OS access
- **Super fast**: Microsecond startup for pure Python
- **Multiplexable**: Many tasks run safely on the same Python process

Usage::

    import flyte.sandbox

    # Decorator approach (standalone)
    @flyte.sandbox.task
    def add(x: int, y: int) -> int:
        return x + y

    # Environment-based approach (preferred for ``flyte run``)
    env = flyte.TaskEnvironment(name="my-env")

    @env.sandbox.orchestrator
    def my_orchestrator(x: int, y: int) -> int:
        return add(x, y)

    # Create a reusable task from a code string
    pipeline = flyte.sandbox.orchestrator(
        "add(x, y) * 2",
        inputs={"x": int, "y": int},
        output=int,
        tasks=[add],
    )

    # One-shot execution of a code string (local only)
    result = await flyte.sandbox.orchestrate_local(
        "x + y",
        inputs={"x": 1, "y": 2},
    )
"""

from ._api import orchestrate_local, orchestrator, task
from ._code_task import CodeTaskTemplate
from ._config import SandboxedConfig
from ._task import SandboxedTaskTemplate

__all__ = [
    "CodeTaskTemplate",
    "SandboxedConfig",
    "SandboxedTaskTemplate",
    "orchestrate_local",
    "orchestrator",
    "task",
]
