"""
Sandbox — Code That Raises an Error
===================================

Minimal example: a sandbox whose code always raises. Run remotely to see
the failure surface as a task error.
"""

import flyte
import flyte.sandbox
from flyte.sandbox import sandbox_environment

env = flyte.TaskEnvironment(
    name="raise-error-demo",
    depends_on=[sandbox_environment],
)


boom_sandbox = flyte.sandbox.create(
    name="boom",
    code='raise ValueError(f"boom: x={x}")',
    inputs={"x": int},
    outputs={"result": int},
    cache="disable",
)


@env.task
async def trigger_failure(x: int = 1) -> int:
    return await boom_sandbox.run.aio(x=x)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(trigger_failure, x=42)
    print("Run URL:", run.url)
