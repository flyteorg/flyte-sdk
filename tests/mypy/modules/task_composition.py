"""Task composition type checking test."""

from flyte import TaskEnvironment

env = TaskEnvironment(name="composition_env")


@env.task
async def step_one(x: int) -> int:
    """First step in composition."""
    return x * 2


@env.task
async def step_two(y: int) -> str:
    """Second step in composition."""
    return f"Result: {y}"


@env.task
async def composed_task(x: int) -> str:
    """Composed task that calls other tasks."""
    intermediate = await step_one(x)
    result = await step_two(intermediate)
    return result


@env.task
def sync_step(x: int) -> int:
    """Synchronous step."""
    return x + 10


@env.task
async def mixed_composition(x: int) -> int:
    """Mix async and sync tasks."""
    sync_result = await sync_step.aio(x)
    async_result = await step_one(sync_result)
    return async_result
