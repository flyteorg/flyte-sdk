"""Basic task type checking test."""

from flyte import TaskEnvironment

env = TaskEnvironment(name="test_env")


@env.task
async def simple_task(x: int) -> int:
    """Simple async task with proper types."""
    return x * 2


@env.task
def sync_task(x: str) -> str:
    """Simple sync task with proper types."""
    return x.upper()


@env.task
async def multi_input_task(x: int, y: str, z: float = 1.0) -> tuple[int, str, float]:
    """Task with multiple inputs and tuple return."""
    a = await simple_task(x)
    b = await sync_task.aio(y)
    return a, b, z
