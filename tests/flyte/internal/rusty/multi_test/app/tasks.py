import flyte

env = flyte.TaskEnvironment(
    "multi_test_env",
)


@env.task
async def square(x: int) -> str:
    return f"{flyte.ctx().action.name} - {x**2}"
