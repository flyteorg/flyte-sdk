import flyte

env = flyte.TaskEnvironment(
    name="my_module",
)


@env.task
async def say_hello(name: str) -> str:
    """A simple task that greets the user."""
    return f"Hello, {name}!"
