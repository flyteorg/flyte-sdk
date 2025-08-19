import flyte

env = flyte.TaskEnvironment(
    name="friendly_names",
)


@env.task(name="my_task")
async def some_task() -> str:
    """
    This task has a friendly name that will be displayed in the UI.
    """
    return "Hello, Flyte!"


@env.task(name="entrypoint")
async def main() -> str:
    """
    This is the entrypoint task that will be displayed in the UI.
    """
    return await some_task()


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    r = flyte.run(main)
    print(r.url)
