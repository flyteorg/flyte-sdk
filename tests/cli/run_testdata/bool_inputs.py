import flyte

env = flyte.TaskEnvironment(name="bool_inputs")


@env.task
async def flags(block: bool = False, verbose: bool = True) -> str:
    return f"{block} {verbose}"
