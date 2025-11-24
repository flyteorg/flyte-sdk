import flyte

env = flyte.TaskEnvironment("custom_context")


@env.task
async def downstream_task(x: int) -> int:
    custom_ctx = flyte.ctx().custom_context
    if "increment" not in custom_ctx:
        raise ValueError("Expected 'increment' in custom context")
    return x + int(custom_ctx["increment"])


@env.task
async def main(x: int) -> int:
    vals = []
    for i in range(3):
        with flyte.custom_context(increment=str(i)):
            vals.append(await downstream_task(x))
    return sum(vals)


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(main, x=10).url)
