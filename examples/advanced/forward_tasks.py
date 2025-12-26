import flyte

env = flyte.TaskEnvironment("forward-tasks")


@env.task
async def inner_task(x: int) -> int:
    return x + 1


@env.task
async def outer_task(x: int) -> int:
    # you can invoke any task function, directly without invoking on remote using .forward() method
    v = await inner_task.forward(x=10)
    return await inner_task(v)


@env.task
def sync_inner_task(x: int) -> int:
    return x + 1


@env.task
def sync_outer_task(x: int) -> int:
    v = sync_inner_task.forward(x=10)
    return sync_inner_task(v)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(outer_task, 10)
    print(r.url)
    r = flyte.run(sync_outer_task, 10)
    print(r.url)
