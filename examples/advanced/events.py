import flyte

env = flyte.TaskEnvironment(name="events")


@env.task
async def next_task(value: int) -> int:
    return value + 1


@env.task
async def my_task(x: int) -> int:
    event = await flyte.new_event(
        name="my_event",
        scope="run",
        prompt="Is it ok to continue?",
        data_type=bool,
    )
    result = await event.wait()
    if result:
        return await next_task(x)
    else:
        return -1


if __name__ == "__main__":
    flyte.init()

    r = flyte.run(my_task, x=10)
    print(r.url)

    import time

    import flyte.remote as remote

    while not (remote_event := remote.Event.get("my_event", r.name)):
        time.sleep(10)

    remote_event.signal(True)
