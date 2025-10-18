import flyte

env = flyte.TaskEnvironment(name="events")


@env.task
async def next_task(value: int) -> int:
    return value + 1


@env.task
async def my_task(x: int) -> int:
    event1 = await flyte.new_event.aio(
        "my_event",
        scope="run",
        prompt="Is it ok to continue?",
        data_type=bool,
    )
    event2 = await flyte.new_event.aio(
        "proceed_event",
        scope="run",
        prompt="What should I add to x?",
        data_type=int,
    )
    event3 = await flyte.new_event.aio(
        "final_event",
        scope="run",
        prompt="What should I return if the first event was negative?",
        data_type=int,
    )
    result = await event1.wait.aio()
    if result:
        print("Event signaled positive response, proceeding to next_task", flush=True)
        result2 = await event2.wait.aio()
        return await next_task(x + result2)
    else:
        print("Event signaled negative response, returning -1", flush=True)
        result3 = await event3.wait.aio()
        return result3


if __name__ == "__main__":
    flyte.init()

    r = flyte.run(my_task, x=10)
    print(r.url)
    print(r.outputs())

    # import flyte.remote as remote
    #
    # while not (remote_event := remote.Event.get("my_event", r.name)):
    #     time.sleep(10)
    #
    # remote_event.signal(True)
