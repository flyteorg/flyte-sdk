import flyte

env = flyte.TaskEnvironment(
    name="resources",
    resources=flyte.Resources(
        cpu="1",
        memory="1Gi",
        shm="auto",
        disk="1Gi",
        gpu="A100 80G:8",
    ),
)


@env.task
async def my_task(x: int) -> int:
    return x + 1


@env.task
async def my_task2(x: int) -> int:
    return await my_task(x) + 1


if __name__ == "__main__":
    print(my_task(x=1))
