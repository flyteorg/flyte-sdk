from flyte import TaskEnvironment
import flyte


env = TaskEnvironment(name="my_task_env")

@env.task
async def my_other_task(y: int) -> int:
    return y * 2

@env.task
def my_task_sync(x: int) -> int:
    return x * 2

@env.task
async def my_task(x: int) -> int:
    a = await my_other_task(x)
    b = await my_task_sync.aio(x)
    return a + b + 1


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.with_runcontext(env_vars={"some": "vars"}).run(task=my_task, x=5)
