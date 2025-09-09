import asyncio

import flyte
import flyte.errors

env = flyte.TaskEnvironment(
    name="fail",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
async def oomer(x: int):
    large_list = [0] * 100000000
    print(len(large_list))


@env.task
async def always_succeeds() -> int:
    await asyncio.sleep(1)
    return 42


@env.task
async def failure_recovery() -> int:
    try:
        await oomer(2)
    except flyte.errors.OOMError as e:
        print(f"Failed with oom trying with more resources: {e}, of type {type(e)}, {e.code}")
        try:
            await oomer.override(resources=flyte.Resources(cpu=1, memory="1Gi"))(5)
        except flyte.errors.OOMError as e:
            print(f"Failed with OOM Again giving up: {e}, of type {type(e)}, {e.code}")
            raise e
    finally:
        await always_succeeds()

    return await always_succeeds()


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())

    run = flyte.run(failure_recovery)
    print(run.url)
    run.wait(run)
