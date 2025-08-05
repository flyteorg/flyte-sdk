import asyncio

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="exc_handler", resources=flyte.Resources(cpu=1, memory="250Mi"))


@env.task
async def oomer(x: int):
    large_list = [0] * 100000000
    print(len(large_list))


@env.task
async def always_fails(x: int) -> int:
    raise ValueError(f"This always fails {x}")


@env.task
async def always_succeeds() -> int:
    await asyncio.sleep(1)
    return 42


@env.task
async def failure_recovery() -> int:
    try:
        await always_fails(1)
    except flyte.errors.RuntimeUserError as e:
        if e.code == "ValueError":
            print(f"Caught exception: {e}, of type {type(e)}, {e.code}")
            try:
                await oomer(2)
            except flyte.errors.OOMError as e:
                print(f"Failed with oom trying with more resources: {e}, of type {type(e)}, {e.code}")
                try:
                    await oomer.override(resources=flyte.Resources(cpu=1, memory="1Gi"))(5)
                except flyte.errors.OOMError as e:
                    print(f"Failed with OOM Again giving up: {e}, of type {type(e)}, {e.code}")
                    raise e
        else:
            print(f"Caught exception: {e}, of type {type(e)}, {e.code}")
            raise e
    return await always_succeeds()


if __name__ == "__main__":
    # print(asyncio.run(failure_recovery()))
    flyte.init_from_config("../../config.yaml")
    print(flyte.run(failure_recovery))
