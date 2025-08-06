import asyncio
import pathlib
import os

import flyte
import flyte.errors

PATH_TO_FASTTASK_WORKER = pathlib.Path("/Users/ytong/go/src/github.com/unionai/flyte/fasttask/worker-v2")

actor_image = (
    flyte.Image.from_debian_base(install_flyte=False)
    .with_apt_packages("curl", "build-essential", "ca-certificates", "pkg-config", "libssl-dev")
    .with_commands(["sh -c 'curl https://sh.rustup.rs -sSf | sh -s -- -y'"])
    .with_env_vars({"PATH": "/root/.cargo/bin:${PATH}"})
    .with_source_folder(PATH_TO_FASTTASK_WORKER, "/root/fasttask")
    .with_pip_packages("uv")
    .with_workdir("/root/fasttask")
    .with_commands(["uv sync --reinstall --active"])
    .with_local_v2()
)
# hopefully this makes it not need to be rebuilt every time
object.__setattr__(actor_image, "_tag", "9043815457d6422e4adb4fb83c5d3c5a")
# ghcr.io/flyteorg/flyte:9043815457d6422e4adb4fb83c5d3c5a

env = flyte.TaskEnvironment(
    name="oomer_parent_actor",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=2,  # Min of 2 replicas are needed to ensure no-starvation of tasks.
        idle_ttl=60,
    ),
)

leaf_env = flyte.TaskEnvironment(
    name="leaf_env",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
async def oomer(x: int):
    print("Leaf (oomer) Environment Variables:", os.environ, flush=True)
    print("About to allocate a large list... should oom", flush=True)
    await asyncio.sleep(1)
    large_list = [0] * 100000000
    print(len(large_list))


@env.task
async def always_succeeds() -> int:
    await asyncio.sleep(1)
    return 42


@env.task
async def failure_recovery() -> int:
    print("A0 (failure recovery) Environment Variables:", os.environ, flush=True)
    await asyncio.sleep(240)
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
        # await always_succeeds()
        print("In finally...")

    return await always_succeeds()


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")

    run = flyte.run(failure_recovery)
    print(run.url)
    run.wait(run)
