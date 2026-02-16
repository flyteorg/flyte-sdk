import asyncio
from typing import List

import flyte

# PATH_TO_FASTTASK_WORKER = pathlib.Path("../../../private/flyte/fasttask/worker-v2")
#
# actor_image = (
#     flyte.Image.from_debian_base(install_flyte=False)
#     .with_apt_packages("curl", "build-essential", "ca-certificates", "pkg-config", "libssl-dev")
#     .with_commands(["sh -c 'curl https://sh.rustup.rs -sSf | sh -s -- -y'"])
#     .with_env_vars({"PATH": "/root/.cargo/bin:${PATH}"})
#     .with_source_file(pathlib.Path(".dockerignore"))
#     .with_source_folder(PATH_TO_FASTTASK_WORKER, "/root/fasttask")
#     .with_pip_packages("uv")
#     .with_workdir("/root/fasttask")
#     .with_commands(["uv sync --reinstall --active"])
#     .with_local_v2()
# )

actor_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse")

env = flyte.TaskEnvironment(
    name="reusable",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    reusable=flyte.ReusePolicy(
        replicas=(1, 4),  # Min of 2 replacas are needed to ensure no-starvation of tasks.
        idle_ttl=300,
    ),
    image=actor_image,
)


@env.task
async def square(x: int) -> int:
    return x**2


@env.task
async def cube(x: int) -> int:
    return x**3


# Clone the environment with a different name and no reuse policy.
cloned_env = env.clone_with(name="nonreusable", reusable=None, depends_on=[env])


# This task will run in the cloned environment without reuse.
@cloned_env.task
async def main(n: int) -> List[int]:
    """
    Run square and cube tasks in parallel for the range of x_list.
    """
    square_coros = []
    cube_coros = []
    for x in range(n):
        square_coros.append(square(x))
        cube_coros.append(cube(x))
    return await asyncio.gather(*square_coros) + await asyncio.gather(*cube_coros)


if __name__ == "__main__":
    flyte.init_from_config()  # establish remote connection from within your script.
    run = flyte.run(main, n=30)  # run remotely inline and pass data.
    print(run.url)
    run.wait()  # wait for the run to finish.

    # # print various attributes of the run.
    # print(run.name)
    # print(run.url)
    #
    # run.wait()  # stream the logs from the root action to the terminal.
