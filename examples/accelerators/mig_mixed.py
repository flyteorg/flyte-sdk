"""Runs tasks that ask for different MIG profiles of an RTX PRO 6000.

The cluster needs an RTX PRO 6000 node pool that uses the MIG strategy mixed with a layout that has
both profiles, such as all-balanced (two 1g.24gb slices and one 2g.48gb slice per GPU). Then all
three tasks can run on the same GPU at once.
"""

import asyncio
import subprocess

import flyte

small_env = flyte.TaskEnvironment(
    "rtx-pro-6000-1g",
    resources=flyte.Resources(gpu=flyte.GPU("RTX_PRO_6000", 1, partition="1g.24gb")),
)
large_env = flyte.TaskEnvironment(
    "rtx-pro-6000-2g",
    resources=flyte.Resources(gpu=flyte.GPU("RTX_PRO_6000", 1, partition="2g.48gb")),
)
env = flyte.TaskEnvironment("rtx-pro-6000-mixed", depends_on=[small_env, large_env])


def gpus() -> str:
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout


@small_env.task
def small() -> str:
    return gpus()


@large_env.task
def large() -> str:
    return gpus()


@env.task
async def main() -> list[str]:
    return list(await asyncio.gather(small.aio(), small.aio(), large.aio()))


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
