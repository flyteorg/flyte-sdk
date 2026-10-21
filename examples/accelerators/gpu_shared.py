"""Runs tasks that share one T4.

A "<model> shared" device asks for a share of a GPU instead of a whole one, so several tasks can
run on the same GPU at once. The cluster needs a T4 node pool that shares its GPUs with time
slicing or MPS. Time slicing doesn't split GPU memory, so tasks on the same GPU compete for it.
"""

import asyncio
import subprocess

import flyte

shared_env = flyte.TaskEnvironment("t4-shared", resources=flyte.Resources(gpu="T4 shared:1"))
env = flyte.TaskEnvironment("t4-shared-driver", depends_on=[shared_env])


@shared_env.task
def share() -> str:
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout


@env.task
async def main() -> list[str]:
    return list(await asyncio.gather(share.aio(), share.aio(), share.aio()))


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
