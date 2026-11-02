"""
A reusable (warm-pool) task, deployed on its own. `reusable_override.py` calls it by reference.

Deploy it first:

    flyte deploy reusable_workers.py env
"""

import flyte

env = flyte.TaskEnvironment(
    name="reusable_workers",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    reusable=flyte.ReusePolicy(replicas=(1, 2), idle_ttl=300),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse"),
)


@env.task
async def square(x: int) -> int:
    return x**2
