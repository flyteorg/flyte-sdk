import asyncio

from async_lru import alru_cache

import flyte

env_reuse = flyte.TaskEnvironment(
    name="reusable_with_mem",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    reusable=flyte.ReusePolicy(
        replicas=1,
        idle_ttl=300,
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.3"),
)

env = env_reuse.clone_with(
    name="nonreusable_with_mem",
    reusable=None,
    depends_on=[env_reuse],
)


class ValueCalculator:
    """Encapsulates stateful computation to avoid global variables."""

    def __init__(self, initial_value: int = 10):
        self.v = initial_value

    @alru_cache(maxsize=1)
    async def get_value(self, x: int) -> int:
        """
        Simulate a long-running computation.
        """
        result = x * self.v
        self.v = self.v + 1
        return result


# Create a singleton instance for the reusable environment, per replica
# This is stateful while the container is running, but not across replicas or across restarts.
# DO NOT USE this pattern for global state like a database. This is for reducing redundant computations,
# or reusing expensive operations opportunistically, like loading a model or a large dataset to memory or gpu.
# For global state, use a database or a distributed cache.
_calculator = ValueCalculator()


async def get_value(x: int) -> int:
    """Wrapper function to maintain the same interface."""
    return await _calculator.get_value(x)


@env_reuse.task
async def add(x: int, multiplier: int) -> int:
    """
    Add a value to the result of get_value.
    """
    value = await get_value(multiplier)
    return value + x


@env.task
async def main(n: int) -> list[int]:
    """
    Run add in parallel for the range of x_list.
    """
    coros = [add(x, 2) for x in range(n)]
    return await asyncio.gather(*coros)


if __name__ == "__main__":
    flyte.init_from_config()  # establish remote connection from within your script.
    run = flyte.run(main, n=30)  # run remotely inline and pass data.
    print(run.url)
    run.wait()  # wait for the run to finish.
