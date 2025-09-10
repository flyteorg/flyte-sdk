# /// script
# requires-python = "==3.12"
# dependencies = [
#    "mashumaro",
# ]
# ///

import asyncio
from functools import partial

import flyte

env = flyte.TaskEnvironment(
    name="recursion",
    image=flyte.Image.from_uv_script(__file__, registry="ghcr.io/flyteorg", name="flyte"),
)


@env.task
async def tree(max_depth: int = 3, n_children: int = 2, index: int = 1, depth: int = 1) -> int:
    """
    A recursive function that creates a tree of tasks.
    """

    if depth >= max_depth:
        return 0

    task = partial(tree, max_depth=max_depth, depth=depth + 1, n_children=n_children)

    base = index * n_children

    runs = [task(index=base + i) for i in range(n_children)]

    children = await asyncio.gather(*runs)

    return n_children + sum(children)


async def main_builder():
    from flyte._internal.imagebuild.image_builder import ImageBuildEngine

    return await ImageBuildEngine.build(env.image)


async def main():
    await flyte.init_from_config.aio()
    out = await flyte.run.aio(tree, max_depth=2, n_children=2)

    print(f"Total nodes in the tree: {out}")


if __name__ == "__main__":
    # asyncio.run(main_builder())
    asyncio.run(main())
