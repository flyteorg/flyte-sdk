# /// script
# dependencies = [
#    "polars",
#    "flyte>=2.0.0b5",
# ]
# ///

import polars as pl

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    image=flyte.Image.from_uv_script(__file__, name="flyte", registry="ghcr.io/flyteorg"),
)


@env.task
async def create_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"name": ["Alice", "Bob", "Charlie"], "age": [25, 32, 37], "city": ["New York", "Paris", "Berlin"]}
    )


@env.task
async def print_dataframe(dataframe: pl.DataFrame):
    print(dataframe)


@env.task
async def workflow():
    df = await create_dataframe()
    await print_dataframe(df)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(workflow)
    print(r)
