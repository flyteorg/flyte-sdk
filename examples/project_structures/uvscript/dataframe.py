# /// script
# dependencies = [
#    "polars",
#    "flyte @ git+https://github.com/flyteorg/flyte-sdk.git@d2110538e008b1be0ae949055ce37dad839cfa0a",
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
    flyte.init_from_config()
    r = flyte.run(workflow)
    print(r)
