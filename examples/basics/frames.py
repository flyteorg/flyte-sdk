import pandas as pd

import flyte

img = flyte.Image.from_debian_base()
# assumes you have permission to push to the flyte default registry, if not
# img = img.clone(registry="ghcr.io/flyteorg", name="flyte")
img = img.with_pip_packages("pandas", "pyarrow")

env = flyte.TaskEnvironment(
    "frames",
    image=img,
)


@env.task
async def create_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person": ["Sean", "Pryce", "Jan", "Samuel", "Todd", "Chris"],
            "company": ["union", "union", "union", "other", "other", "other"],
        }
    )


@env.task
async def filter_df(input: pd.DataFrame, filter: str) -> pd.DataFrame:
    return input[input["company"] == filter]


@env.task
async def print_df(input: pd.DataFrame) -> list[str]:
    print(input)
    return input["person"].to_list()


@env.task
async def main(filter: str = "union") -> list[str]:
    output = await create_df()
    filtered_output = await filter_df(input=output, filter=filter)
    return await print_df(input=filtered_output)


if __name__ == "__main__":
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")

    run = flyte.with_runcontext(mode="local").run(main)

    # print(run.url)
    # run.wait(run)
