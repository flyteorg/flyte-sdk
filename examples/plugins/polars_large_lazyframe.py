from typing import Any


from polars.lazyframe.frame import LazyFrame


import flyte
import polars as pl


env = flyte.TaskEnvironment(
    name="polars",
    resources=flyte.Resources(memory="1Gi"),
    image=flyte.Image.from_debian_base(
        install_flyte=True, flyte_version="2.0.0b52"
    ).with_pip_packages("flyteplugins-polars", pre=True),
)


@env.task
async def get_big_lazyframe() -> pl.LazyFrame:
    # N_ROWS = 10_000
    N_ROWS = 10
    N_REPEATS = 5_000  # total rows = N_ROWS * N_REPEATS
    STR_SIZE = 32  # bytes per string (approx)

    lf = pl.DataFrame({
        "id": pl.arange(0, N_ROWS, eager=True),
        "value": (pl.arange(0, N_ROWS, eager=True) * 0.1),
        "payload_a": pl.select(pl.lit("x" * STR_SIZE).repeat_by(N_ROWS).flatten()),
        "payload_b": pl.select(pl.lit("y" * STR_SIZE).repeat_by(N_ROWS).flatten()),
    }).lazy()

    return pl.concat([lf] * N_REPEATS)


@env.task
async def preprocess_data(lf: pl.LazyFrame) -> str:
    df = lf.collect()
    print(df.estimated_size("mb"))
    print(df)
    return f"{int(df.estimated_size('mb'))}MB"


@env.task
async def main() -> str:
    print("polars fun")
    lf = await get_big_lazyframe[Any, Any, LazyFrame]()
    return await preprocess_data(
        lf=lf
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
