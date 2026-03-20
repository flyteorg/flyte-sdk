"""
Pandera + Flyte: Polars (`pandera.typing.polars.DataFrame` and `LazyFrame`).

Requires `flyteplugins-polars` for Polars structured dataset I/O.
"""

from __future__ import annotations

import argparse
import logging

import pandera.typing.polars as pt
import polars as pl
from pandera.polars import DataFrameModel

import flyte

img = flyte.Image.from_debian_base(name="flyteplugins-pandera-polars").with_pip_packages(
    "flyte>=2.0.9",
    "flyteplugins-pandera",
    "flyteplugins-polars>=2.0.0b52",
    "pandera[polars]",
    pre=True,
)

env = flyte.TaskEnvironment(
    "pandera_polars_schema",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class MetricsSchema(DataFrameModel):
    """Simple metric columns."""

    item: str
    value: float


@env.task(report=True)
async def metrics_eager() -> pt.DataFrame[MetricsSchema]:
    return pl.DataFrame({"item": ["a", "b"], "value": [1.0, 2.0]})


@env.task(report=True)
async def metrics_lazy() -> pt.LazyFrame[MetricsSchema]:
    return pl.LazyFrame({"item": ["x", "y"], "value": [3.0, 4.0]})


@env.task(report=True)
async def filter_metrics(lf: pt.LazyFrame[MetricsSchema]) -> pt.DataFrame[MetricsSchema]:
    return lf.filter(pl.col("value") > 0.0).collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandera + Polars Flyte example.")
    parser.add_argument(
        "--mode",
        choices=("local", "remote"),
        default="remote",
        help="Run tasks locally or submit to a remote Flyte cluster.",
    )
    args = parser.parse_args()

    flyte.init_from_config(log_level=logging.DEBUG)

    r1 = flyte.with_runcontext(args.mode).run(metrics_eager)
    r1.wait()
    print("eager:", r1.outputs()[0])

    r2 = flyte.with_runcontext(args.mode).run(metrics_lazy)
    r2.wait()
    lf = r2.outputs()[0]

    r3 = flyte.with_runcontext(args.mode).run(filter_metrics, lf=lf)
    r3.wait()
    print("polars pandera example OK:", r3.outputs()[0])
