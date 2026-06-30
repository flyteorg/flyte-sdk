"""
Pandera + Flyte: Polars (`DataFrame` and `LazyFrame`).

Demonstrates `DataFrameModel` validated on task boundaries via `flyteplugins-pandera`,
including eager frames, lazy plans, and `collect()`.
Requires `flyteplugins-polars` for Polars structured dataset I/O.
"""

from __future__ import annotations

import argparse
import logging
from typing import Annotated

import pandera.polars as pa
import pandera.typing.polars as pt
import polars as pl
from flyteplugins.pandera import ValidationConfig

import flyte

img = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages(
        "flyteplugins-polars==2.0.9",
        "pandera[polars]",
    )
    .with_local_v2_plugins("flyteplugins-pandera")
)

env = flyte.TaskEnvironment(
    "pandera_polars_schema",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class EmployeeSchema(pa.DataFrameModel):
    """Employee rows: id and name."""

    employee_id: int = pa.Field(ge=0)
    name: str


class EmployeeSchemaWithStatus(EmployeeSchema):
    status: str = pa.Field(isin=["active", "inactive"])


class MetricsSchema(pa.DataFrameModel):
    """Simple metric columns (used for LazyFrame validation demo)."""

    item: str
    value: float


@env.task(report=True)
async def build_valid_employees() -> pt.DataFrame[EmployeeSchema]:
    return pl.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "name": ["Ada", "Grace", "Barbara"],
        }
    )


@env.task(report=True)
async def pass_through(df: pt.DataFrame[EmployeeSchema]) -> pt.DataFrame[EmployeeSchemaWithStatus]:
    return df.with_columns(pl.lit("active").alias("status"))


@env.task(report=True)
async def pass_through_with_error_warn(
    df: Annotated[pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")],
) -> Annotated[pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="warn")]:
    return df.drop("name")


@env.task(report=True)
async def pass_through_with_error_raise(
    df: Annotated[pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")],
) -> Annotated[pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="raise")]:  # raise is the default
    return df.drop("name")


@env.task(report=True)
async def metrics_eager() -> pt.DataFrame[MetricsSchema]:
    return pl.DataFrame({"item": ["a", "b"], "value": [1.0, 2.0]})


@env.task(report=True)
async def metrics_lazy() -> pt.LazyFrame[MetricsSchema]:
    return pl.LazyFrame({"item": ["x", "y"], "value": [3.0, 4.0]})


@env.task(report=True)
async def filter_metrics(lf: pt.LazyFrame[MetricsSchema]) -> pt.DataFrame[MetricsSchema]:
    return lf.filter(pl.col("value") > 0.0).collect()


@env.task(report=True)
async def main() -> pt.DataFrame[EmployeeSchemaWithStatus]:
    # Pandera still validates here, but HTML reports for df/df2 are not duplicated on this
    # parent task—only encode/decode inside worker tasks (build_*, pass_*) emit report tabs.
    df = await build_valid_employees()
    df2 = await pass_through(df)

    # error on the input
    await pass_through_with_error_warn(df.drop("employee_id"))

    await pass_through_with_error_warn(df.with_columns(pl.lit(-1).alias("employee_id")))

    # error on the output
    try:
        await pass_through_with_error_raise(df)
    except Exception as exc:
        print(exc)

    # LazyFrame: validated at task I/O; collect moves from lazy to eager DataFrame.
    _ = await metrics_eager()
    lazy = await metrics_lazy()
    _ = await filter_metrics(lazy)

    return df2


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

    run = flyte.with_runcontext(args.mode).run(main)
    print(run.url)
    run.wait()
    print("polars pandera example OK:", run.outputs()[0])
