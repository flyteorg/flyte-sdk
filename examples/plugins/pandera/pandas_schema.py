"""
Pandera + Flyte: pandas (`pandera.typing.pandas.DataFrame`).

Demonstrates a `DataFrameModel` validated on task input/output via `flyteplugins-pandera`.
"""

from __future__ import annotations

import argparse
import logging
from typing import Annotated

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pt
from flyteplugins.pandera import ValidationConfig

import flyte

img = flyte.Image.from_debian_base(python_version=(3, 12)).with_local_v2_plugins("flyteplugins-pandera")

env = flyte.TaskEnvironment(
    "pandera_pandas_schema",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class EmployeeSchema(pa.DataFrameModel):
    """Employee rows: id and name."""

    employee_id: int = pa.Field(ge=0)
    name: str


class EmployeeSchemaWithStatus(EmployeeSchema):
    status: str = pa.Field(isin=["active", "inactive"])


@env.task(report=True)
async def build_valid_employees() -> pt.DataFrame[EmployeeSchema]:
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "name": ["Ada", "Grace", "Barbara"],
        }
    )


@env.task(report=True)
async def pass_through(df: pt.DataFrame[EmployeeSchema]) -> pt.DataFrame[EmployeeSchemaWithStatus]:
    return df.assign(status="active")


@env.task(report=True)
async def pass_through_with_error_warn(
    df: Annotated[pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")],
) -> Annotated[pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="warn")]:
    del df["name"]
    return df


@env.task(report=True)
async def pass_through_with_error_raise(
    df: Annotated[pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")],
) -> Annotated[pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="raise")]:  # raise is the default
    del df["name"]
    return df


@env.task(report=True)
async def main() -> pt.DataFrame[EmployeeSchemaWithStatus]:
    # Pandera still validates here, but HTML reports for df/df2 are not duplicated on this
    # parent task—only encode/decode inside worker tasks (build_*, pass_*) emit report tabs.
    df = await build_valid_employees()
    df2 = await pass_through(df)

    # error on the input
    await pass_through_with_error_warn(df.drop(["employee_id"], axis="columns"))

    await pass_through_with_error_warn(df.assign(employee_id=-1))

    # error on the output
    try:
        await pass_through_with_error_raise(df)
    except Exception as exc:
        print(exc)

    return df2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandera + pandas Flyte example.")
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
    print("pandas pandera example OK:", run.outputs()[0])
