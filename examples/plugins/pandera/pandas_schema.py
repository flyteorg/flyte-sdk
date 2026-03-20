"""
Pandera + Flyte: pandas (`pandera.typing.pandas.DataFrame`).

Demonstrates a `DataFrameModel` validated on task input/output via `flyteplugins-pandera`.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pt

import flyte
import flyte.io

img = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages(
        # "flyte>=2.0.9",
        # "flyteplugins-pandera",
        "pandera[pandas]",
        "pyarrow",
        "fastparquet",
    )
    .with_local_v2_plugins("flyteplugins-pandera")
)

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
async def main() -> pt.DataFrame[EmployeeSchemaWithStatus]:
    # Pandera still validates here, but HTML reports for df/df2 are not duplicated on this
    # parent task—only encode/decode inside worker tasks (build_*, pass_*) emit report tabs.
    df = await build_valid_employees()
    df2 = await pass_through(df)
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
