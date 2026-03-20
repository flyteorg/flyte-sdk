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

img = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages(
        "flyte>=2.0.9",
        # "flyteplugins-pandera",
        "pandera[pandas]",
        pre=True,
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


@env.task(report=True)
async def build_valid_employees() -> pt.DataFrame[EmployeeSchema]:
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "name": ["Ada", "Grace", "Barbara"],
        }
    )


@env.task(report=True)
async def pass_through(df: pt.DataFrame[EmployeeSchema]) -> pt.DataFrame[EmployeeSchema]:
    return df


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

    run = flyte.with_runcontext(args.mode).run(build_valid_employees)
    run.wait()
    df_out = run.outputs()[0]
    assert isinstance(df_out, pd.DataFrame)

    run2 = flyte.with_runcontext(args.mode).run(pass_through, df=df_out)
    run2.wait()
    print("pandas pandera example OK:", run2.outputs()[0].shape)
