"""
Pandera + Flyte: pandas (`pandera.typing.pandas.DataFrame`).

Demonstrates a `DataFrameModel` validated on task input/output via `flyteplugins-pandera`.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pt

import flyte

img = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyte>=2.0.0b52",
    "flyteplugins-pandera",
    "pandera[pandas]",
    pre=True,
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
    flyte.init_from_config(log_level=logging.DEBUG)

    run = flyte.with_runcontext("local").run(build_valid_employees)
    run.wait()
    df_out = run.outputs()[0]
    assert isinstance(df_out, pd.DataFrame)

    run2 = flyte.with_runcontext("local").run(pass_through, df=df_out)
    run2.wait()
    print("pandas pandera example OK:", run2.outputs()[0].shape)
