"""
Pandera + Flyte: Dask (`pandera.typing.dask.DataFrame`).

Dask uses the pandas :class:`~pandera.pandas.DataFrameModel` with ``dask.dataframe``.

.. note::
    Flyte does not ship a Dask dataframe encoder in this repo. This script
    documents the pandera + Flyte type annotation; register Dask handlers or
    materialize via pandas/polars for remote I/O.
"""

from __future__ import annotations

import argparse
import logging

import dask.dataframe as dd
import pandas as pd
import pandera.pandas as pa
import pandera.typing.dask as pt

import flyte

img = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyte>=2.0.9",
    "flyteplugins-pandera",
    "dask[dataframe]",
    "pandera[pandas]",
    pre=True,
)

env = flyte.TaskEnvironment(
    "pandera_dask_schema",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class PartitionSchema(pa.DataFrameModel):
    key: str
    n: int = pa.Field(ge=0)


@env.task(report=True)
async def dask_partition() -> pt.DataFrame[PartitionSchema]:
    pdf = pd.DataFrame({"key": ["a", "b"], "n": [1, 2]})
    return dd.from_pandas(pdf, npartitions=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandera + Dask Flyte example.")
    parser.add_argument(
        "--mode",
        choices=("local", "remote"),
        default="remote",
        help="Run tasks locally or submit to a remote Flyte cluster.",
    )
    args = parser.parse_args()

    flyte.init_from_config(log_level=logging.DEBUG)
    try:
        run = flyte.with_runcontext(args.mode).run(dask_partition)
        run.wait()
        print("dask pandera example OK:", run.outputs()[0])
    except Exception as exc:
        print("Expected if Dask has no Flyte dataframe handler:", type(exc).__name__, exc)
