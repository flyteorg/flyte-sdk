"""
Pandera + Flyte: Ibis (`pandera.typing.ibis.Table`).

Uses `pandera.ibis.DataFrameModel` with an `ibis.memtable` table.

.. note::
    Flyte core does not ship a structured-dataset encoder for ``ibis.Table``.
    This example shows the pandera typing + validation pattern; for remote
    execution you may need a custom :class:`flyte.io.DataFrameEncoder` /
    ``DataFrameDecoder`` pair or another interchange format.
"""

from __future__ import annotations

import argparse
import logging

import ibis
import pandera.typing.ibis as pt
from pandera.ibis import DataFrameModel

import flyte

img = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyte>=2.0.9",
    "flyteplugins-pandera",
    "pandera[ibis]",
    pre=True,
)

env = flyte.TaskEnvironment(
    "pandera_ibis_schema",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class SalesSchema(DataFrameModel):
    """Sales rows."""

    region: str
    units: int


@env.task(report=True)
async def build_sales() -> pt.Table[SalesSchema]:
    return ibis.memtable(
        {
            "region": ["east", "west"],
            "units": [10, 20],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandera + Ibis Flyte example.")
    parser.add_argument(
        "--mode",
        choices=("local", "remote"),
        default="remote",
        help="Run tasks locally or submit to a remote Flyte cluster.",
    )
    args = parser.parse_args()

    flyte.init_from_config(log_level=logging.DEBUG)
    try:
        run = flyte.with_runcontext(args.mode).run(build_sales)
        run.wait()
        print("ibis pandera example OK:", run.outputs()[0])
    except Exception as exc:
        print(
            "Expected if ibis.Table has no Flyte dataframe handler:",
            type(exc).__name__,
            exc,
        )
