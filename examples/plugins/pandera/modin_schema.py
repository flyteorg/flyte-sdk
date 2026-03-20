"""
Pandera + Flyte: Modin (`pandera.typing.modin.DataFrame`).

Modin uses the pandas :class:`~pandera.pandas.DataFrameModel` with ``modin.pandas``.

.. note::
    Flyte does not ship a Modin dataframe encoder in this repo. This script
    documents the pandera + Flyte type annotation; register Modin handlers or
    use an interchange type for remote I/O.
"""

from __future__ import annotations

import argparse
import logging

import modin.pandas as mpd
import pandera.pandas as pa
import pandera.typing.modin as pt

import flyte

img = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages(
        "flyte>=2.0.9",
        # "flyteplugins-pandera",
        "modin",
        "pandera[pandas]",
        pre=True,
    )
    .with_local_v2_plugins("flyteplugins-pandera")
)

env = flyte.TaskEnvironment(
    "pandera_modin_schema",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class RowSchema(pa.DataFrameModel):
    k: str
    v: int = pa.Field(ge=0)


@env.task(report=True)
async def modin_rows() -> pt.DataFrame[RowSchema]:
    return mpd.DataFrame({"k": ["x", "y"], "v": [1, 2]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandera + Modin Flyte example.")
    parser.add_argument(
        "--mode",
        choices=("local", "remote"),
        default="remote",
        help="Run tasks locally or submit to a remote Flyte cluster.",
    )
    args = parser.parse_args()

    flyte.init_from_config(log_level=logging.DEBUG)
    try:
        run = flyte.with_runcontext(args.mode).run(modin_rows)
        run.wait()
        print("modin pandera example OK:", run.outputs()[0])
    except Exception as exc:
        print("Expected if Modin has no Flyte dataframe handler:", type(exc).__name__, exc)
