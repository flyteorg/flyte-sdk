"""Example 2 — a pyarrow.Table as "lance", plus eager reads and column subsetting.

``pyarrow.Table`` already defaults to Parquet, so this is the one place you opt
into Lance explicitly, with ``Annotated[DataFrame, "lance"]``. The same stored
data can then be read back two ways:

- as a streaming ``lance.LanceDataset`` (lazy), or
- eagerly as a ``pyarrow.Table`` — optionally subsetting columns with an
  ``Annotated[pa.Table, OrderedDict(...)]`` annotation on the parameter.

Eager decode materializes the whole dataset, so prefer ``lance.LanceDataset`` for
large or multimodal data (see example 4).

Run: python examples/02_arrow_table_and_annotation.py
"""

from collections import OrderedDict
from typing import Annotated

import flyte
import lance
import pyarrow as pa
from flyte.io import DataFrame

env = flyte.TaskEnvironment(
    name="lance-ex-arrow",
    image=flyte.Image.from_debian_base(name="lance-examples").with_local_v2_plugins("flyteplugins-lance"),
)


@env.task
async def make_table() -> Annotated[DataFrame, "lance"]:
    """Hand off an in-memory Arrow table stored as Lance. The annotation selects
    the "lance" encoder; a bare ``DataFrame``/``pa.Table`` would default to Parquet."""
    table = pa.table(
        {
            "city": ["NYC", "SF", "LA", "SEA"],
            "temp_c": [7, 15, 20, 11],
            "humidity": [55, 70, 40, 80],
        }
    )
    return DataFrame.from_df(table)


@env.task
async def read_streaming(ds: lance.LanceDataset) -> int:
    """The same "lance" data, opened as a streaming handle."""
    return ds.count_rows()


@env.task
async def read_eager(table: Annotated[pa.Table, OrderedDict(city=str, temp_c=int)]) -> dict:
    """Read eagerly as a pyarrow.Table, subset to two columns via the annotation."""
    return {"columns": table.column_names, "rows": table.num_rows}


@env.task
async def main() -> dict:
    df = await make_table()
    streamed = await read_streaming(df)  # DataFrame -> lance.LanceDataset
    eager = await read_eager(df)  # DataFrame -> pa.Table (column subset)
    return {"streaming_rows": streamed, "eager": eager}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(f"Run URL: {run.url}")
