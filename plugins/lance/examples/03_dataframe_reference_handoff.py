"""Example 3 — hand off a Lance dataset you already wrote, by reference.

When a task has already written a Lance dataset (often large, and written in
chunks), the cheapest handoff is ``DataFrame(uri=..., format="lance")``: Flyte
uploads the ``.lance`` directory as-is, with no re-encoding.

Returning a ``lance.LanceDataset`` instead would run the encoder, which re-reads
and rewrites every fragment through Arrow — wasteful for a big dataset. The
consumer can still accept a raw ``lance.LanceDataset`` regardless of which form
the producer used; Flyte decodes the "lance" literal into a handle at the boundary.

Run:  python examples/03_dataframe_reference_handoff.py
"""

import os
import tempfile

import flyte
import lance
import pyarrow as pa
from flyte.io import DataFrame

env = flyte.TaskEnvironment(
    name="lance-ex-reference",
    image=flyte.Image.from_debian_base(name="lance-examples").with_local_v2_plugins("flyteplugins-lance"),
)


@env.task
async def convert(n: int = 1000, chunk: int = 500) -> DataFrame:
    """Write a Lance dataset in chunks (memory-bounded), then hand it off by
    reference so Flyte moves the bytes without re-encoding them."""
    uri = os.path.join(tempfile.mkdtemp(), "dataset.lance")
    mode = "create"
    for start in range(0, n, chunk):
        rows = list(range(start, min(start + chunk, n)))
        lance.write_dataset(pa.table({"id": rows}), uri, mode=mode)
        mode = "append"
    return DataFrame(uri=uri, format="lance")


@env.task
async def inspect(ds: lance.LanceDataset) -> dict:
    """Consumer takes the raw lance type even though the producer returned a
    DataFrame — the cross-type handoff is resolved by the "lance" format."""
    return {"rows": ds.count_rows(), "fragments": len(ds.get_fragments())}


@env.task
async def main(n: int = 1000) -> dict:
    return await inspect(await convert(n))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, n=1000)
    print(f"Run URL: {run.url}")
