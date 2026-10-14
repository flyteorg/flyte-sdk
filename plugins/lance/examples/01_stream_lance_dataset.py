"""Example 1 — pass a lance.LanceDataset between tasks and stream it.

The plugin registers ``lance.LanceDataset`` as the default in-memory type for the
"lance" format, so one task can return a ``lance.LanceDataset`` and another can
accept one directly — no ``DataFrame`` wrapper, no ``.open()``. The consumer
receives a live, lazily-opened handle it streams from (sequential scan and
random-access take), never materializing the whole dataset.

Run:
    python examples/01_stream_lance_dataset.py
"""

import tempfile

import flyte
import lance
import pyarrow as pa

env = flyte.TaskEnvironment(
    name="lance-ex-streaming",
    image=flyte.Image.from_debian_base(name="lance-examples").with_local_v2_plugins("flyteplugins-lance"),
)


@env.task
async def make_dataset(n: int = 1000) -> lance.LanceDataset:
    """Build a Lance dataset and hand it off. Returning a lance.LanceDataset encodes
    it as the "lance" format automatically (it is the default for that type)."""
    uri = f"{tempfile.mkdtemp()}/points.lance"
    table = pa.table({"id": list(range(n)), "value": [i * i for i in range(n)]})
    lance.write_dataset(table, uri)
    return lance.dataset(uri)


@env.task
async def summarize(ds: lance.LanceDataset) -> dict:
    """`ds` arrives already open. Stream it two ways without materializing it:
    a sequential columnar scan, and a random-access take of specific rows."""
    total = 0
    for batch in ds.scanner(columns=["value"], batch_size=256).to_batches():
        total += sum(batch.column("value").to_pylist())
    sample = ds.take([0, 42, 999], columns=["id", "value"]).to_pylist()
    return {"rows": ds.count_rows(), "sum_of_values": total, "sample": sample}


@env.task
async def main(n: int = 1000) -> dict:
    ds = await make_dataset(n)
    return await summarize(ds)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, n=1000)
    print(f"Run URL: {run.url}")
