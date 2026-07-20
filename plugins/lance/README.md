# Lance Plugin

This plugin adds a "lance" format to the Flyte DataFrame, so a Lance dataset can
be passed between tasks as a typed `flyte.io.DataFrame`.

Lance is a columnar, multimodal, streaming-optimized format. Its central property
is that a dataset is opened lazily and streamed on demand — sequentially for a
scan or by random access for shuffled training — without materializing the whole
thing in memory. This plugin preserves that: the primary decoder hands back a live
`lance.LanceDataset` handle you can stream from, not a materialized table.

The plugin registers:

- `lance.LanceDataset` as the default in-memory type for the "lance" format —
  encoded by copying the dataset to Flyte-managed storage, decoded lazily via
  `lance.dataset(uri)`. This is the streaming path.
- `pyarrow.Table` for the "lance" format, for handing off an in-memory table.
  Because `pyarrow.Table` already defaults to Parquet, this is the one case where
  you opt into Lance explicitly, with `Annotated[DataFrame, "lance"]`. Encoded with
  `lance.write_dataset` and decoded eagerly with `dataset.to_table()`, which
  materializes the whole dataset — prefer `lance.LanceDataset` for large or
  multimodal data.

Object-store credentials are threaded through Lance's `storage_options` from
Flyte's storage configuration, so remote reads and writes go through the same
credentials as the rest of Flyte.

To install the plugin, run the following command:

```bash
pip install flyteplugins-lance
```

Usage:

```python
import tempfile

import flyte
import lance
import pyarrow as pa

# Installing the plugin in the task image is all that is needed. Flyte discovers
# it through the flyte.plugins.types entry point and registers the "lance" format
# automatically, so there is nothing to import in your task code.
env = flyte.TaskEnvironment(
    name="lance-example",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-lance"),
)


@env.task
async def make() -> lance.LanceDataset:
    uri = f"{tempfile.mkdtemp()}/example.lance"
    lance.write_dataset(pa.table({"id": [1, 2, 3]}), uri)
    return lance.dataset(uri)  # encoded as "lance" — the default format for a LanceDataset


@env.task
async def consume(ds: lance.LanceDataset) -> int:
    return ds.count_rows()  # a live, streaming handle — no wrapper, no .open()


@env.task
async def main() -> int:
    return await consume(await make())
```
