"""Example 4 — "convert once, stream forever" with multimodal rows.

This mirrors a real training-data pipeline: each row carries image bytes
(``large_binary``) alongside structured labels, all in one Lance dataset. The
convert task writes it in chunks and hands it off by reference; the train task
streams shuffled batches by random access — the access pattern SGD needs — without
ever materializing the dataset.

This is exactly why you keep it a ``lance.LanceDataset`` (streaming) rather than
decoding to a ``pyarrow.Table``: a table would pull every image into memory, which
defeats the point on a dataset larger than RAM.

Run:  python examples/04_multimodal_streaming.py
"""

import io
import os
import random
import tempfile

import flyte
import lance
import pyarrow as pa
from flyte.io import DataFrame

SCHEMA = pa.schema(
    [
        ("id", pa.int32()),
        ("image", pa.large_binary()),  # stand-in for encoded PNG/JPEG bytes
        ("label", pa.int32()),
    ]
)

env = flyte.TaskEnvironment(
    name="lance-ex-multimodal",
    image=flyte.Image.from_debian_base(name="lance-examples").with_local_v2_plugins("flyteplugins-lance"),
)


def _fake_image_bytes(i: int) -> bytes:
    # Pretend PNG bytes of varying size, so rows are non-uniform like real images.
    return (f"IMG{i}".encode()) * (i % 7 + 1)


@env.task
async def convert(n: int = 2000, chunk: int = 512) -> DataFrame:
    """Fold multimodal samples into one Lance dataset, chunk by chunk, and hand it
    off by reference (see example 3 for why DataFrame, not lance.LanceDataset)."""
    uri = os.path.join(tempfile.mkdtemp(), "images.lance")
    mode = "create"
    for start in range(0, n, chunk):
        rows = list(range(start, min(start + chunk, n)))
        table = pa.table(
            {
                "id": rows,
                "image": [_fake_image_bytes(i) for i in rows],
                "label": [i % 10 for i in rows],
            },
            schema=SCHEMA,
        )
        lance.write_dataset(table, uri, mode=mode)
        mode = "append"
    return DataFrame(uri=uri, format="lance")


@env.task
async def train_one_epoch(ds: lance.LanceDataset, batch_size: int = 128, seed: int = 0) -> dict:
    """Stream one epoch of shuffled batches by random access — no download, no
    full materialization. Only the requested columns/rows are read per batch.
    """
    order = list(range(ds.count_rows()))
    random.Random(seed).shuffle(order)

    seen = 0
    label_hist: dict[int, int] = {}
    for i in range(0, len(order), batch_size):
        batch = ds.take(order[i : i + batch_size], columns=["image", "label"])
        for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            _ = io.BytesIO(img)  # stand-in for decoding the image
            label_hist[label] = label_hist.get(label, 0) + 1
            seen += 1
    # String keys: this dict is stored as MessagePack, and the UI renders it back
    # only when the map keys are strings (integer keys show up as base64).
    return {"rows_streamed": seen, "labels": {str(label): count for label, count in sorted(label_hist.items())}}


@env.task
async def main(n: int = 2000) -> dict:
    dataset = await convert(n)
    return await train_one_epoch(dataset)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, n=2000)
    print(f"Run URL: {run.url}")
