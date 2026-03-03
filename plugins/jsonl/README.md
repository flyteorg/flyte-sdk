# JSONL

JSONL (JSON Lines) file and directory types for Flyte, backed by `orjson` for
fast serialization and optional `zstd` compression.

```bash
pip install flyteplugins-jsonl

# For Arrow RecordBatch support
pip install 'flyteplugins-jsonl[arrow]'
```

## Types

### `JsonlFile`

A single JSONL file. Inherits from `flyte.io.File` so it works with remote
storage, upload/download and the Flyte type engine out of the box.

```python
from flyteplugins.jsonl import JsonlFile

# Async read
@env.task
async def process(f: JsonlFile):
    async for record in f.iter_records():
        print(record)

# Async write
@env.task
async def create() -> JsonlFile:
    f = JsonlFile.new_remote("data.jsonl")
    async with f.writer() as w:
        await w.write({"key": "value"})
    return f

# Sync write
@env.task
async def create_sync() -> JsonlFile:
    f = JsonlFile.new_remote("data.jsonl")
    with f.writer_sync() as w:
        w.write({"key": "value"})
    return f
```

### `JsonlDir`

A directory of sharded JSONL files (`part-00000.jsonl`, `part-00001.jsonl`,
etc.). Inherits from `flyte.io.Dir`. Supports automatic shard rotation on write
and transparent cross-shard iteration on read.

```python
from flyteplugins.jsonl import JsonlDir

# Write with automatic sharding
@env.task
async def create() -> JsonlDir:
    d = JsonlDir.new_remote("output_shards")
    async with d.writer(max_records_per_shard=10_000) as w:
        for i in range(50_000):
            await w.write({"id": i})
    return d

# Read across all shards
@env.task
async def process(d: JsonlDir):
    async for record in d.iter_records():
        print(record)
```

## Features

### Compression

Both types support zstd compression transparently via file extension. Use
`.jsonl.zst` to enable:

```python
# Single file
f = JsonlFile.new_remote("data.jsonl.zst")

# Sharded directory
async with d.writer(shard_extension=".jsonl.zst") as w:
    await w.write({"compressed": True})
```

### Prefetch (JsonlDir)

When iterating over a sharded directory, the next shard is prefetched in the
background to overlap network I/O with processing. This is enabled by default
and can be tuned or disabled:

```python
async for record in d.iter_records(prefetch=True, queue_size=8192):
    process(record)
```

`queue_size` is the memory safety bound on the read-ahead buffer.

### Batch iteration

Both types support batched iteration for bulk processing:

```python
# List-of-dicts batches
async for batch in d.iter_batches(batch_size=1000):
    process_batch(batch)  # list[dict]

# Arrow RecordBatches (requires pyarrow)
async for batch in d.iter_arrow_batches(batch_size=65536):
    table = pa.Table.from_batches([batch])
```

Sync variants are available: `iter_batches_sync()`, `iter_arrow_batches_sync()`.

### Error handling

All read methods accept an `on_error` parameter:

- `"raise"` (default) -- propagate parse errors immediately
- `"skip"` -- log a warning and skip corrupt lines
- A callable `(line_number: int, raw_line: bytes, exception: Exception) -> None`
  for custom handling

```python
async for record in f.iter_records(on_error="skip"):
    print(record)
```

### Shard rotation

The directory writer rotates shards based on record count, byte size or both:

```python
async with d.writer(
    max_records_per_shard=10_000,       # rotate after 10k records
    max_bytes_per_shard=256 << 20,      # or after 256 MB (default)
) as w:
    ...
```

### Append

Opening a writer on a directory that already contains shards is safe -- the
writer scans for existing `part-NNNNN` files and starts from the next index.

## Sync vs Async

Every read/write method has both async and sync variants:

| Async                      | Sync                            |
| -------------------------- | ------------------------------- |
| `iter_records()`           | `iter_records_sync()`           |
| `iter_batches()`           | `iter_batches_sync()`           |
| `iter_arrow_batches()`     | `iter_arrow_batches_sync()`     |
| `writer()`                 | `writer_sync()`                 |

## Examples

See [`examples/`](examples/) for runnable scripts:

- `jsonl_file.py` -- single-file read/write with compression and error handling
- `jsonl_dir.py` -- sharded directory read/write, append, and compression
- `jsonl_arrow.py` -- Arrow RecordBatch iteration for analytics workloads
