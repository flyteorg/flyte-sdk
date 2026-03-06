"""Arrow RecordBatch iteration for JsonlFile and JsonlDir."""

from pathlib import Path

import flyte
import nest_asyncio
from flyte._image import PythonWheels

from flyteplugins.jsonl import JsonlDir, JsonlFile

nest_asyncio.apply()

env = flyte.TaskEnvironment(
    name="jsonl-arrow",
    image=(
        flyte.Image.from_debian_base(name="jsonl-arrow")
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-jsonl",
                pre=True,
            )
        )
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent.parent.parent / "dist",
                package_name="flyte",
                pre=True,
            ),
        )
    ).with_pip_packages("nest-asyncio", "pyarrow"),
)


@env.task
async def create_file() -> JsonlFile:
    """Create a single JSONL file with sample data."""
    f = JsonlFile.new_remote("metrics.jsonl")
    async with f.writer() as w:
        for i in range(10_000):
            await w.write(
                {
                    "timestamp": i,
                    "cpu": 50.0 + (i % 30),
                    "memory_mb": 1024 + (i % 512),
                    "host": f"node-{i % 5}",
                }
            )
    return f


@env.task
async def create_dir() -> JsonlDir:
    """Create a sharded JSONL directory with sample data."""
    d = JsonlDir.new_remote("metrics_sharded")
    async with d.writer(max_records_per_shard=2000) as w:
        for i in range(10_000):
            await w.write(
                {
                    "timestamp": i,
                    "cpu": 50.0 + (i % 30),
                    "memory_mb": 1024 + (i % 512),
                    "host": f"node-{i % 5}",
                }
            )
    return d


@env.task
async def arrow_from_file(f: JsonlFile) -> int:
    """Stream a single file as Arrow RecordBatches."""
    import pyarrow as pa

    total_rows = 0
    batches = []

    async for batch in f.iter_arrow_batches(batch_size=4096):
        batches.append(batch)
        total_rows += batch.num_rows

    # Combine into a table for analysis
    table = pa.Table.from_batches(batches)
    print(f"File -> Arrow table: {table.num_rows} rows, {table.num_columns} columns")
    print(f"Schema: {table.schema}")

    # Example: compute mean CPU
    cpu_mean = table.column("cpu").to_pylist()
    avg = sum(cpu_mean) / len(cpu_mean)
    print(f"Average CPU: {avg:.1f}")

    return total_rows


@env.task
async def arrow_from_dir(d: JsonlDir) -> int:
    """Stream a sharded directory as Arrow RecordBatches."""
    import pyarrow as pa

    total_rows = 0
    batches = []

    async for batch in d.iter_arrow_batches(batch_size=4096):
        batches.append(batch)
        total_rows += batch.num_rows

    table = pa.Table.from_batches(batches)
    print(f"Dir -> Arrow table: {table.num_rows} rows, {table.num_columns} columns")

    return total_rows


@env.task
def arrow_from_file_sync(f: JsonlFile) -> int:
    """Sync Arrow batch iteration for a single file."""
    total_rows = 0
    for batch in f.iter_arrow_batches_sync(batch_size=2048):
        total_rows += batch.num_rows
    print(f"Sync file -> {total_rows} rows")
    return total_rows


@env.task
def arrow_from_dir_sync(d: JsonlDir) -> int:
    """Sync Arrow batch iteration across shards."""
    total_rows = 0
    for batch in d.iter_arrow_batches_sync(batch_size=2048):
        total_rows += batch.num_rows
    print(f"Sync dir -> {total_rows} rows")
    return total_rows


@env.task
async def batches_from_dir(d: JsonlDir) -> int:
    """Iterate in list-of-dict batches across shards."""
    total = 0
    async for batch in d.iter_batches(batch_size=500):
        total += len(batch)
        # Each batch is a list[dict]
        print(f"batch of {len(batch)} records (first id={batch[0]['timestamp']})")
    return total


@env.task
def batches_from_dir_sync(d: JsonlDir) -> int:
    """Sync list-of-dict batch iteration across shards."""
    total = 0
    for batch in d.iter_batches_sync(batch_size=500):
        total += len(batch)
    print(f"Sync batches: {total} records total")
    return total


@env.task
async def main() -> None:
    # Create test data
    f = await create_file()
    d = await create_dir()

    # Async Arrow batches
    n = await arrow_from_file(f)
    print(f"Async file arrow: {n} rows")  # 10000

    n = await arrow_from_dir(d)
    print(f"Async dir arrow: {n} rows")  # 10000

    # Sync Arrow batches
    n = await arrow_from_file_sync.aio(f)
    print(f"Sync file arrow: {n} rows")  # 10000

    n = await arrow_from_dir_sync.aio(d)
    print(f"Sync dir arrow: {n} rows")  # 10000

    # List-of-dict batches
    n = await batches_from_dir(d)
    print(f"Async batches: {n} rows")  # 10000

    n = await batches_from_dir_sync.aio(d)
    print(f"Sync batches: {n} rows")  # 10000


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
