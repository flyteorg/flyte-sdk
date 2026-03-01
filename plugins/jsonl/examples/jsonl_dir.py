"""JsonlDir: sharded JSONL for large datasets."""

import flyte
from flyteplugins.jsonl import JsonlDir
from flyte._image import PythonWheels
from pathlib import Path

env = flyte.TaskEnvironment(
    name="jsonl-dir",
    image=(
        flyte.Image.from_debian_base(name="jsonl-dir")
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
    ),
)


@env.task
async def write_sharded(base_dir: str) -> JsonlDir:
    """Write 500 records across shards of 100 records each."""
    d = JsonlDir.from_existing_remote(base_dir)
    async with d.writer(max_records_per_shard=100) as w:
        for i in range(500):
            await w.write({"id": i, "value": f"record-{i}"})
    return d


@env.task
async def write_compressed(base_dir: str) -> JsonlDir:
    """Write zstd-compressed shards."""
    d = JsonlDir.from_existing_remote(base_dir)
    async with d.writer(
        shard_extension=".jsonl.zst",
        max_records_per_shard=100,
    ) as w:
        for i in range(300):
            await w.write({"id": i, "compressed": True})
    return d


@env.task
async def read_all(d: JsonlDir) -> int:
    """Read all records across shards, return total count."""
    count = 0
    async for record in d.iter_records():
        print(f"Read record {record}")
        count += 1
    print(f"Read {count} records from {d.path}")
    return count


@env.task
async def read_with_error_handling(d: JsonlDir) -> int:
    """Read records, skipping any corrupt lines."""
    count = 0
    async for record in d.iter_records(on_error="skip"):
        count += 1
    return count


@env.task
async def append_to_existing(d: JsonlDir) -> JsonlDir:
    """Append more records to a directory that already has shards.

    The writer detects existing shard indices and continues from the next one.
    """
    async with d.writer(max_records_per_shard=100) as w:
        for i in range(200):
            await w.write({"id": i, "appended": True})
    return d


@env.task
async def main() -> None:
    base = flyte.ctx().run_base_dir

    # Write 500 records across 5 shards (100 each)
    d = await write_sharded(f"{base}/sharded")
    count = await read_all(d)
    print(f"Sharded: {count} records")  # 500

    # Append 200 more records (shards part-00005 and part-00006)
    d = await append_to_existing(d)
    count = await read_all(d)
    print(f"After append: {count} records")  # 700

    # Write compressed shards
    d_zst = await write_compressed(f"{base}/compressed")
    count = await read_all(d_zst)
    print(f"Compressed: {count} records")  # 300


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
