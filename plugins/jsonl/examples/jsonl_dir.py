"""JsonlDir: sharded JSONL for large datasets with sync/async, compression and verification."""

from pathlib import Path

import flyte
import nest_asyncio
from flyte._image import PythonWheels

from flyteplugins.jsonl import JsonlDir

nest_asyncio.apply()

_LARGE_RECORD_COUNT = 500_000
_RECORDS_PER_SHARD = 50_000

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
    ).with_pip_packages("nest-asyncio"),
)


@env.task
async def write_sharded() -> JsonlDir:
    """Write 500 records across shards of 100 records each."""
    d = JsonlDir.new_remote("sharded")
    async with d.writer(max_records_per_shard=100) as w:
        for i in range(500):
            await w.write({"id": i, "value": f"record-{i}"})
    return d


@env.task
async def write_compressed() -> JsonlDir:
    """Write zstd-compressed shards."""
    d = JsonlDir.new_remote("compressed")
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
def write_sharded_sync() -> JsonlDir:
    """Write sharded JSONL using the sync API."""
    d = JsonlDir.new_remote("sharded_sync")
    with d.writer_sync(max_records_per_shard=100) as w:
        for i in range(300):
            w.write({"id": i, "sync": True})
    return d


@env.task
def read_all_sync(d: JsonlDir) -> int:
    """Read all records across shards using the sync API."""
    count = 0
    for record in d.iter_records_sync():
        count += 1
    print(f"Sync read {count} records from {d.path}")
    return count


@env.task
async def write_large() -> JsonlDir:
    """Write a large sharded directory (500k records across 10 shards)."""
    d = JsonlDir.new_remote("large")
    async with d.writer(max_records_per_shard=_RECORDS_PER_SHARD) as w:
        for i in range(_LARGE_RECORD_COUNT):
            await w.write(
                {
                    "id": i,
                    "payload": f"data-{i:06d}",
                    "nested": {"x": i * 0.1, "y": i * 0.2},
                }
            )
    return d


@env.task
async def write_large_compressed() -> JsonlDir:
    """Write a large zstd-compressed sharded directory (500k records)."""
    d = JsonlDir.new_remote("large_compressed")
    async with d.writer(
        shard_extension=".jsonl.zst",
        max_records_per_shard=_RECORDS_PER_SHARD,
    ) as w:
        for i in range(_LARGE_RECORD_COUNT):
            await w.write(
                {
                    "id": i,
                    "payload": f"data-{i:06d}",
                    "nested": {"x": i * 0.1, "y": i * 0.2},
                }
            )
    return d


@env.task
async def verify_large(d: JsonlDir, expected: int) -> bool:
    """Read back a large directory and verify record count and cross-shard ordering."""
    count = 0
    prev_id = -1
    async for record in d.iter_records():
        assert record["id"] == prev_id + 1, f"Out-of-order: expected {prev_id + 1}, got {record['id']}"
        prev_id = record["id"]
        count += 1
    assert count == expected, f"Count mismatch: expected {expected}, got {count}"
    print(f"verify_large: {count} records OK, cross-shard ordering verified")
    return True


@env.task
def verify_large_sync(d: JsonlDir, expected: int) -> bool:
    """Sync variant: read back and verify."""
    count = 0
    prev_id = -1
    for record in d.iter_records_sync():
        assert record["id"] == prev_id + 1, f"Out-of-order: expected {prev_id + 1}, got {record['id']}"
        prev_id = record["id"]
        count += 1
    assert count == expected, f"Count mismatch: expected {expected}, got {count}"
    print(f"verify_large_sync: {count} records OK")
    return True


@env.task
async def main() -> None:
    # --- Basic sharded write + read ---
    d = await write_sharded()
    count = await read_all(d)
    print(f"Sharded: {count} records")  # 500

    # --- Append ---
    d = await append_to_existing(d)
    count = await read_all(d)
    print(f"After append: {count} records")  # 700

    # --- Compressed ---
    d_zst = await write_compressed()
    count = await read_all(d_zst)
    print(f"Compressed: {count} records")  # 300

    # --- Sync write + read ---
    d_sync = await write_sharded_sync.aio()
    count = await read_all_sync.aio(d_sync)
    print(f"Sync sharded: {count} records")  # 300

    # --- Large directory correctness (plain) ---
    d_large = await write_large()
    await verify_large(d_large, _LARGE_RECORD_COUNT)
    await verify_large_sync.aio(d_large, _LARGE_RECORD_COUNT)

    # --- Large directory correctness (compressed) ---
    d_large_zst = await write_large_compressed()
    await verify_large(d_large_zst, _LARGE_RECORD_COUNT)

    print("All tests passed!")


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
