"""JsonlFile: single-file JSONL read/write with compression, error handling and large files."""

from pathlib import Path

import flyte
import nest_asyncio
from flyte._image import PythonWheels

from flyteplugins.jsonl import JsonlFile

nest_asyncio.apply()

env = flyte.TaskEnvironment(
    name="jsonl-file",
    image=(
        flyte.Image.from_debian_base(name="jsonl-file")
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

_LARGE_RECORD_COUNT = 500_000


@env.task
async def write_plain() -> JsonlFile:
    """Write a plain JSONL file."""
    f = JsonlFile.new_remote("data.jsonl")
    async with f.writer() as w:
        for i in range(100):
            await w.write({"id": i, "value": f"item-{i}"})
    return f


@env.task
async def write_compressed() -> JsonlFile:
    """Write a zstd-compressed JSONL file."""
    f = JsonlFile.new_remote("data.jsonl.zst")
    async with f.writer(compression_level=3) as w:
        for i in range(100):
            await w.write({"id": i, "compressed": True})
    return f


@env.task
async def read_async(f: JsonlFile) -> int:
    """Read all records asynchronously, return count."""
    count = 0
    async for record in f.iter_records():
        count += 1
    print(f"Read {count} records from {f.path}")
    return count


@env.task
def write_sync() -> JsonlFile:
    """Write using the sync API."""
    f = JsonlFile.new_remote("sync.jsonl")
    with f.writer_sync() as w:
        for i in range(50):
            w.write({"id": i, "sync": True})
    return f


@env.task
def read_sync(f: JsonlFile) -> int:
    """Read all records using the sync API."""
    count = 0
    for record in f.iter_records_sync():
        count += 1
    print(f"Sync read {count} records from {f.path}")
    return count


@env.task
async def write_bulk() -> JsonlFile:
    """Write many records at once using write_many."""
    f = JsonlFile.new_remote("bulk.jsonl")
    records = [{"id": i, "bulk": True} for i in range(200)]
    async with f.writer() as w:
        await w.write_many(records)
    return f


@env.task
async def write_corrupted() -> JsonlFile:
    """Write a JSONL file with some corrupt lines mixed in."""
    f = JsonlFile.new_remote("corrupted.jsonl")
    async with f.open("wb") as fh:
        # 3 valid records
        await fh.write(b'{"id": 0, "ok": true}\n')
        await fh.write(b'{"id": 1, "ok": true}\n')
        await fh.write(b'{"id": 2, "ok": true}\n')
        # 2 corrupt lines
        await fh.write(b"this is not json\n")
        await fh.write(b'{"truncated": tru\n')
        # 2 more valid records
        await fh.write(b'{"id": 3, "ok": true}\n')
        await fh.write(b'{"id": 4, "ok": true}\n')
    return f


@env.task
async def read_with_skip(f: JsonlFile) -> int:
    """Read records, skipping corrupt lines (logs warnings)."""
    count = 0
    async for record in f.iter_records(on_error="skip"):
        count += 1
    print(f"read_with_skip: got {count} valid records (expected 5)")
    return count


@env.task
async def read_with_custom_handler(f: JsonlFile) -> int:
    """Read records with a custom error handler that collects errors."""
    errors: list[dict] = []

    def on_error(line_number: int, raw_line: bytes, exc: Exception) -> None:
        errors.append({"line": line_number, "error": str(exc)})

    count = 0
    async for record in f.iter_records(on_error=on_error):
        count += 1

    print(f"read_with_custom_handler: {count} valid, {len(errors)} errors")
    for e in errors:
        print(f"  line {e['line']}: {e['error']}")
    return count


@env.task
async def write_large() -> JsonlFile:
    """Write a large JSONL file (500k records, ~50 MB)."""
    f = JsonlFile.new_remote("large.jsonl")
    async with f.writer() as w:
        for i in range(_LARGE_RECORD_COUNT):
            await w.write(
                {
                    "id": i,
                    "payload": f"data-{i:06d}",
                    "nested": {"x": i * 0.1, "y": i * 0.2},
                }
            )
    return f


@env.task
async def write_large_compressed() -> JsonlFile:
    """Write a large zstd-compressed JSONL file (500k records)."""
    f = JsonlFile.new_remote("large.jsonl.zst")
    async with f.writer(compression_level=3) as w:
        for i in range(_LARGE_RECORD_COUNT):
            await w.write(
                {
                    "id": i,
                    "payload": f"data-{i:06d}",
                    "nested": {"x": i * 0.1, "y": i * 0.2},
                }
            )
    return f


@env.task
async def verify_large(f: JsonlFile, expected: int) -> bool:
    """Read back a large file and verify record count and ordering."""
    count = 0
    prev_id = -1
    async for record in f.iter_records():
        assert record["id"] == prev_id + 1, f"Out-of-order: expected {prev_id + 1}, got {record['id']}"
        prev_id = record["id"]
        count += 1
    assert count == expected, f"Count mismatch: expected {expected}, got {count}"
    print(f"verify_large: {count} records OK, ordering verified")
    return True


@env.task
def verify_large_sync(f: JsonlFile, expected: int) -> bool:
    """Sync variant: read back a large file and verify."""
    count = 0
    prev_id = -1
    for record in f.iter_records_sync():
        assert record["id"] == prev_id + 1, f"Out-of-order: expected {prev_id + 1}, got {record['id']}"
        prev_id = record["id"]
        count += 1
    assert count == expected, f"Count mismatch: expected {expected}, got {count}"
    print(f"verify_large_sync: {count} records OK")
    return True


@env.task
async def main() -> None:
    # --- Basic read/write ---
    f = await write_plain()
    count = await read_async(f)
    print(f"Plain: {count} records")  # 100

    f_zst = await write_compressed()
    count = await read_async(f_zst)
    print(f"Compressed: {count} records")  # 100

    # --- Sync read/write ---
    f_sync = await write_sync.aio()
    count = await read_sync.aio(f_sync)
    print(f"Sync: {count} records")  # 50

    # --- Bulk write ---
    f_bulk = await write_bulk()
    count = await read_async(f_bulk)
    print(f"Bulk: {count} records")  # 200

    # --- Corrupted file + error handling ---
    f_bad = await write_corrupted()

    count = await read_with_skip(f_bad)
    print(f"Skip mode: {count} valid records")  # 5

    count = await read_with_custom_handler(f_bad)
    print(f"Custom handler: {count} valid records")  # 5

    # --- Large file correctness ---
    f_large = await write_large()
    await verify_large(f_large, _LARGE_RECORD_COUNT)

    f_large_zst = await write_large_compressed()
    await verify_large(f_large_zst, _LARGE_RECORD_COUNT)

    # Sync verification of large file
    await verify_large_sync.aio(f_large, _LARGE_RECORD_COUNT)

    print("All tests passed!")


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
