"""JsonlFile: single-file JSONL read/write with compression and error handling."""

import flyte
from flyteplugins.jsonl import JsonlFile
from flyte._image import PythonWheels
from pathlib import Path

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
    ),
)


# --- Async write + read ---


@env.task
async def write_plain(base_dir: str) -> JsonlFile:
    """Write a plain JSONL file."""
    f = JsonlFile.new_remote(f"{base_dir}/data.jsonl")
    async with f.writer() as w:
        for i in range(100):
            await w.write({"id": i, "value": f"item-{i}"})
    return f


@env.task
async def write_compressed(base_dir: str) -> JsonlFile:
    """Write a zstd-compressed JSONL file."""
    f = JsonlFile.new_remote(f"{base_dir}/data.jsonl.zst")
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


# --- Sync write + read ---


@env.task
async def write_sync(base_dir: str) -> JsonlFile:
    """Write using the sync API (useful in non-async contexts)."""
    f = JsonlFile.new_remote(f"{base_dir}/sync.jsonl")
    with f.writer_sync() as w:
        for i in range(50):
            w.write({"id": i, "sync": True})
    return f


@env.task
async def read_sync(f: JsonlFile) -> int:
    """Read all records using the sync API."""
    count = 0
    for record in f.iter_records_sync():
        count += 1
    print(f"Sync read {count} records from {f.path}")
    return count


# --- Bulk write ---


@env.task
async def write_bulk(base_dir: str) -> JsonlFile:
    """Write many records at once using write_many."""
    f = JsonlFile.new_remote(f"{base_dir}/bulk.jsonl")
    records = [{"id": i, "bulk": True} for i in range(200)]
    async with f.writer() as w:
        await w.write_many(records)
    return f


# --- Error handling ---


@env.task
async def read_with_skip(f: JsonlFile) -> int:
    """Read records, skipping corrupt lines."""
    count = 0
    async for record in f.iter_records(on_error="skip"):
        count += 1
    return count


@env.task
async def read_with_custom_handler(f: JsonlFile) -> int:
    """Read records with a custom error handler."""
    errors = []

    def on_error(line_number: int, raw_line: bytes, exc: Exception) -> None:
        errors.append({"line": line_number, "error": str(exc)})

    count = 0
    async for record in f.iter_records(on_error=on_error):
        count += 1

    if errors:
        print(f"Encountered {len(errors)} errors: {errors}")
    return count


# --- Main ---


@env.task
async def main() -> None:
    base = flyte.ctx().run_base_dir

    # Plain write + read
    f = await write_plain(f"{base}/plain")
    count = await read_async(f)
    print(f"Plain: {count} records")  # 100

    # Compressed write + read
    f_zst = await write_compressed(f"{base}/compressed")
    count = await read_async(f_zst)
    print(f"Compressed: {count} records")  # 100

    # Sync write + read
    f_sync = await write_sync(f"{base}/sync")
    count = await read_sync(f_sync)
    print(f"Sync: {count} records")  # 50

    # Bulk write + read
    f_bulk = await write_bulk(f"{base}/bulk")
    count = await read_async(f_bulk)
    print(f"Bulk: {count} records")  # 200


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
