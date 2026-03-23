"""Tests for JsonlDir, sharded writers, and helpers."""

from __future__ import annotations

import os

import orjson
import pytest

from flyteplugins.jsonl._jsonl_dir import (
    JsonlDir,
    _is_jsonl_path,
    _parse_shard_index,
    _shard_name,
)
from flyteplugins.jsonl._jsonl_file import JsonlFile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("part-00000.jsonl", True),
        ("data.jsonl.zst", True),
        ("data.jsonl.zstd", True),
        ("data.json", False),
        ("data.txt", False),
    ],
)
def test_is_jsonl_path(path, expected):
    assert _is_jsonl_path(path) is expected


@pytest.mark.parametrize(
    "index,ext,expected",
    [
        (0, ".jsonl", "part-00000.jsonl"),
        (42, ".jsonl.zst", "part-00042.jsonl.zst"),
        (99999, ".jsonl", "part-99999.jsonl"),
    ],
)
def test_shard_name(index, ext, expected):
    assert _shard_name(index, ext) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("part-00000.jsonl", 0),
        ("part-00042.jsonl.zst", 42),
        ("/some/dir/part-00010.jsonl.zstd", 10),
        ("not-a-shard.jsonl", None),
        ("readme.txt", None),
    ],
)
def test_parse_shard_index(path, expected):
    assert _parse_shard_index(path) == expected


def test_next_index():
    shards = [
        JsonlFile(path="/d/part-00000.jsonl"),
        JsonlFile(path="/d/part-00003.jsonl"),
    ]
    assert JsonlDir._next_index(shards) == 4

    assert JsonlDir._next_index([]) == 0


# ---------------------------------------------------------------------------
# Async write / read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_single_shard(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(20)]
    async with d.writer() as w:
        for r in records:
            await w.write(r)

    result = [r async for r in d.iter_records()]
    assert result == records

    # Only one shard should exist
    shards = [f for f in os.listdir(tmp_path) if f.endswith(".jsonl")]
    assert len(shards) == 1


@pytest.mark.asyncio
async def test_write_rotates_by_record_count(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(25)]
    async with d.writer(max_records_per_shard=10) as w:
        for r in records:
            await w.write(r)

    shards = sorted(f for f in os.listdir(tmp_path) if f.endswith(".jsonl"))
    assert len(shards) == 3  # 10 + 10 + 5

    result = [r async for r in d.iter_records()]
    assert result == records


@pytest.mark.asyncio
async def test_write_compressed_shards(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(20)]
    async with d.writer(shard_extension=".jsonl.zst") as w:
        for r in records:
            await w.write(r)

    shards = [f for f in os.listdir(tmp_path) if f.endswith(".jsonl.zst")]
    assert len(shards) == 1

    result = [r async for r in d.iter_records()]
    assert result == records


@pytest.mark.asyncio
async def test_iter_records_no_prefetch(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(30)]
    async with d.writer(max_records_per_shard=10) as w:
        for r in records:
            await w.write(r)

    result = [r async for r in d.iter_records(prefetch=False)]
    assert result == records


@pytest.mark.asyncio
async def test_iter_records_with_prefetch(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(30)]
    async with d.writer(max_records_per_shard=10) as w:
        for r in records:
            await w.write(r)

    result = [r async for r in d.iter_records(prefetch=True)]
    assert result == records


@pytest.mark.asyncio
async def test_iter_records_empty_dir(tmp_path):
    d = JsonlDir(path=str(tmp_path))
    result = [r async for r in d.iter_records()]
    assert result == []


@pytest.mark.asyncio
async def test_read_user_created_shards(tmp_path):
    """JsonlDir can read .jsonl files the user wrote manually (not via the plugin)."""
    # Write files with non-standard names — any .jsonl file should be picked up
    (tmp_path / "events-2024-01.jsonl").write_bytes(
        b"".join(orjson.dumps({"month": 1, "id": i}) + b"\n" for i in range(5))
    )
    (tmp_path / "events-2024-02.jsonl").write_bytes(
        b"".join(orjson.dumps({"month": 2, "id": i}) + b"\n" for i in range(3))
    )
    # A non-jsonl file should be ignored
    (tmp_path / "readme.txt").write_text("ignore me")

    d = JsonlDir(path=str(tmp_path))
    result = [r async for r in d.iter_records()]
    assert len(result) == 8


def test_read_user_created_shards_sync(tmp_path):
    """Sync variant: reads user-created .jsonl files."""
    (tmp_path / "batch_a.jsonl").write_bytes(b"".join(orjson.dumps({"src": "a", "i": i}) + b"\n" for i in range(4)))
    (tmp_path / "batch_b.jsonl").write_bytes(b"".join(orjson.dumps({"src": "b", "i": i}) + b"\n" for i in range(6)))

    d = JsonlDir(path=str(tmp_path))
    result = list(d.iter_records_sync())
    assert len(result) == 10


@pytest.mark.asyncio
async def test_ordering_across_shards(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"seq": i} for i in range(50)]
    async with d.writer(max_records_per_shard=7) as w:
        for r in records:
            await w.write(r)

    result = [r async for r in d.iter_records()]
    assert [r["seq"] for r in result] == list(range(50))


# ---------------------------------------------------------------------------
# Sync write / read
# ---------------------------------------------------------------------------


def test_write_and_read_sync(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(20)]
    with d.writer_sync() as w:
        for r in records:
            w.write(r)

    result = list(d.iter_records_sync())
    assert result == records


def test_write_sync_rotation(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(25)]
    with d.writer_sync(max_records_per_shard=10) as w:
        for r in records:
            w.write(r)

    shards = sorted(f for f in os.listdir(tmp_path) if f.endswith(".jsonl"))
    assert len(shards) == 3

    result = list(d.iter_records_sync())
    assert result == records


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_to_existing(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    # First write
    async with d.writer(max_records_per_shard=10) as w:
        for i in range(15):
            await w.write({"id": i})

    shards_before = sorted(f for f in os.listdir(tmp_path) if f.endswith(".jsonl"))
    assert len(shards_before) == 2  # 10 + 5

    # Second write — should append starting at part-00002
    async with d.writer(max_records_per_shard=10) as w:
        for i in range(15, 25):
            await w.write({"id": i})

    shards_after = sorted(f for f in os.listdir(tmp_path) if f.endswith(".jsonl"))
    assert len(shards_after) == 3  # 10 + 5 + 10

    result = [r async for r in d.iter_records()]
    assert [r["id"] for r in result] == list(range(25))


# ---------------------------------------------------------------------------
# Batch iteration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter_batches_async(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(25)]
    async with d.writer() as w:
        for r in records:
            await w.write(r)

    batches = [b async for b in d.iter_batches(batch_size=10)]
    assert len(batches) == 3  # 10+10+5
    assert sum(len(b) for b in batches) == 25
    # Flatten and verify ordering
    flat = [r for b in batches for r in b]
    assert flat == records


def test_iter_batches_sync(tmp_path):
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(25)]
    with d.writer_sync() as w:
        for r in records:
            w.write(r)

    batches = list(d.iter_batches_sync(batch_size=10))
    assert len(batches) == 3
    assert sum(len(b) for b in batches) == 25


# ---------------------------------------------------------------------------
# Arrow batches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter_arrow_batches_async(tmp_path):
    pa = pytest.importorskip("pyarrow")
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i, "val": f"v{i}"} for i in range(50)]
    async with d.writer(max_records_per_shard=20) as w:
        for r in records:
            await w.write(r)

    batches = [b async for b in d.iter_arrow_batches(batch_size=15)]
    total = sum(b.num_rows for b in batches)
    assert total == 50
    assert all(isinstance(b, pa.RecordBatch) for b in batches)


def test_iter_arrow_batches_sync(tmp_path):
    pytest.importorskip("pyarrow")
    d = JsonlDir(path=str(tmp_path))

    records = [{"id": i} for i in range(50)]
    with d.writer_sync(max_records_per_shard=20) as w:
        for r in records:
            w.write(r)

    batches = list(d.iter_arrow_batches_sync(batch_size=15))
    total = sum(b.num_rows for b in batches)
    assert total == 50
