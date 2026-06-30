"""Tests for JsonlFile, writers, buffer, and helpers."""

from __future__ import annotations

import sys
from unittest.mock import patch

import orjson
import pytest

from flyteplugins.jsonl._jsonl_file import (
    JsonlFile,
    _is_zstd_path,
    _JsonlBuffer,
    _parse_line,
)

# ---------------------------------------------------------------------------
# _is_zstd_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("data.jsonl.zst", True),
        ("data.jsonl.zstd", True),
        ("/tmp/DATA.JSONL.ZST", True),
        ("data.jsonl", False),
        ("data.json", False),
        ("data.zst", False),
    ],
)
def test_is_zstd_path(path, expected):
    assert _is_zstd_path(path) is expected


# ---------------------------------------------------------------------------
# _parse_line
# ---------------------------------------------------------------------------


def test_parse_line_valid():
    result = _parse_line(b'{"a": 1}\n', 1, None)
    assert result == {"a": 1}


def test_parse_line_blank():
    assert _parse_line(b"", 1, None) is None
    assert _parse_line(b"\n", 1, None) is None
    assert _parse_line(b"   \n", 1, None) is None


def test_parse_line_corrupt_raise():
    with pytest.raises(orjson.JSONDecodeError):
        _parse_line(b"not-json\n", 1, None)


def test_parse_line_corrupt_skip():
    errors = []

    def handler(line_no, raw, exc):
        errors.append((line_no, raw))

    result = _parse_line(b"not-json\n", 42, handler)
    assert result is None
    assert len(errors) == 1
    assert errors[0][0] == 42


# ---------------------------------------------------------------------------
# _JsonlBuffer
# ---------------------------------------------------------------------------


def test_buffer_append():
    buf = _JsonlBuffer(flush_bytes=1 << 20)
    needs_flush = buf.append({"x": 1})
    assert not needs_flush
    assert buf.has_data()
    assert b'"x"' in buf.data()


def test_buffer_append_many():
    buf = _JsonlBuffer(flush_bytes=1 << 20)
    needs_flush = buf.append_many([{"a": i} for i in range(5)])
    assert not needs_flush
    lines = bytes(buf.data()).strip().split(b"\n")
    assert len(lines) == 5


def test_buffer_append_raw():
    buf = _JsonlBuffer(flush_bytes=1 << 20)
    raw = orjson.dumps({"r": 1}) + b"\n"
    needs_flush = buf.append_raw(raw)
    assert not needs_flush
    assert buf.data() == bytearray(raw)


def test_buffer_flush_threshold():
    buf = _JsonlBuffer(flush_bytes=10)
    needs_flush = buf.append({"long_key": "a" * 100})
    assert needs_flush


def test_buffer_clear():
    buf = _JsonlBuffer(flush_bytes=1 << 20)
    buf.append({"x": 1})
    assert buf.has_data()
    buf.clear()
    assert not buf.has_data()


# ---------------------------------------------------------------------------
# _resolve_error_handler
# ---------------------------------------------------------------------------


def test_resolve_error_handler_raise():
    assert JsonlFile._resolve_error_handler("raise") is None


def test_resolve_error_handler_skip():
    handler = JsonlFile._resolve_error_handler("skip")
    assert callable(handler)


def test_resolve_error_handler_callable():

    def fn(ln, raw, exc):
        pass

    assert JsonlFile._resolve_error_handler(fn) is fn


def test_resolve_error_handler_invalid():
    with pytest.raises(ValueError, match="on_error must be"):
        JsonlFile._resolve_error_handler(123)


# ---------------------------------------------------------------------------
# Async writer / reader (plain)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_and_read_plain(tmp_path):
    path = str(tmp_path / "test.jsonl")
    f = JsonlFile(path=path)

    records = [{"id": i, "val": f"v{i}"} for i in range(50)]
    async with f.writer() as w:
        for r in records:
            await w.write(r)

    result = [r async for r in f.iter_records()]
    assert result == records


@pytest.mark.asyncio
async def test_write_and_read_compressed(tmp_path):
    path = str(tmp_path / "test.jsonl.zst")
    f = JsonlFile(path=path)

    records = [{"id": i} for i in range(50)]
    async with f.writer() as w:
        for r in records:
            await w.write(r)

    result = [r async for r in f.iter_records()]
    assert result == records


@pytest.mark.asyncio
async def test_write_many(tmp_path):
    path = str(tmp_path / "test.jsonl")
    f = JsonlFile(path=path)

    records = [{"id": i} for i in range(30)]
    async with f.writer() as w:
        await w.write_many(records)

    result = [r async for r in f.iter_records()]
    assert result == records


@pytest.mark.asyncio
async def test_read_empty_file(tmp_path):
    path = str(tmp_path / "empty.jsonl")
    (tmp_path / "empty.jsonl").write_bytes(b"")
    f = JsonlFile(path=path)

    result = [r async for r in f.iter_records()]
    assert result == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_error_skip(tmp_path):
    path = str(tmp_path / "mixed.jsonl")
    (tmp_path / "mixed.jsonl").write_bytes(b'{"a":1}\nnot-json\n{"b":2}\n')
    f = JsonlFile(path=path)

    result = [r async for r in f.iter_records(on_error="skip")]
    assert result == [{"a": 1}, {"b": 2}]


@pytest.mark.asyncio
async def test_on_error_custom(tmp_path):
    path = str(tmp_path / "mixed.jsonl")
    (tmp_path / "mixed.jsonl").write_bytes(b'{"a":1}\nbad\n{"b":2}\n')
    f = JsonlFile(path=path)

    errors = []

    def handler(ln, raw, exc):
        errors.append(ln)

    result = [r async for r in f.iter_records(on_error=handler)]
    assert result == [{"a": 1}, {"b": 2}]
    assert errors == [2]


@pytest.mark.asyncio
async def test_on_error_raise(tmp_path):
    path = str(tmp_path / "bad.jsonl")
    (tmp_path / "bad.jsonl").write_bytes(b"not-json\n")
    f = JsonlFile(path=path)

    with pytest.raises(orjson.JSONDecodeError):
        async for _ in f.iter_records(on_error="raise"):
            pass


# ---------------------------------------------------------------------------
# Sync writer / reader
# ---------------------------------------------------------------------------


def test_write_and_read_sync(tmp_path):
    path = str(tmp_path / "test.jsonl")
    f = JsonlFile(path=path)

    records = [{"id": i} for i in range(50)]
    with f.writer_sync() as w:
        for r in records:
            w.write(r)

    result = list(f.iter_records_sync())
    assert result == records


def test_write_and_read_compressed_sync(tmp_path):
    path = str(tmp_path / "test.jsonl.zst")
    f = JsonlFile(path=path)

    records = [{"id": i} for i in range(50)]
    with f.writer_sync() as w:
        for r in records:
            w.write(r)

    result = list(f.iter_records_sync())
    assert result == records


# ---------------------------------------------------------------------------
# Arrow batches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter_arrow_batches_async(tmp_path):
    pa = pytest.importorskip("pyarrow")
    path = str(tmp_path / "test.jsonl")
    f = JsonlFile(path=path)

    records = [{"id": i, "val": f"v{i}"} for i in range(100)]
    async with f.writer() as w:
        await w.write_many(records)

    batches = [b async for b in f.iter_arrow_batches(batch_size=30)]
    assert len(batches) == 4  # 30+30+30+10
    assert all(isinstance(b, pa.RecordBatch) for b in batches)

    total = sum(b.num_rows for b in batches)
    assert total == 100


def test_iter_arrow_batches_sync(tmp_path):
    pytest.importorskip("pyarrow")
    path = str(tmp_path / "test.jsonl")
    f = JsonlFile(path=path)

    records = [{"id": i} for i in range(100)]
    with f.writer_sync() as w:
        w.write_many(records)

    batches = list(f.iter_arrow_batches_sync(batch_size=30))
    assert len(batches) == 4
    assert sum(b.num_rows for b in batches) == 100


@pytest.mark.asyncio
async def test_iter_arrow_batches_no_pyarrow(tmp_path):
    path = str(tmp_path / "test.jsonl")
    (tmp_path / "test.jsonl").write_bytes(b'{"a":1}\n')
    f = JsonlFile(path=path)

    with patch.dict(sys.modules, {"pyarrow": None}):
        with pytest.raises(ModuleNotFoundError, match="pyarrow is required"):
            async for _ in f.iter_arrow_batches():
                pass
