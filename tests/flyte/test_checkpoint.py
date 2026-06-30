"""Tests flyte._checkpoint"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tarfile
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from flyte._checkpoint import (
    _PAYLOAD_BASENAME,
    CHECKPOINT_CACHE_KEY,
    Checkpoint,
    latest_checkpoint,
)
from flyte.models import ActionID, CheckpointPaths, RawDataPath, TaskContext
from flyte.report import Report
from flyte.storage._parallel_reader import DownloadQueueEmpty


def _write_dir_checkpoint_tar(
    tar_path: pathlib.Path,
    *,
    inner_name: str,
    text: str,
) -> None:
    """Write a gzip tar at *tar_path* with one text member *inner_name*."""
    with tarfile.open(tar_path, "w:gz") as tar:
        data = text.encode("utf-8")
        ti = tarfile.TarInfo(name=inner_name)
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))


def _read_state_step(workspace: pathlib.Path) -> int:
    return json.loads((workspace / "state.json").read_text(encoding="utf-8"))["step"]


def test_checkpoint_prev_exists() -> None:
    cp = Checkpoint("s3://bucket/out", None)
    assert not cp.prev_exists()
    cp2 = Checkpoint("s3://bucket/out", "s3://bucket/prev")
    assert cp2.prev_exists()


def test_latest_checkpoint_prefers_newest_mtime() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        old = root / "a" / "last.ckpt"
        new = root / "b" / "last.ckpt"
        old.parent.mkdir(parents=True)
        new.parent.mkdir(parents=True)
        old.write_text("x", encoding="utf-8")
        new.write_text("y", encoding="utf-8")
        got = latest_checkpoint(root)
        assert got == new


def test_latest_checkpoint_custom_glob() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "a.ckpt").write_text("a", encoding="utf-8")
        assert latest_checkpoint(root, "*.ckpt") == root / "a.ckpt"


def test_latest_checkpoint_custom_sort_key() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        low = root / "ckpt_step0.ckpt"
        high = root / "ckpt_step9.ckpt"
        low.write_text("a", encoding="utf-8")
        high.write_text("b", encoding="utf-8")
        got = latest_checkpoint(root, "*.ckpt", key=lambda p: int(p.stem.split("step")[-1]))
        assert got == high


# --- load_sync / save_sync (file URIs) ---


def test_checkpoint_directory_checkpoint_roundtrip_file_uris() -> None:
    """Directory layout: tar on remote; ``load_sync`` / ``save_sync`` roundtrip via ``file://`` URIs."""
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        prev_blob = base / "prev_sync"
        out_blob = base / "out_sync"
        _write_dir_checkpoint_tar(
            prev_blob,
            inner_name="state.json",
            text=json.dumps({"step": 1}),
        )
        cp = Checkpoint(out_blob.as_uri(), prev_blob.as_uri())
        assert cp.load_sync() is not None
        assert _read_state_step(cp.path) == 1
        (cp.path / "state.json").write_text(json.dumps({"step": 2}), encoding="utf-8")
        cp.save_sync(cp.path)
        cp2 = Checkpoint(base / "out_sync_2", out_blob.as_uri())
        assert cp2.load_sync() is not None
        assert _read_state_step(cp2.path) == 2


def test_checkpoint_tar_roundtrip() -> None:
    """Remote checkpoint is a single tar object; prev/next URIs are concrete file paths."""
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        prev_blob = base / "prev_flytecheckpoints"
        out_blob = base / "out_flytecheckpoints"
        out2_blob = base / "out2_flytecheckpoints"

        _write_dir_checkpoint_tar(
            prev_blob,
            inner_name="state.json",
            text=json.dumps({"step": 3}),
        )

        cp = Checkpoint(out_blob.as_uri(), prev_blob.as_uri())
        assert cp.load_sync() is not None
        state_file = cp.path / "state.json"
        assert state_file.is_file()
        assert _read_state_step(cp.path) == 3

        state_file.write_text(json.dumps({"step": 4}), encoding="utf-8")
        cp.save_sync(cp.path)

        cp2 = Checkpoint(out2_blob.as_uri(), out_blob.as_uri())
        assert cp2.load_sync() is not None
        assert _read_state_step(cp2.path) == 4


def test_checkpoint_single_non_tar_object_roundtrip() -> None:
    """Raw (non-tar) remote object is moved to ``path / payload``; :meth:`load_sync` returns that path."""
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        prev_blob = base / "prev_raw"
        out_blob = base / "out_raw"
        prev_blob.write_bytes(b"hello-checkpoint")

        cp = Checkpoint(out_blob.as_uri(), prev_blob.as_uri())
        payload_path = cp.load_sync()
        assert payload_path is not None
        assert payload_path == cp.path / _PAYLOAD_BASENAME
        assert payload_path.is_file()
        assert payload_path.read_bytes() == b"hello-checkpoint"

        cp.save_sync(b"next")

        cp2 = Checkpoint(base / "unused_out", out_blob.as_uri())
        p2 = cp2.load_sync()
        assert p2 is not None
        assert p2.read_bytes() == b"next"


# --- load / save (async) ---


@pytest.mark.asyncio
async def test_checkpoint_load_treats_empty_remote_as_no_checkpoint() -> None:
    """Download with zero bytes or missing listing is recoverable on load."""
    with patch("flyte.storage.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = DownloadQueueEmpty()
        cp = Checkpoint("s3://bucket/out", "s3://bucket/prev_checkpoint_file")
        restored = await cp.load()
        assert restored is None


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.TaskGroup uses ExceptionGroup on 3.11+")
@pytest.mark.asyncio
async def test_checkpoint_load_treats_exception_group_download_queue_empty() -> None:
    """Obstore uses TaskGroup; empty listing becomes ExceptionGroup(DownloadQueueEmpty(...))."""
    from builtins import BaseExceptionGroup

    eg = BaseExceptionGroup("unhandled errors in a TaskGroup", [DownloadQueueEmpty()])
    with patch("flyte.storage.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = eg
        cp = Checkpoint("s3://bucket/out", "s3://bucket/prev_checkpoint_file")
        restored = await cp.load()
        assert restored is None


def test_checkpoint_load_sync_treats_sync_get_errors_as_no_checkpoint() -> None:
    with patch("flyte.io._file.File.download_sync", side_effect=DownloadQueueEmpty()):
        cp = Checkpoint("s3://bucket/out", "s3://bucket/prev_checkpoint_file")
        assert cp.load_sync() is None


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.TaskGroup uses ExceptionGroup on 3.11+")
def test_checkpoint_load_sync_treats_exception_group_download_queue_empty() -> None:
    from builtins import BaseExceptionGroup

    eg = BaseExceptionGroup("unhandled errors in a TaskGroup", [DownloadQueueEmpty()])
    with patch("flyte.io._file.File.download_sync", side_effect=eg):
        cp = Checkpoint("s3://bucket/out", "s3://bucket/prev_checkpoint_file")
        assert cp.load_sync() is None


@pytest.mark.asyncio
async def test_checkpoint_load_save_async_on_task_loop() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        remote_out = base / "checkpoint_out"
        remote_prev = base / "checkpoint_prev"
        _write_dir_checkpoint_tar(
            remote_prev,
            inner_name="state.json",
            text=json.dumps({"step": 7}),
        )
        cp = Checkpoint(remote_out.as_uri(), remote_prev.as_uri())
        assert await cp.load() is not None
        assert _read_state_step(cp.path) == 7
        await cp.save(cp.path / "state.json")


def test_checkpoint_save_empty_directory_uploads_tar() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        out_blob = base / "empty_dir_chkpt"
        cp = Checkpoint(out_blob.as_uri(), None)
        cp.save_sync(cp.path)

        cp2 = Checkpoint(base / "out2", out_blob.as_uri())
        assert cp2.load_sync() is not None
        assert list(cp2.path.iterdir()) == []


# --- TaskContext.checkpoint ---


def test_task_context_checkpoint_property_cached() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        out = (base / "out").as_uri()
        prev = (base / "prev").as_uri()
        tctx = TaskContext(
            action=ActionID(name="a0"),
            version="v1",
            raw_data_path=RawDataPath(path=str(base)),
            output_path=str(base / "outputs"),
            run_base_dir=str(base),
            report=Report(name="t"),
            checkpoint_paths=CheckpointPaths(checkpoint_path=out, prev_checkpoint_path=prev),
        )
        assert tctx.checkpoint is not None
        assert tctx.checkpoint is tctx.checkpoint
        assert CHECKPOINT_CACHE_KEY in tctx.data
        assert isinstance(tctx.data[CHECKPOINT_CACHE_KEY], Checkpoint)


def test_task_context_checkpoint_none_without_prefix() -> None:
    tctx = TaskContext(
        action=ActionID(name="a0"),
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/rd"),
        output_path="/tmp/o",
        run_base_dir="/tmp",
        report=Report(name="t"),
        checkpoint_paths=None,
    )
    assert tctx.checkpoint is None


def test_task_context_checkpoint_none_empty_dest() -> None:
    tctx = TaskContext(
        action=ActionID(name="a0"),
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/rd"),
        output_path="/tmp/o",
        run_base_dir="/tmp",
        report=Report(name="t"),
        checkpoint_paths=CheckpointPaths(checkpoint_path="  ", prev_checkpoint_path=None),
    )
    assert tctx.checkpoint is None
